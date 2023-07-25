import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class FedMemAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.model_candidate_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    # def set_baseline_init_prune_model(self):
    #     self.trainer.init_prune_loop(self.args, self.device, self.train_global)
    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def get_coarse_prune_model(self):
        self.trainer.init_mask(args=self.args,)

    def get_mask_dict(self):
        return self.trainer.mask_dict

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, candidate_set):
        logging.debug("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        self.model_candidate_dict[index] = candidate_set

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def update_penalty_index(self):
        model_mask = self.trainer.mask
        model_mask_dict = self.trainer.get_model_mask_dict()
        num_growth = self.trainer.get_num_growth()
        for name, weight in model_mask.module.named_parameters():
            if name not in model_mask_dict or len(model_mask_dict[name]) == 0:
                continue
            mask = model_mask_dict[name]
            removed = num_growth[name]

            num_zeros = int((mask.numel() - mask.sum()).cpu().item())
            # if name == "features.15.conv.1.0.weight":
            #     logging.info(f"features.15.conv.1.0.weight has {len(mask)} parameter")
            #     logging.info(f"need remove {num_growth[name]} parameter")
            #     logging.info(f"num of zeros {num_zeros}")
            k = num_zeros + removed
            _, new_idx = torch.sort(torch.abs(weight.cpu().flatten()))
            _, old_idx = torch.sort(mask.cpu().flatten())
            self.trainer.penalty_index[name] = \
                torch.tensor(list(set(new_idx[:k].numpy()) - set(old_idx[:num_zeros].numpy())))

    def grow_and_record_prune_idx(self, round, training_num):
        for idx in self.model_candidate_dict:
            for name in self.model_candidate_dict[idx]:
                self.model_candidate_dict[idx][name] = dict(self.model_candidate_dict[idx][name])
        self.trainer.mask.to_module_device_()
        self.trainer.mask.step(self.args.epochs * (round + 1))
        candidate_set = dict()
        candidate_layer_names = list(self.model_candidate_dict[0].keys())
        for name in candidate_layer_names:
            candidate_set[name] = dict()

        for idx in range(self.worker_num):
            w = self.sample_num_dict[idx] / training_num
            for name in candidate_set:
                for index, grad in self.model_candidate_dict[idx][name].items():
                    tensor_grad = torch.tensor(grad)
                    if index in candidate_set[name]:
                        candidate_set[name][index] += tensor_grad * w
                    else:
                        candidate_set[name][index] = tensor_grad * w

        model_mask = self.trainer.mask
        model_mask_dict = self.trainer.get_model_mask_dict()
        model_mask.gather_statistics()
        model_mask.adjust_prune_rate()
        # total_nonzero_new = 0
        num_growth = self.trainer.get_num_growth()
        for name, weight in model_mask.module.named_parameters():
            if name not in model_mask_dict or name not in candidate_set:
                continue
            mask = model_mask_dict[name]
            removed = num_growth[name]
            num_zeros = int((mask.numel() - mask.sum()).cpu().item())
            k = num_zeros + removed
            _, new_idx = torch.sort(torch.abs(weight.cpu().flatten()))
            _, old_idx = torch.sort(mask.cpu().flatten())
            self.trainer.penalty_index[name] = \
                    torch.tensor(list(set(new_idx[:k].numpy()) - set(old_idx[:num_zeros].numpy())))

            logging.info(f"layer {name} record {len(self.trainer.penalty_index[name])} parameters that need to be pruned, the overall is {mask.numel()}")
            regrowth = sorted(candidate_set[name].items(), key=lambda x: torch.abs(x[1]), reverse=True)[:removed]
            regrowth_index = [x[0] for x in regrowth]

            model_mask_dict[name].data.view(-1)[regrowth_index] = 1.0
            weight.data.view(-1)[regrowth_index] = 0.0

            # new_nonzero = mask.sum().item()
            # total_nonzero_new += new_nonzero
            # logging.info(f"layer {name} regrow {removed}, density increase {removed / weight.numel()}")

        self.trainer.set_model_mask_dict(model_mask_dict)
        self.trainer.mask.apply_mask()

        # logging.debug("############## After Growing ##########")
        # model_mask.calculate_density()

        # model_mask.adjustments.append(
        #     model_mask.baseline_nonzero - total_nonzero_new
        # )  # will be zero if deterministic
        # model_mask.adjusted_growth = (
        #         0.25 * model_mask.adjusted_growth
        #         + (0.75 * model_mask.adjustments[-1])
        #         + np.mean(model_mask.adjustments)
        # )
        # if model_mask.stats.total_nonzero > 0:
        #     logging.debug(
        #         f"Nonzero before/after: {model_mask.stats.total_nonzero}/{total_nonzero_new}. "
        #         f"Growth adjustment: {model_mask.adjusted_growth:.2f}."
        #     )
        #
        # model_mask.gather_statistics()

    def prune_penalty_index(self,):
        model_mask = self.trainer.mask
        model_mask_dict = self.trainer.get_model_mask_dict()
        total_nonzero_new = 0
        for name, weight in model_mask.module.named_parameters():
            if name not in model_mask_dict or name not in self.trainer.penalty_index:
                continue

            prune_index = self.trainer.penalty_index[name]
            model_mask_dict[name].data.view(-1)[prune_index] = 0.0
            weight.data.view(-1)[prune_index] = 0.0
            new_nonzero = model_mask_dict[name].sum().item()
            model_mask_dict[name] = model_mask_dict[name].to(self.device)
            total_nonzero_new += new_nonzero

        self.trainer.set_model_mask_dict(model_mask_dict)
        self.trainer.mask.apply_mask()

        model_mask.adjustments.append(
            model_mask.baseline_nonzero - total_nonzero_new
        )  # will be zero if deterministic
        model_mask.adjusted_growth = (
                0.25 * model_mask.adjusted_growth
                + (0.75 * model_mask.adjustments[-1])
                + np.mean(model_mask.adjustments)
        )
        if model_mask.stats.total_nonzero > 0:
            logging.debug(
                f"Nonzero before/after: {model_mask.stats.total_nonzero}/{total_nonzero_new}. "
                f"Growth adjustment: {model_mask.adjusted_growth:.2f}."
            )

        model_mask.gather_statistics()
        # logging.debug("############## After Pruning ##########")
        # model_mask.calculate_density()

    def aggregate_and_prune(self, round, training_num):
        model_mask = self.trainer.mask

        new_mask_dict = dict()
        for idx in self.model_candidate_dict:
            for name in self.model_candidate_dict[idx]:
                if name not in new_mask_dict:
                    new_mask_dict[name] = self.model_candidate_dict[idx][name].to(self.device)
                else:
                    new_mask_dict[name] = torch.logical_or(new_mask_dict[name],
                                self.model_candidate_dict[idx][name].to(self.device)).float()

        # for name in  self.model_candidate_dict[0]:
        #     logging.info(name)
        #     logging.info("before agg")
        #     logging.info(torch.sum(new_mask_dict[name]))
        #     logging.info(torch.sum(self.model_candidate_dict[0][name]))
        #     logging.info("all")
        #     logging.info(torch.numel(new_mask_dict[name]))

        num_remove = dict()
        for name, mask in self.trainer.get_model_mask_dict().items():
            num_remove[name] = int((torch.sum(new_mask_dict[name]) -
                            torch.sum(self.model_candidate_dict[0][name])).item())

        total_nonzero_new = 0
        for name, weight in model_mask.module.named_parameters():
            if name not in num_remove or num_remove[name] == 0 or len(new_mask_dict[name]) == 0:
                continue

            removed = num_remove[name]
            mask = new_mask_dict[name]
            num_zeros = int((mask.numel() - mask.sum()).cpu().item())
            k = num_zeros + removed
            _, new_idx = torch.sort(torch.abs(weight.cpu().flatten()))
            _, old_idx = torch.sort(mask.cpu().flatten())
            prune_index = torch.tensor(list(set(new_idx[:k].numpy()) - set(old_idx[:num_zeros].numpy())))
            # if logging.info(len(prune_index) == 0):
            #     logging.info(name)
            #     logging.info(removed)
            #     logging.info(num_zeros)
            #     continue
            new_mask_dict[name].data.view(-1)[prune_index] = 0.0

            new_nonzero = new_mask_dict[name].sum().item()
            total_nonzero_new += new_nonzero

        self.trainer.set_model_mask_dict(new_mask_dict)
        self.trainer.mask.apply_mask()

        model_mask.adjustments.append(
            model_mask.baseline_nonzero - total_nonzero_new
        )  # will be zero if deterministic
        model_mask.adjusted_growth = (
                0.25 * model_mask.adjusted_growth
                + (0.75 * model_mask.adjustments[-1])
                + np.mean(model_mask.adjustments)
        )
        if model_mask.stats.total_nonzero > 0:
            logging.debug(
                f"Nonzero before/after: {model_mask.stats.total_nonzero}/{total_nonzero_new}. "
                f"Growth adjustment: {model_mask.adjusted_growth:.2f}."
            )

        model_mask.gather_statistics()

    def prune_and_grow(self, round, training_num):
        for idx in self.model_candidate_dict:
            for name in self.model_candidate_dict[idx]:
                self.model_candidate_dict[idx][name] = dict(self.model_candidate_dict[idx][name])
        self.trainer.mask.to_module_device_()
        self.trainer.mask.step(self.args.epochs * (round + 1))
        candidate_set = dict()
        candidate_layer_names = list(self.model_candidate_dict[0].keys())
        for name in candidate_layer_names:
            candidate_set[name] = dict()

        for idx in range(self.worker_num):
            w = self.sample_num_dict[idx] / training_num
            for name in candidate_set:
                for index, grad in self.model_candidate_dict[idx][name].items():
                    tensor_grad = torch.tensor(grad)
                    if index in candidate_set[name]:
                        candidate_set[name][index] += tensor_grad * w
                    else:
                        candidate_set[name][index] = tensor_grad * w

        model_mask = self.trainer.mask
        model_mask_dict = self.trainer.get_model_mask_dict()
        model_mask.gather_statistics()
        model_mask.adjust_prune_rate()

        num_growth = self.trainer.get_num_growth()
        total_nonzero_new = 0
        for name, weight in model_mask.module.named_parameters():
            if name not in model_mask_dict or name not in candidate_set or len(model_mask_dict[name]) == 0:
                continue
            mask = model_mask_dict[name]
            new_mask = model_mask.prune_func(model_mask, mask, weight, name)
            removed = num_growth[name]
            model_mask.stats.total_removed += removed
            model_mask.stats.removed_dict[name] = removed

            regrowth = sorted(candidate_set[name].items(), key=lambda x: torch.abs(x[1]), reverse=True)[:removed]
            regrowth_index = [x[0] for x in regrowth]

            new_mask.data.view(-1)[regrowth_index] = 1.0
            weight.data.view(-1)[regrowth_index] = 0.0
            new_nonzero = new_mask.sum().item()
            model_mask_dict.pop(name)
            model_mask_dict[name] = new_mask.float()
            total_nonzero_new += new_nonzero

        self.trainer.set_model_mask_dict(model_mask_dict)
        self.trainer.mask.apply_mask()

        model_mask.adjustments.append(
            model_mask.baseline_nonzero - total_nonzero_new
        )  # will be zero if deterministic
        model_mask.adjusted_growth = (
                0.25 * model_mask.adjusted_growth
                + (0.75 * model_mask.adjustments[-1])
                + np.mean(model_mask.adjustments)
        )
        if model_mask.stats.total_nonzero > 0:
            logging.debug(
                f"Nonzero before/after: {model_mask.stats.total_nonzero}/{total_nonzero_new}. "
                f"Growth adjustment: {model_mask.adjusted_growth:.2f}."
            )

        model_mask.gather_statistics()

    def aggregate(self, round, mode):
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            if mode != 0 and "mask" in k:
                continue
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        if mode == 2:
            if self.args.pruning == "FedTiny":
                self.prune_and_grow(round, training_num)
            elif self.args.pruning == "FedDST":
                self.aggregate_and_prune(round, training_num)
            elif self.args.pruning == "FedMem":
                self.grow_and_record_prune_idx(round, training_num)
            elif self.args.pruning == "FedMem_v2":
                self.prune_penalty_index()
                self.grow_and_record_prune_idx(round, training_num)

        elif mode == 4:
            if len(self.trainer.penalty_index) > 0:
                logging.info(f"penalty sum is {self.trainer.calculate_penalty()}")

            if self.args.pruning == "FedMem":
                if round % self.args.delta_epochs == self.args.transfer_epochs:
                    self.prune_penalty_index()
                if round % self.args.delta_epochs > self.args.transfer_epochs:
                    assert Exception("Bug here !")

            elif self.args.pruning == "FedMem_v2":
                pass

        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.debug("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # train_num_samples = []
            # train_tot_corrects = []
            # train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
            #     train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))
            #
            #     """
            #     Note: CI environment is CPU-based computing.
            #     The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            #     """
            #     if self.args.ci == 1:
            #         break
            #
            # # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {'training_acc': train_acc, 'training_loss': train_loss}
            # logging.info(stats)
            #
            # # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            metrics = self.trainer.test(self.val_global, self.device, self.args)

            # if round_idx == self.args.comm_round - 1:
            #     metrics = self.trainer.test(self.test_global, self.device, self.args)
            # else:
            #     metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
