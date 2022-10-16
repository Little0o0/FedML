import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        # self.val_global = self._generate_validation_set(n_shots=self.args.n_shots)
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        # self.val_data_local_dict, self.val_data_local_num_dict = self.generate_val_data_local_dict(train_data_local_dict, 0.1)

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.test_num_dict = dict()
        self.metrics_dict = dict()
        self.BN_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.model_candidate_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.dense_param = dict()

    def train(self, epochs):
        self.trainer.train(self.train_global, self.device, self.args, epochs=epochs)

    def apply_mask(self):
        self.trainer.mask.apply_mask()

    def set_baseline_init_prune_model(self, epochs, sparsity=None):
        self.trainer.init_prune_loop(self.train_global, self.device, self.args, epochs=epochs, sparsity=sparsity)

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, candidate_set):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.model_candidate_dict[index] = candidate_set
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_init_message_result(self, index, metrics, BNs, test_num):
        logging.info("get evaluation result from index  %d" % index)
        self.test_num_dict[index] = test_num
        self.metrics_dict[index] = metrics
        self.BN_dict[index] = BNs
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate_BN(self):
        aggregate_BN = {}
        test_num = 0
        for idx in range(self.worker_num):
            test_num += self.test_num_dict[idx]
            for index in self.BN_dict[idx]:
                aggregate_BN[index] = {}
                for key, value in self.BN_dict[idx][index].items():
                    if key not in aggregate_BN[index]:
                        aggregate_BN[index][key] = value * self.test_num_dict[idx]
                    else:
                        aggregate_BN[index][key] += value * self.test_num_dict[idx]

        for index in aggregate_BN:
            for key in aggregate_BN[index]:
                aggregate_BN[index][key] /= test_num

        return aggregate_BN


    def aggregate_evaluation(self):
        test_num = 0
        average_losses= {}
        average_accuracy = {}

        for idx in range(self.worker_num):
            test_num += self.test_num_dict[idx]
            for key in self.metrics_dict[idx]:
                if key in average_losses:
                    average_losses[key] += self.metrics_dict[idx][key]["test_loss"]
                    average_accuracy[key] += self.metrics_dict[idx][key]["test_correct"]
                else:
                    average_losses[key] = self.metrics_dict[idx][key]["test_loss"]
                    average_accuracy[key] = self.metrics_dict[idx][key]["test_correct"]

        for key in average_losses:
            average_losses[key] /= test_num
            average_accuracy[key] /= test_num

        logging.info(average_losses)
        logging.info(average_accuracy)

        best_idx, best_loss = sorted(average_losses.items(), key=lambda x: x[1])[0]
        best_acc = average_accuracy[best_idx]
        return best_idx, best_acc

    def get_mask(self):
        return self.trainer.mask

    def update_num_growth(self):
        self.trainer.update_num_growth()

    def aggregate(self, round, mode):
        start_time = time.time()
        model_list = []
        training_num = 0
        if mode == 3:
            #  because the type of the self.model_candidate_dict[idx][name] is tuple,
            #  so it needs to be converted to dict

            for idx in self.model_candidate_dict:
                for name in self.model_candidate_dict[idx]:
                    self.model_candidate_dict[idx][name] = dict(self.model_candidate_dict[idx][name])

            self.get_mask().to_module_device_()
            self.get_mask().step(self.args.epochs)
            candidate_set = dict()
            layer_names = list(self.model_candidate_dict[0].keys())
            # logging.info(layer_names)
            if self.args.grand == "layer":
                # window_size = 3
                if self.args.reverse:
                    idx = len(layer_names) - (round % len(layer_names))
                else:
                    idx = round % len(layer_names)
                # select one layer for one round
                candidate_layer_names = [layer_names[idx]]
            elif self.args.grand == "block":
                if self.args.model in ["resnet50", "resnet18"]:
                    if self.args.reverse:
                        idx = -(round % 4) + 5
                    else:
                        idx = (round % 4) + 2
                    candidate_layer_names = list(filter(lambda x : f"conv{idx}_x" in x, layer_names))
                    if idx == 5:
                        candidate_layer_names.append('fc.weight')

                elif self.args.model == "vgg11":
                    if self.args.reverse:
                        idx = -(round % 5) + 4
                    else:
                        idx = (round % 5)
                    candidate_layer_names = layer_names[idx*2 : idx*2+2]

                else:
                    raise Exception("TODO yet")

            elif self.args.grand == "entire":
                candidate_layer_names = layer_names

            else:
                raise Exception("TODO yet")

            logging.info("update layer list : " + str(candidate_layer_names))
            for name in candidate_layer_names:
                candidate_set[name] = dict()

            for idx in range(self.worker_num):
                for name in candidate_set:
                    for index, grad in self.model_candidate_dict[idx][name].items():
                        tensor_grad = torch.tensor(grad)
                        if index in candidate_set[name]:
                            candidate_set[name][index].append(tensor_grad)
                        else:
                            candidate_set[name][index] = [tensor_grad]

            # These for average
            # for name in candidate_set:
            #     for index in candidate_set[name]:
            #         candidate_set[name][index] =  torch.mean(torch.stack(candidate_set[name][index]), dim=0)

            for name in candidate_set:
                for index in candidate_set[name]:
                    candidate_set[name][index] = torch.sum(torch.stack(candidate_set[name][index]), dim=0)/self.worker_num



        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            if "mask" in k:
                continue

            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w


        if mode == 7:
            averaged_grad = {}
            for idx, val in self.model_candidate_dict.items():
                local_sample_number = self.sample_num_dict[idx]
                w = local_sample_number / training_num
                for key, gard in val.items():
                    if key in averaged_grad:
                        averaged_grad[key] += w * gard
                    else:
                        averaged_grad[key] = w * gard
            # logging.info(averaged_grad)

            # logging.info(averaged_params.keys())

            for name in averaged_grad:
                if "mask" in name and "weight" in name and \
                        "shortcut" not in name and \
                        "conv1" not in name and \
                        "fc."  not in name and \
                        len(averaged_params[name].size()) != 1:
                    # logging.info(f"mask name is {name}")

                    # prune
                    ratio = self.args.adjust_rate
                    num_connections = int(torch.prod(torch.tensor(averaged_params[name].size())).item())
                    num_ones = int(torch.sum(averaged_params[name]).item())
                    num_zeros = num_connections - num_ones
                    num_adjust = int(num_ones * ratio)
                    k = num_zeros + num_adjust
                    logging.info(f"{name} size is {averaged_params[name].size()}, {num_connections} weights, {num_ones} unpruned, {num_adjust} need to adjust")

                    weight = averaged_params[name.replace("_mask", "")]
                    assert torch.equal(torch.tensor(weight.size()), torch.tensor(averaged_params[name].size()))

                    if weight.dtype == torch.float16:
                        remain_weight = weight * averaged_params[name].half()
                    else:
                        remain_weight = weight * averaged_params[name].float()

                    x, idx = torch.sort(torch.abs(remain_weight.data.view(-1)))

                    averaged_params[name].data.view(-1)[idx[:k]] = 0.0

                    if averaged_grad[name].dtype == torch.float16:
                        averaged_grad[name] = averaged_grad[name] * (averaged_params[name] == 0).half()
                    else:
                        averaged_grad[name] = averaged_grad[name] * (averaged_params[name] == 0).float()

                    y, idx = torch.sort(torch.abs(averaged_grad[name]).flatten(), descending=True)
                    averaged_params[name].data.view(-1)[idx[: int(num_adjust)]] = 1.0
                    # k = int((1.0 - self.args.density) * averaged_grad[name].numel())
                    # threshold, _ = torch.kthvalue(torch.flatten(averaged_grad[name]), k)
                    # zero = torch.tensor([0.])
                    # one = torch.tensor([1.])
                    # averaged_params[name] = torch.where(averaged_grad[name] <= threshold, zero, one)
                    # logging.info(f"{name} size is {averaged_params[name].size()}, left {torch.sum(averaged_params[name])}")

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        # if mode in [3, 5, 6]:
        #     self.trainer.model.to(self.device)
        #     self.apply_mask()

        if mode == 3:
            model_mask = self.get_mask()
            model_mask_dict = self.trainer.get_model_mask_dict()
            model_mask.gather_statistics()
            model_mask.adjust_prune_rate()

            num_growth = self.trainer.get_num_growth()
            # logging.info(num_growth)
            total_nonzero_new = 0
            for name, weight in model_mask.module.named_parameters():
                if name not in model_mask_dict or name not in candidate_set:
                    continue
                mask = model_mask_dict[name]
                num_unpruned_weight = int(mask.sum().item())

                # prune
                if self.args.feddst == 0:
                    new_mask = model_mask.prune_func(model_mask, mask, weight, name)
                    removed = num_growth[name]
                    model_mask.stats.total_removed += removed
                    model_mask.stats.removed_dict[name] = removed

                    # # growth from candidate
                    # if self.args.SFt and self.args.ABNS:
                    #     removed = int(1.2 * removed)

                    regrowth = sorted(candidate_set[name].items(), key=lambda x:torch.abs(x[1]), reverse=True)[:removed]
                    regrowth_index = [x[0] for x in regrowth]

                    new_mask.data.view(-1)[regrowth_index] = 1.0
                    weight.data.view(-1)[regrowth_index] = 0.0
                    new_nonzero = new_mask.sum().item()
                    model_mask_dict.pop(name)
                    model_mask_dict[name] = new_mask.float()
                    total_nonzero_new += new_nonzero

                else:
                    # combine all mask
                    # prune
                    new_mask = model_mask.prune_func(model_mask, mask, weight, name)
                    removed = num_growth[name]
                    model_mask.stats.total_removed += removed
                    model_mask.stats.removed_dict[name] = removed

                    # growth
                    regrowth_index = [x[0] for x in candidate_set[name].items()]
                    new_mask.data.view(-1)[regrowth_index] = 1.0
                    tmp_tensor = self.dense_param[name]
                    weight.data.view(-1)[regrowth_index] = tmp_tensor.view(-1)[regrowth_index].to(self.device)
                    # prune
                    num_prune = len(regrowth_index) - removed
                    if num_prune <= 0 :
                        logging.info("num prune error !!!!")
                    else:
                        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                        weight.data.view(-1)[idx[:-num_unpruned_weight]] = 0.0
                        new_mask.data.view(-1)[idx[:-num_unpruned_weight]] = 0.0

                    new_nonzero = new_mask.sum().item()
                    model_mask_dict.pop(name)
                    model_mask_dict[name] = new_mask.float()
                    total_nonzero_new += new_nonzero

                self.trainer.set_model_mask_dict(model_mask_dict)
                self.apply_mask()


            # model_mask.reset_momentum()
            # model_mask.apply_mask_gradients()

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

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        if self.args.feddst == 1:
            model_dict = self.trainer.get_model_mask_dict()
            for name, m in model_dict.items():
                idx = (m.view(-1) == 1).nonzero(as_tuple=True)
                self.dense_param[name].data.view(-1)[idx] = averaged_params[name].data.view(-1)[idx]
        return averaged_params
    # def aggregate(self, round_id, mode):
    #     start_time = time.time()
    #     model_list = []
    #     training_num = 0
    #     if mode == 3:
    #         pass
    #
    #     for idx in range(self.worker_num):
    #         if self.args.is_mobile == 1:
    #             self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
    #         model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
    #         training_num += self.sample_num_dict[idx]
    #
    #     logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
    #
    #     # logging.info("################aggregate: %d" % len(model_list))
    #     (num0, averaged_params) = model_list[0]
    #     for k in averaged_params.keys():
    #         if "mask" in k:
    #             logging.info(k)
    #         for i in range(0, len(model_list)):
    #             local_sample_number, local_model_params = model_list[i]
    #             w = local_sample_number / training_num
    #             if i == 0:
    #                 averaged_params[k] = local_model_params[k] * w
    #             else:
    #                 averaged_params[k] += local_model_params[k] * w
    #
    #     # update the global model which is cached at the server side
    #     self.set_global_model_params(averaged_params)
    #
    #     if mode == 3:
    #         pass
    #
    #     end_time = time.time()
    #     logging.info("aggregate time cost: %d" % (end_time - start_time))
    #     return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000, n_shots=1):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global
        # elif self.args.dataset == "cifar10":
        #     # implement val split
        #     sample_indices = []
        #     label_set = set(self.test_global.dataset.target)
        #     for label in label_set:
        #          sample_indices += random.sample(np.where(label == self.test_global.dataset.target), n_shots)
        #     subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        #     sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        #     return sample_testset
        # else:
        #     raise Exception(f"Did not implement data split for {self.args.dataset} dataset")


    def val_on_server_for_all_clients(self):
        # this evaluates the model on all clients with their val data set,
        # For simplicity, the eval process will complete on server.
        pass


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

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            metrics = self.trainer.test(self.test_global, self.device, self.args)
            # if round_idx == self.args.comm_round - 1:
            #     metrics = self.trainer.test(self.test_global,s self.device, self.args)
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
