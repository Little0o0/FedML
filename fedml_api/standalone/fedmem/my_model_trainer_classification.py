import logging

import torch
from torch import nn
import math
import numpy as np
import torch.utils.data as data
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
    from fedml_api.sparselearning.core import Masking
    from fedml_api.sparselearning.funcs.decay import CosineDecay

except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    from FedML.fedml_api.sparselearning.core import Masking
    from FedML.fedml_api.sparselearning.funcs.decay import CosineDecay

class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model=model, args=args)
        self.mask = None
        self.mask_dict = None
        self.candidate_set = dict()
        self.num_growth = dict()
        self.penalty_index = dict()

    def get_num_growth(self):
        return self.num_growth

    def calculate_penalty(self):
        penalty = 0
        with torch.no_grad():
            for name, weight in self.model.named_parameters():
                if name not in self.penalty_index:
                    continue
                penalty += torch.norm(weight.flatten()[self.penalty_index[name]], p=1)
        return penalty

    def set_num_growth(self, num_growth):
        self.num_growth = num_growth

    def set_model_mask_dict(self, mask_dict):
        self.mask.mask_dict = mask_dict
        self.mask_dict = self.mask.mask_dict
        self.mask.to_module_device_()

    def get_model_mask_dict(self):
        return self.mask_dict

    def get_model_candidate_set(self):
        return self.candidate_set

    def get_model_params(self):
        params = self.model.cpu().state_dict()
        return {key: params[key] for key in params.keys() if "mask" not in key}

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def update_num_growth(self):
        self.mask.gather_statistics()
        self.mask.adjust_prune_rate()

        for name, weight in self.mask.module.named_parameters():
            if name not in self.mask_dict:
                continue
            else:
                num_remove = math.ceil(
                    self.mask.name2prune_rate[name] * self.mask.stats.nonzeros_dict[name]
                )
                self.num_growth[name] = num_remove

    def init_mask(self, args, mask_dict=None):
        # return mask_dict
        if args.pruning in ["FedTiny", "FedDST", "FedMem", "FedMem_v2", "Mag"]:
            if self.mask is None:
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
                self.mask = Masking(
                    optimizer,
                    CosineDecay(prune_rate=args.adjust_rate, T_max=args.T_max * args.epochs),
                    density=args.density,
                    dense_gradients=False,
                    sparse_init=args.init_sparse,
                    prune_mode="magnitude",
                    growth_mode="absolute-gradient",
                    redistribution_mode="none"
                )
            if mask_dict is None:
                self.mask.add_module(self.model)
                self.mask_dict = self.mask.mask_dict
            else:
                self.mask_dict = mask_dict
                self.mask.attach_model_with_mask_dict(self.model, mask_dict)
        else:
            raise Exception("Support FedTiny, FedDST, FedMem, Mag Only!!!")

    # def init_prune_loop(self, args, device, train_data):
    #     density = args.density
    #     pass


    def get_top_k_grad(self, train_data, device, k_dict, model, args):

        model.train()
        criterion = nn.CrossEntropyLoss().to(device)
        if args.growing_type == "Single":
            random_index = int(np.random.random() * len(train_data.dataset))
            data_sample = train_data.dataset[random_index]
            x, label = data_sample
            x = x.unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            x, label = x.to(device), label.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, label)
            loss.backward()
        else:
            model.zero_grad()
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                if args.growing_type == "Batch":
                    break

        # k_dict is the k for each layer
        top_k = {}
        for name, weight in self.model.named_parameters():
            if name not in self.mask_dict:
                continue

            num_remove = k_dict[name]
            mask_matrix = self.mask_dict[name].data.bool().cpu()

            grad = weight.grad.cpu()
            if grad.dtype == torch.float16:
                grad = grad * (mask_matrix == 0).half()
            else:
                grad = grad * (mask_matrix == 0).float()

            _, idx = torch.sort(torch.abs(grad).flatten(), descending=True)

            # logging.debug(f"layer {name} num of remove {num_remove} and num of total {len(idx)}")
            idx = idx[:num_remove].tolist()
            TopGrad = grad.flatten()[idx]
            # must use this to avoid the error
            TopGrad = TopGrad.tolist()
            # logging.debug(f"layer {name} top {num_remove} grad", TopGrad)
            # logging.info(f"length of layer {name} top {num_remove} grad TopGrad {len(TopGrad)} ")
            # self.candidate_set[name] = dict(zip(idx, grad))
            # use list to reduce the key usage.
            top_k[name] = list(zip(idx, TopGrad))

        model.zero_grad()
        return top_k


    def train(self, train_data, device, args, mode=0, forgetting_stats=None):
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        lr_scheduler.step(epoch=args.round_idx)
        alpha = lr_scheduler.get_lr()[0]
        beta = args.min_lr

        if mode in [1, 2, 3, 4]:
            self.mask.optimizer = optimizer
            self.mask.to_module_device_()
            self.mask.apply_mask()

        for epoch in range(args.epochs):
            if epoch == args.epochs//2 and mode == 2 and args.pruning == "FedDST":
                topk_grad = \
                    self.get_top_k_grad(train_data, device, self.num_growth, model, args)

                for name, weight in model.named_parameters():
                    if name not in self.mask_dict or name not in topk_grad:
                        continue
                    mask = self.mask_dict[name]
                    removed = self.num_growth[name]
                    regrowth_index = [x[0] for x in topk_grad[name]]
                    assert removed == len(regrowth_index)
                    num_zeros = int((mask.numel() - mask.sum()).cpu().item())
                    k = num_zeros + removed

                    _, new_idx = torch.sort(torch.abs(weight.cpu().flatten()))
                    _, old_idx = torch.sort(mask.cpu().flatten())
                    prune_index = torch.tensor(list(set(new_idx[:k].numpy()) - set(old_idx[:num_zeros].numpy())))

                    self.mask_dict[name].data.view(-1)[regrowth_index] = 1.0
                    weight.data.view(-1)[regrowth_index] = 0.0

                    self.mask_dict[name].data.view(-1)[prune_index] = 0.0
                    weight.data.view(-1)[prune_index] = 0.0

                self.mask.mask_dict = self.mask_dict
                self.mask.apply_mask()

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                if mode == 4:
                    penalty = 0
                    for name, weight in self.model.named_parameters():
                        if name not in self.penalty_index:
                            continue
                        # loss += 0.01 * torch.norm(weight.flatten()[self.penalty_index[name]])
                        penalty += args.lam * torch.norm(weight.flatten()[self.penalty_index[name]], p=args.p)
                    loss += penalty

                    if args.budget_training:
                        p = 1 - (args.round_idx % args.transfer_epochs + 1)/ args.transfer_epochs
                        beta = args.budget_scaling * p * torch.sigmoid(penalty.cpu()).item()

                lr = max(alpha, beta)
                # logging.info(f"budgeted aware learnin rate is {lr}")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                loss.backward()

                if mode in [1, 2, 3, 4]:
                    self.mask.step()
                else:
                    optimizer.step()
            # lr_scheduler.step()

        if mode == 2:
            assert len(self.num_growth) != 0
            if args.forgetting_set and forgetting_stats is not None:
                assert len(forgetting_stats) == len(train_data.dataset)

            if args.forgetting_set:
                p = int(0.1 * len(forgetting_stats))
                idx = np.argsort(forgetting_stats)[::-1][:p]
                forgetting_dataset = torch.utils.data.Subset(train_data.dataset, idx)
                forgetting_data = data.DataLoader(dataset=forgetting_dataset,
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
                self.candidate_set = self.get_top_k_grad(forgetting_data,
                     device, self.num_growth, model, args)
            else:
                self.candidate_set = self.get_top_k_grad(train_data,
                            device, self.num_growth, model, args)

        if args.pruning in ["FedMem", "FedMem_v2"] and args.forgetting_set:
            forgetting_stats = self.update_forgetting_set(train_data, device, model, args, forgetting_stats)

        return forgetting_stats

    def update_forgetting_set(self, train_data, device, model, args, forgetting_stats):
        model.eval()
        model.to(device)
        DataLoader = data.DataLoader(dataset=train_data.dataset,
            batch_size=args.batch_size, shuffle=False, drop_last=False)

        incorrect = np.array([])
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(DataLoader):
                x = x.to(device)
                pred = model(x)
                _, predicted = torch.max(pred, -1)
                tmp_np = predicted.cpu().eq(target).flatten().logical_not().int().numpy()
                incorrect = np.concatenate([incorrect, tmp_np] )
        forgetting_stats += incorrect
        return forgetting_stats

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
