import logging

import torch
from torch import nn
import math
import numpy as np
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
        self.optimizer_state_dict = dict()
        self.lr_scheduler_state_dict = dict()
        self.forgetting_stats = None

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
        if args.pruning in ["FedTiny", "FedDST", "FedMem"]:
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
            raise Exception("Support FedTiny, FedDST, FedMem Only!!!")

    # def init_prune_loop(self, args, device, train_data):
    #     density = args.density
    #     pass

    def get_top_k_grad(self, data_sample, device, k_dict, model):
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        x, label = data_sample
        x = x.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        x, label = x.to(device), label.to(device)
        model.zero_grad()
        log_probs = model(x)
        loss = criterion(log_probs, label)
        loss.backward()

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

            # logging.info(f"layer {name} num of remove {num_remove} and num of total {len(idx)}")
            idx = idx[:num_remove]
            idx = [x.item() for x in idx]
            grad = grad.flatten()[idx]
            # must use this to avoid the error
            grad = [x.item() for x in grad]

            # self.candidate_set[name] = dict(zip(idx, grad))
            # use list to reduce the key usage.
            top_k[name] = list(zip(idx, grad))

        model.zero_grad()
        return top_k
    def train(self, train_data, device, args, mode=0):
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        if len(self.optimizer_state_dict) != 0:
            optimizer.load_state_dict(self.optimizer_state_dict)

        if len(self.lr_scheduler_state_dict) != 0:
            lr_scheduler.load_state_dict(self.lr_scheduler_state_dict)

        if mode in [1, 2, 3, 4]:
            self.mask.optimizer = optimizer
            self.mask.to_module_device_()
            self.mask.apply_mask()

        for epoch in range(args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                if mode == 4:
                    for name, weight in self.model.named_parameters():
                        if name not in self.penalty_index:
                            continue
                        # loss += 0.01 * torch.norm(weight.flatten()[self.penalty_index[name]])
                        loss += args.lam * torch.norm(weight.flatten()[self.penalty_index[name]], p=args.p)

                loss.backward()

                if mode in [1, 2, 3, 4]:
                    self.mask.step()
                else:
                    optimizer.step()
            lr_scheduler.step()

        if mode == 2:
            assert len(self.num_growth) != 0
            # here needs to be verified !!!!!!
            random_index =  int(np.random.random()*len(train_data.dataset))
            data_sample = train_data.dataset[random_index]
            self.candidate_set = self.get_top_k_grad(data_sample, device, self.num_growth, model)

        self.optimizer_state_dict = optimizer.state_dict()
        self.lr_scheduler_state_dict = lr_scheduler.state_dict()

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

    def cal_forgetting_stats(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(train_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)

                _, predicted = torch.max(pred, -1)
                incorrect = predicted.neq(target)
                # this may have bug
                self.forgetting_stats[batch_idx*args.batch_size: (batch_idx+1)*args.batch_size][incorrect] += 1


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
