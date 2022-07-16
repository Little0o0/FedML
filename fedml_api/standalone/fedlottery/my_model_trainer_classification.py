import logging

import torch
from torch import nn
import numpy as np
import math
import pandas as pd

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
    from fedml_api.initpruning.Pruners.pruners import SNIP, GraSP, SynFlow, Mag, Rand
    from fedml_api.initpruning.Utils.generator import masked_parameters
    from fedml_api.initpruning.Utils.prune import prune_loop
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    from FedML.fedml_api.initpruning.Pruners.pruners import SNIP, GraSP, SynFlow, Mag, Rand
    from FedML.fedml_api.initpruning.Utils.generator import masked_parameters
    from FedML.fedml_api.initpruning.Utils.prune import prune_loop


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model=model, args=args)
        self.mask = None
        self.mask_dict = None
        self.candidate_set = dict()
        self.num_growth = dict()



    def set_model_mask(self, mask):
        self.mask = mask
        self.mask_dict = self.mask.mask_dict
        self.mask.attach_model_with_mask_dict(self.model,  self.mask_dict)

    def set_model_mask_dict(self, mask_dict):
        self.mask.mask_dict = mask_dict
        self.mask_dict = self.mask.mask_dict

    def get_model_mask_dict(self):
        return self.mask_dict

    def get_num_growth(self):
        return self.num_growth

    def set_num_growth(self, num_growth):
        self.num_growth = num_growth

    def get_model_candidate_set(self):
        return self.candidate_set

    def get_model_params(self, noMask=False):
        if noMask:
            params =  self.model.cpu().state_dict()
            return {key: params[key] for key in params.keys() if "mask" not in key}
        else:
            return self.model.cpu().state_dict()

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

    def init_prune_loop(self, prune_data, device, args, epochs, schedule = "exponential", scope="local",
                        reinitialize=False, train_mode=False, shuffle=False, invert=False):
        pruner = None
        self.model.to(device)
        if args.baseline == "SNIP":
            pruner = SNIP(masked_parameters(self.model, 0, 0, 0))
        elif args.baseline == "GraSP":
            pruner = GraSP(masked_parameters(self.model, 0, 0, 0))
        elif args.baseline == "SynFlow":
            pruner = SynFlow(masked_parameters(self.model, 0, 0, 0))
        elif args.baseline == "Mag":
            pruner = Mag(masked_parameters(self.model, 0, 0, 0))
        elif args.baseline == "Random":
            pruner = Rand(masked_parameters(self.model, 0, 0, 0))
        else:
            raise Exception(f"have not implement {args.baseline} yet !!!!")
        sparsity = args.density
        loss = nn.CrossEntropyLoss()
        prune_loop(self.model, loss, pruner, prune_data, device, sparsity,
                   schedule, scope, epochs, reinitialize, train_mode, shuffle, invert)


    def train(self, train_data, device, args, mode=0, epochs=None):
        if epochs is None:
            epochs = args.epochs

        assert mode in [0, 3, 5, 6]

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

        if mode in [3, 5, 6]:
            # following need to revise
            self.mask.optimizer = optimizer
            self.mask.to_module_device_()
            self.mask.apply_mask()
            masking_print_FLOPs = True

        epoch_loss = []

        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                if (mode == 3
                    and (epoch + 1) == args.epochs
                    and (batch_idx + 1) == len(train_data)
                    ):
                    # self.update_connections()

                    for name, weight in self.model.named_parameters():
                        if name not in self.mask_dict:
                            continue

                        num_remove = self.num_growth[name]
                        new_mask = self.mask_dict[name].data.bool().cpu()

                        grad = weight.grad.cpu()
                        if grad.dtype == torch.float16:
                            grad = grad * (new_mask == 0).half()
                        else:
                            grad = grad * (new_mask == 0).float()

                        _, idx = torch.sort(torch.abs(grad).flatten(), descending=True)

                        # logging.info(f"layer {name} num of remove {num_remove} and num of total {len(idx)}")
                        idx = idx[:num_remove]
                        idx = [x.item() for x in idx]
                        grad = grad.flatten()[idx]
                        # must use this to avoid the error
                        grad = [x.item() for x in grad]

                        # self.candidate_set[name] = dict(zip(idx, grad))
                        # use list to reduce the key usage.
                        self.candidate_set[name] = list(zip(idx, grad))

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if mode in [3, 5, 6]:
                    self.mask.step()
                else:
                    optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def update_BN_with_local_data(self, train_data, device, args, num_epoch = 5):
        model = self.model
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        for param in model.parameters():
            param.requires_grad = False

        for _ in range(num_epoch):
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)

        model.zero_grad()
        for param in model.parameters():
            param.requires_grad = True
        return

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
