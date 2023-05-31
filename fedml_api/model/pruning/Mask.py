import logging

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from fedml_api.model.pruning.Pruning_method import registry as init_registry

class Masking(object):
    def __init__(self, density, init_prune="random", prune_mode="magnitude", growth_mode="gradient_magnitude"):
        self.init_prune = init_prune
        self.prune_mode = prune_mode
        self.growth_mode = growth_mode
        self.density = density
        self.prune_rate_decay = None

        # For Stats
        self.baseline_nonzero = 0
        self.total_params = 0

        # attached module (link to the dense model)
        self.module = None
        self.mask_dict = field(default_factory=dict)

    def generate_mask(self, module):
        self.module = module
        self.baseline_nonzero = 0
        self.total_params = 0
        for name, weight in module.named_parameters():
            self.mask_dict[name] = torch.zeros_like(
                weight, dtype=torch.float32, requires_grad=False )

        self.remove_weight_partial_name("bias")
        self.remove_type(nn.BatchNorm2d)
        self.remove_type(nn.BatchNorm1d)

        self.sparsify()
        self.to_module_device_()
        return self.mask_dict

    @torch.no_grad()
    def apply_mask(self) -> object:
        """
        Applies boolean mask to modules
        """
        for name, weight in self.module.named_parameters():
            if name in self.mask_dict:
                weight.data = weight.data * self.mask_dict[name]


    def sparsify(self, **kwargs):
        """
        Call sparsity init func
        (see sparselearning/funcs/init_scheme.py)
        """
        init_registry[self.sparse_init](self, **kwargs)

    def to_module_device_(self):
        """
        Send to module's device
        """
        for name, weight in self.module.named_parameters():
            if name in self.mask_dict:
                device = weight.device
                self.mask_dict[name] = self.mask_dict[name].to(device)

    def remove_type(self, nn_type):
        """
        Remove layer by type (eg: nn.Linear, nn.Conv2d, etc.)

        :param nn_type: type of layer
        :type nn_type: nn.Module
        """
        for name, module in self.module.named_modules():
            if isinstance(module, nn_type):
                self.remove_weight(name)

    def remove_weight_partial_name(self, partial_name: str):
        """
        Remove module by partial name (eg: conv).

        :param partial_name: partial layer name
        :type partial_name: str
        """
        _removed = 0
        for name in list(self.mask_dict.keys()):
            if partial_name in name:
                logging.debug(
                    f"Removing {name} of size {self.mask_dict[name].shape} with {self.mask_dict[name].numel()} parameters."
                )
                _removed += 1
                self.mask_dict.pop(name)

        logging.debug(f"Removed {_removed} layers.")

