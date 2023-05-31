import numpy as np

from .utils import transform_tensor_to_list

class FedMemTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.forgetting_stats_dict = {}
        for k, dataloader in train_data_local_dict.items():
            self.forgetting_stats_dict[k] = np.zeros(len(dataloader.dataset))
        self.forgetting_stats = None

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
        self.forgetting_stats = self.forgetting_stats_dict[client_index]

    def train(self, round_idx=None, mode=0):
        self.args.round_idx = round_idx
        self.forgetting_stats_dict[self.client_index] = self.trainer.train(
            self.train_local, self.device, self.args, mode=mode, forgetting_stats=self.forgetting_stats)

        weights = self.trainer.get_model_params()
        if mode == 2:
            if self.args.pruning in ["FedTiny", "FedMem", "FedMem_v2"]:
                candidate_set = self.trainer.get_model_candidate_set()
            elif self.args.pruning == "FedDST":
                candidate_set = self.trainer.get_model_mask_dict()
        else:
            candidate_set = dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, candidate_set

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample