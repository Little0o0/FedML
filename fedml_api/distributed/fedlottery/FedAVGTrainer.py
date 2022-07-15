import logging

from .utils import transform_tensor_to_list


class FedAVGTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.test_data_local_num_dict = {}
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.local_test_number = None
        # self.val_local = None

        self.device = device
        self.args = args

        for key in self.test_data_local_dict:
            self.test_data_local_num_dict[key] = len(self.test_data_local_dict[key].dataset)
        # self.val_data_local_dict, self.val_data_local_num_dict = self.generate_val_data_local_dict(train_data_local_dict, 0.1)



    # def generate_val_data_local_dict(self, dataloader_dict, ratio):
    #     val_data_local_dict = {}
    #     val_data_local_num_dict = {}
    #     for key, dataloader in dataloader_dict.items():
    #         train_data_num = len(dataloader.dataset)
    #         sample_indices = random.sample(range(train_data_num), int(ratio * train_data_num))
    #
    #         # a bug at here, It should use specific Dataset
    #         subset = torch.utils.data.Subset(dataloader.dataset, sample_indices)
    #         sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
    #         val_data_local_dict[key] = sample_testset
    #         val_data_local_num_dict[key] = len(sample_indices)
    #
    #     return val_data_local_dict, val_data_local_num_dict

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def apply_mask(self):
        self.trainer.mask.apply_mask()

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
        self.local_test_number = self.test_data_local_num_dict[client_index]
        # self.val_local = self.val_data_local_dict[client_index]

    def train(self, round_idx = None, mode=0):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args, mode=mode)
        weights = self.trainer.get_model_params(noMask=True)

        candidate_set = dict() if mode != 3 else self.trainer.get_model_candidate_set()
        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, candidate_set

    def test(self):
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        # acc = test_metrics['test_correct'] / test_metrics['test_total']
        return test_metrics # key: ['test_correct', 'test_total', 'test_loss']

    def update_BN(self, epochs=10):
        self.trainer.update_BN_with_local_data(self.test_local, self.device, self.args, epochs)
        weights = self.trainer.get_model_params()
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        BN = {}
        for k in weights.keys():
            if  "running_mean" in k or "running_var" in k:
                BN[k] = weights[k]

        return BN
    # def test(self):
    #     # train data
    #     train_metrics = self.trainer.test(self.train_local, self.device, self.args)
    #     train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
    #                                                       train_metrics['test_total'], train_metrics['test_loss']
    #
    #     # test data
    #     test_metrics = self.trainer.test(self.test_local, self.device, self.args)
    #     test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
    #                                                       test_metrics['test_total'], test_metrics['test_loss']
    #
    #     return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample