import logging
import os
import sys
import torch
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
    from fedml_api.sparselearning.core import Masking
    from fedml_api.sparselearning.funcs.decay import CosineDecay

except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_api.sparselearning.core import Masking
    from FedML.fedml_api.sparselearning.funcs.decay import CosineDecay

from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class FedAVGClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0
        # 0 for fedavg,
        # 1 for training with mask and init with evaluation,
        # 2 for init with update BN,
        # 3 for update mask and training with new mask

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def evaluation(self, lottery_path_dict):
        metrics= {}
        full_model_params = self.trainer.trainer.get_model_params()
        for idx, lottery_path in lottery_path_dict.items():
            mask_dict = torch.load(lottery_path)
            model_params = copy.deepcopy(full_model_params)

            # apply mask
            for name in mask_dict:
                model_params[name] *= mask_dict[name]
            self.trainer.update_model(model_params)
            metrics[idx] = self.trainer.test()
        return metrics

    def ABN_evaluation(self, lottery_path_dict, agg_BNs):
        metrics= {}
        # logging.info(f"lottery_path_dict: {lottery_path_dict}")
        # logging.info(f"agg_BNs: {agg_BNs}")
        full_model_params = self.trainer.trainer.get_model_params()
        for idx, lottery_path in lottery_path_dict.items():
            mask_dict = torch.load(lottery_path)
            model_params = copy.deepcopy(full_model_params)

            # apply mask
            for name in model_params:
                if name in mask_dict:
                    model_params[name] *= mask_dict[name]
                if name in agg_BNs[idx]:
                    model_params[name] = agg_BNs[idx][name]
            self.trainer.update_model(model_params)
            metrics[idx] = self.trainer.test()
        return metrics


    def get_adaptive_BNs(self, lottery_path_dict):
        BNs = {}
        full_model_params = self.trainer.trainer.get_model_params()
        for idx, lottery_path in lottery_path_dict.items():
            mask_dict = torch.load(lottery_path)
            model_params = copy.deepcopy(full_model_params)

            # apply mask
            for name in mask_dict:
                model_params[name] *= mask_dict[name]
            self.trainer.update_model(model_params)
            BNs[idx] = self.trainer.update_BN(epochs = 10)
        return BNs

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        lottery_path_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LOTTERY_PATH_LIST)
        agg_BNs = msg_params.get(MyMessage.MSG_ARG_KEY_AGG_BNS)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        assert self.mode in [0, 1, 2, 4]
        if self.mode == 0:
            self.__train()
        elif self.mode == 1:
            metrics = self.evaluation(lottery_path_dict)
            self.send_init_message_to_server(0, metrics=metrics, local_test_number=self.trainer.local_test_number)
        elif self.mode == 2:
            BNs = self.get_adaptive_BNs(lottery_path_dict)
            self.send_init_message_to_server(0, BNs=copy.deepcopy(BNs), local_test_number=self.trainer.local_test_number)
        elif self.mode == 4:
            metrics = self.ABN_evaluation(lottery_path_dict, agg_BNs)
            self.send_init_message_to_server(0, metrics=metrics, local_test_number=self.trainer.local_test_number)

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE)
        mask_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MASK_DICT)
        num_growth = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_GROWTH)
        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        assert self.mode in [0, 3, 5, 6, 7]

        if self.mode in [0, 6, 7]:
            pass
        elif self.mode == 5:
            if self.trainer.trainer.mask is None:
                init_mask = Masking(
                    None,
                    CosineDecay(prune_rate=self.args.adjust_rate, T_max=self.args.T_max*self.args.epochs),
                    density=self.args.density,
                    prune_mode="magnitude",
                    growth_mode="absolute-gradient",
                    redistribution_mode="none"
                )
                init_mask.mask_dict = mask_dict
                self.trainer.trainer.set_model_mask(init_mask)
            else:
                self.trainer.trainer.set_model_mask_dict(mask_dict)
        elif self.mode == 3:
            self.trainer.trainer.set_num_growth(num_growth)

        self.trainer.update_model(model_params)
        self.__train()

        # if self.round_idx == self.num_rounds - 1:
        #     # post_complete_message_to_sweep_process(self.args)
        #     self.finish()

    def send_init_message_to_server(self, receive_id, metrics={}, BNs={}, local_test_number=0):
        assert self.mode in [1, 2, 4]
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_INIT_MESSAGE, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_METRICS, metrics)
        message.add_params(MyMessage.MSG_ARG_KEY_BNS, BNs)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_TEST_SAMPLES, local_test_number)
        self.send_message(message)

    def send_model_to_server(self, receive_id, weights, local_sample_num, candidate_set):
        # for mode 0, mode 3, mode 5, mode 6
        assert self.mode in [0, 3, 5, 6, 7]
        logging.info(f" round_id = {self.round_idx}, client mode is {self.mode}")
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CANDIDATE_SET, candidate_set)
        self.round_idx += 1
        self.send_message(message)

    def __train(self):
        # logging.info("#######training########### round_id = %d" % self.round_idx)
        noMask = True if self.mode != 7 else False
        weights, local_sample_num, candidate_set = self.trainer.train(round_idx = self.round_idx, mode=self.mode, noMask=noMask)
        # logging.info(f"candidate_set is {candidate_set}")
        self.send_model_to_server(0, weights, local_sample_num, candidate_set)
