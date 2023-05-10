import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class FedMemClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE)
        mask_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MASK_DICT)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        if self.mode in [0]:
            pass
        elif self.mode == 1:
            self.trainer.trainer.init_mask(self.args, mask_dict)

        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE)
        mask_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MASK_DICT)
        num_growth = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_GROWTH)
        penalty_index = msg_params.get(MyMessage.MSG_ARG_KEY_PENALTY_IDX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))

        if self.mode in [0, 1]:
            pass
        elif self.mode == 2:
            self.trainer.trainer.set_num_growth(num_growth)
        elif self.mode == 3:
            self.trainer.trainer.set_model_mask_dict(mask_dict)
        elif self.mode == 4:
            self.trainer.trainer.penalty_index = penalty_index
            self.trainer.trainer.set_model_mask_dict(mask_dict)

        self.round_idx += 1
        self.__train()
        # if self.round_idx == self.num_rounds - 1:
        #     # post_complete_message_to_sweep_process(self.args)
        #     self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, candidate_set):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CANDIDATE_SET, candidate_set)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num, candidate_set = self.trainer.train(self.round_idx, self.mode)
        self.send_model_to_server(0, weights, local_sample_num, candidate_set)
