import logging
import os, signal
import sys
import torch

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager

except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager



class FedMemServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0
        # mode 0, server aggregate dense model, send aggregated dense model
        # mode 1, server aggregate sparse model, send sparse model
        # mode 2, server  mask aggregate sparse model, update mask, send sparse model and mask
        # mode 3, server

    def run(self):
        super().run()

    def coarse_prune(self, ):
        self.aggregator.get_coarse_prune_model()
        self.mode = 1
    # def init_prune(self):
    #     self.aggregator.set_baseline_init_prune_model()

    def send_init_msg(self):
        # sampling clients
        logging.info(f"init mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        mask_dict = self.aggregator.get_mask_dict() if self.mode == 1 else None
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id,
                        global_model_params,
                        client_indexes[process_id - 1],
                        self.mode,
                        mask_dict)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        candidate_set = msg_params.get(MyMessage.MSG_ARG_KEY_CANDIDATE_SET)
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number, candidate_set)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            logging.info(f"############current mode is {self.mode} ##############")
            global_model_params = self.aggregator.aggregate(self.round_idx, self.mode)
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            if self.mode == 1 \
                and self.round_idx <= self.args.T_max \
                and self.round_idx != 0 \
                and self.round_idx % self.args.delta_epochs == 0:

                if self.args.pruning in ["FedTiny", "FedMem", "FedDST"]:
                    self.mode = 2
                    self.aggregator.trainer.update_num_growth()
                elif self.args.pruning == "FedMem_v2":
                    self.mode = 4
                    self.aggregator.trainer.update_num_growth()
                    self.aggregator.update_penalty_index()

            elif self.mode == 2:
                if self.args.pruning in ["FedTiny", "FedDST"]:
                    self.mode = 3
                elif self.args.pruning == "FedMem":
                    self.mode = 4
                elif self.args.pruning == "FedMem_v2":
                    self.mode = 4 if self.round_idx <= self.args.T_max else 3

            elif self.mode == 3:
                self.mode = 1

            elif self.mode == 4:
                if self.args.pruning == "FedMem":
                    if self.round_idx % self.args.delta_epochs == self.args.transfer_epochs:
                        self.mode = 3
                elif self.args.pruning == "FedMem_v2":
                    if self.round_idx % self.args.delta_epochs == 0:
                        self.mode = 2
                        self.aggregator.trainer.update_num_growth()
                        self.aggregator.update_penalty_index()

            num_growth = dict() if self.mode != 2 else self.aggregator.trainer.get_num_growth()
            mask_dict = dict() if self.mode not in [3, 4] else self.aggregator.trainer.get_model_mask_dict()
            penalty_index = dict() if self.mode != 4 else self.aggregator.trainer.penalty_index
            # penalty_index = None
            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
            
            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                       client_indexes[receiver_id - 1],
                                                       self.mode, num_growth, mask_dict, penalty_index)

    def send_message_init_config(self, receive_id, global_model_params, client_index, mode, mask_dict):
        # assert  mode in [0, 1, 2]
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE, mode)
        message.add_params(MyMessage.MSG_ARG_KEY_MASK_DICT, mask_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params,
                                          client_index, mode, num_growth, mask_dict,
                                          penalty_index):
        logging.debug("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_MODE, mode)
        message.add_params(MyMessage.MSG_ARG_KEY_MASK_DICT, mask_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_GROWTH, num_growth)
        message.add_params(MyMessage.MSG_ARG_KEY_PENALTY_IDX, penalty_index)
        self.send_message(message)
