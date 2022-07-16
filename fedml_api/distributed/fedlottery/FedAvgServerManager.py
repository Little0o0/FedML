import copy
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
    from fedml_api.sparselearning.core import Masking
    from fedml_api.sparselearning.funcs.decay import CosineDecay
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager
    from FedML.fedml_api.sparselearning.core import Masking
    from FedML.fedml_api.sparselearning.funcs.decay import CosineDecay


class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 # mode 0 regular fedavg, mode 1 for candidate pools, mode 2 for ABNS, mode 3 for SFt
        self.lottery_paths = {}
        self.mask = None
        self.agg_BNs = dict()


    def run(self):
        super().run()

    def pre_train(self, epochs):
        logging.info("Begin pretraining !!!!!")
        self.aggregator.train(epochs = epochs)

    def init_prune_model(self, epochs):
        self.aggregator.set_baseline_init_prune_model(epochs)

    def generate_lottery_pool(self, path = "./lottery_pool"):
        if not os.path.exists(path):
            os.mkdir(path)
        pool_name = f"_N{self.args.client_num_in_total}S{self.args.client_num_per_round}_{self.args.model}_{self.args.dataset}"
        pool_name = f"ABNS{self.args.ABNS}_SFt{self.args.SFt}_D{self.args.density}" + pool_name
        pool_path = os.path.join(path, pool_name)
        if not os.path.exists(pool_path):
            os.mkdir(pool_path)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.aggregator.trainer.model.parameters()), lr=self.args.lr)
        self.mask = Masking(
            optimizer,
            CosineDecay(prune_rate=self.args.adjust_rate, T_max=self.args.T_max*self.args.epochs),
            density=self.args.density,
            dense_gradients=False,
            sparse_init=self.args.init_sparse,
            prune_mode="magnitude",
            growth_mode="absolute-gradient",
            redistribution_mode="none"
        )

        for idx in range(self.args.num_candidates):
            # pooling.
            self.mask.__post_init__()
            logging.info(f"generate candidate {idx} ")
            mask_dict = self.mask.generate_mask_only(self.aggregator.trainer.model)
            lottery_name = f"{idx}_{self.args.init_sparse}_D{self.mask.act_density:.4f}.pth"
            lottery_path = os.path.join(pool_path, lottery_name)
            for name in mask_dict:
                mask_dict[name] = mask_dict[name].cpu()

            torch.save(mask_dict, lottery_path)
            self.lottery_paths[idx] = lottery_path
        if self.args.ABNS:
            self.mode = 2
        else:
            self.mode = 1

    def send_init_msg(self):
        # sampling clients
        logging.info(f"current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)

        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params,
                                          client_indexes[process_id - 1], self.mode, self.lottery_paths)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_INIT_MESSAGE,
                                              self.handle_message_receive_init_message_from_client)

    def handle_message_receive_init_message_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        local_test_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_TEST_SAMPLES)
        BNs = msg_params.get(MyMessage.MSG_ARG_KEY_BNS)
        metrics = msg_params.get(MyMessage.MSG_ARG_KEY_METRICS)
        self.aggregator.add_local_init_message_result(sender_id - 1, metrics=metrics, BNs=BNs, test_num=local_test_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        assert self.mode in [1, 2, 4]
        if b_all_received:
            if self.mode in [1, 4]:
                best_idx, best_accuracy = self.aggregator.aggregate_evaluation()
                logging.info(f"The best model is idx {best_idx}, whose accuracy is {best_accuracy:.4f}")
                best_mask_dict = torch.load(self.lottery_paths[best_idx])
                self.mask.mask_dict = best_mask_dict
                self.aggregator.trainer.set_model_mask(self.mask)
                global_model_params = self.aggregator.get_global_model_params()

                # update BN
                if self.mode == 4:
                    best_BN = self.agg_BNs[best_idx]
                    for name in best_BN:
                        global_model_params[name] =  best_BN[name]

                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
                # change to mode 5
                self.mode = 5
                if self.args.is_mobile == 1:
                    global_model_params = transform_tensor_to_list(global_model_params)

                logging.info(f"current mode is {self.mode}")
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                           client_indexes[receiver_id - 1], self.mode, best_mask_dict)

            elif self.mode == 2:
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
                global_model_params = self.aggregator.get_global_model_params()

                self.agg_BNs = self.aggregator.aggregate_BN()

                # change to mode 4
                self.mode = 4
                logging.info(f"current mode is {self.mode}")
                for receiver_id in range(1, self.size):
                    self.send_message_init_config(receiver_id, global_model_params, client_indexes[receiver_id - 1],
                                                  mode=self.mode, lottery_paths=self.lottery_paths,
                                                  agg_BNs=self.agg_BNs)


    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        candidate_set = msg_params.get(MyMessage.MSG_ARG_KEY_CANDIDATE_SET)
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number, candidate_set)

        b_all_received = self.aggregator.check_whether_all_receive()
        # logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate(self.round_idx, self.mode)
            # for name in self.aggregator.trainer.get_model_mask_dict():
            #     logging.info(global_model_params[name])
            #     break
            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            if self.mode == 5:
                self.mode = 6

            elif self.mode == 3:
                # generate new mask.
                self.mode = 5

            elif self.mode == 6 and self.args.SFt :
                if self.args.grand == "entire" and self.round_idx % self.args.delta_epochs == 0 :
                    self.mode = 3
                    self.aggregator.update_num_growth()
                elif self.args.grand == "block" and self.round_idx % self.args.delta_epochs < 4:
                    self.mode = 3
                    self.aggregator.update_num_growth()
                elif self.args.grand == "layer":
                    self.mode = 3
                    self.aggregator.update_num_growth()
            else:
                pass

            num_growth = dict() if self.mode != 3 else self.aggregator.trainer.get_num_growth()
            mask_dict = dict() if self.mode != 5 else self.aggregator.trainer.get_model_mask_dict()

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                # self.aggregator.test_on_server_for_all_clients(self.round_idx)
                self.finish()
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
                                                                 self.args.client_num_per_round, )
            
            # print('indexes of clients: ' + str(client_indexes))
            # print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            logging.info(f"current mode is {self.mode}")
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                       client_indexes[receiver_id - 1],
                                                       mode = self.mode,
                                                       mask_dict=mask_dict,
                                                       num_growth=num_growth)


    def send_message_init_config(self, receive_id, global_model_params, client_index, mode, lottery_paths={}, agg_BNs={}):
        assert mode in [0, 1, 2, 4]
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE, mode)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_AGG_BNS, agg_BNs)
        message.add_params(MyMessage.MSG_ARG_KEY_LOTTERY_PATH_LIST, lottery_paths)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, mode,
                                          mask_dict={}, num_growth={}):
        assert mode in [0, 3, 5, 6]
        # logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE, mode)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_MASK_DICT, mask_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_GROWTH, num_growth)
        self.send_message(message)
