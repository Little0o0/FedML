class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_S2C_CANDIDATES = 3

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 4
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_INIT_MESSAGE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_NUM_TEST_SAMPLES = "num_test_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_MODE = "mode"
    MSG_ARG_KEY_LOTTERY_PATH_LIST = "lottery_path_list"
    MSG_ARG_KEY_METRICS = "metrics"
    MSG_ARG_KEY_BNS = "BN_measurements_dict"
    MSG_ARG_KEY_MASK_DICT= "mask_dict"
    MSG_ARG_KEY_NUM_GROWTH = "num_growth"
    MSG_ARG_KEY_CANDIDATE_SET = "model_candidate_set"


