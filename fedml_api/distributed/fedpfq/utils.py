import os

import torch
import numpy as np


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def quantization(model_params, bit):
    for k in model_params.keys():
        if bit == 32:
            model_params[k] = model_params[k]
        elif bit == 1:
            model_params[k] = torch.sign(model_params[k])
        else:
            n = float(2 ** bit - 1)
            model_params[k] = torch.round(model_params[k] * n) / n
    return model_params

def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
