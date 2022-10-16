import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist, load_partition_data_mnist_v2
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.data_preprocessing.Tiny_ImageNet.data_loader import load_partition_data_TinyImageNet_v2
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10, load_partition_data_cifar10_v2
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100, load_partition_data_cifar100_v2
from fedml_api.data_preprocessing.svhn.data_loader import load_partition_data_svhn, load_partition_data_svhn_v2
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10, load_partition_data_cinic10_v2

from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.cv.efficientnet import EfficientNet

from fedml_api.distributed.fedlottery.FedAvgAPI import FedML_init, FedML_FedAvg_distributed

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--model", type=str, default="mobilenet", metavar="N", help="neural network used in training")

    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")

    parser.add_argument("--data_dir", type=str, default="./../../../data/cifar10", help="data directory")

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--client_num_in_total", type=int, default=1000, metavar="NN", help="number of workers in a distributed cluster"
    )

    parser.add_argument("--client_num_per_round", type=int, default=4, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument("--client_optimizer", type=str, default="adam", help="SGD with momentum; adam")

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=5, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--comm_round", type=int, default=10, help="how many round of communications we shoud use")

    parser.add_argument(
        "--is_mobile", type=int, default=0, help="whether the program is running on the FedML-Mobile server side"
    )

    parser.add_argument("--frequency_of_the_test", type=int, default=1, help="the frequency of the algorithms")

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        "--trpc_master_config_path",
        type=str,
        default="trpc_master_config.csv",
        help="config indicating ip address and port of the master (rank 0) node",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument("--pretrain_path", type=str, default="")

    parser.add_argument("--prune", type=int, default=0, help="whether applying pruning")

    parser.add_argument("--density", type=float, default=0.003, help="If applying pruning, model density")

    parser.add_argument("--num_candidates", type=int, default=1, help="the number of candidates for selection")

    parser.add_argument("--ABNS", type=int, default=0, help="whether applying adaptive BN selection")

    parser.add_argument("--SFt", type=int, default=0, help="whether applying sparse fine-tuning")

    parser.add_argument("--init_sparse", type=str, default="noise-uniform-magnitude",
    help="[erdos-renyi-magnitude|uniform-magnitude|noise-erdos-renyi-magnitude-kernel|noise-uniform-magnitude|noise-erdos-renyi-magnitude]")
    # parser.add_argument("--init_sparse", type=str, default="noise-uniform-magnitude",

    parser.add_argument("--baseline", type=str, default="none", help="[SNIP|GraSP|SynFlow|iteratively|PruneFL|Mag|Random|none]")

    parser.add_argument("--n_shots", type=int, default=5, help="the number of shot for pre training and pruning")

    parser.add_argument("--pre_train_epochs", type=int, default=3, help="the number of epochs for pre training")

    parser.add_argument("--init_prune_epochs", type=int, default=5, help="the number of epochs for init pruning methods: [SNIP|GraSP|SynFlow|Mag|Random]")

    parser.add_argument("--grand", type=str, default="entire", help="[entire|block|layer]")

    parser.add_argument("--reverse", type=int, default=1)

    parser.add_argument("--T_max", type=int, default=100)

    parser.add_argument("--adjust_rate", type=float, default=0.1)

    parser.add_argument("--delta_epochs", type=int, default=10)

    parser.add_argument("--feddst", type=int, default=1)

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_mnist(args.batch_size,)
        # ) = load_partition_data_mnist_v2(args.batch_size,
        #                                  args.client_num_in_total,
        #                                  n_shots = args.n_shots,)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_ImageNet(
            dataset=dataset_name,
            data_dir=args.data_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "tiny-imagenet":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.data_dir = os.path.join(args.data_dir, "tiny-imagenet-200")
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_TinyImageNet_v2(
            dataset=dataset_name,
            data_dir=args.data_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
            n_shots = args.n_shots,
        )

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, "mini_gld_train_split.csv")
        fed_test_map_file = os.path.join(args.data_dir, "mini_gld_test.csv")
        args.data_dir = os.path.join(args.data_dir, "images")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, "federated_train.csv")
        fed_test_map_file = os.path.join(args.data_dir, "test.csv")
        args.data_dir = os.path.join(args.data_dir, "images")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    else:
        if dataset_name == "cifar10":
            # data_loader = load_partition_data_cifar10
            data_loader = load_partition_data_cifar10_v2
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100_v2
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10_v2
        elif dataset_name == "svhn":
            data_loader = load_partition_data_svhn_v2
        else:
            data_loader = load_partition_data_cifar10_v2

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = data_loader(
            args.dataset,
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            n_shots=args.n_shots
        )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    # elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
    #     logging.info("ResNet18_GN + Federated_CIFAR100")
    #     model = resnet18()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10004, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("CNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()

    elif model_name == "resnet18":
        if args.baseline == "none":
            from fedml_api.model.cv.fedlottery_model.resnet import resnet18
            model = resnet18(output_dim)
        else:
            from fedml_api.model.cv.pruning_baselines.Models.tinyimagenet_resnet import resnet18
            model = resnet18(output_dim)

    elif model_name == "resnet50":
        if args.baseline == "none":
            from fedml_api.model.cv.fedlottery_model.resnet import resnet50
            model = resnet50(output_dim)
        else:
            from fedml_api.model.cv.pruning_baselines.Models.tinyimagenet_resnet import resnet50
            # logging.info("!!!!!!!!! load the model from baseline architecture !!!!!!")
            model = resnet50(output_dim)

    elif model_name == "vgg16":
        if args.baseline == "none":
            from fedml_api.model.cv.vgg import vgg16
            model = vgg16(num_classes=output_dim)

        else:
            from fedml_api.model.cv.pruning_baselines.Models.tinyimagenet_vgg import vgg16
            model = vgg16(num_classes = output_dim)

    elif model_name == "vgg11":
        if args.baseline == "none":
            from fedml_api.model.cv.vgg import vgg11_bn
            model = vgg11_bn(num_classes=output_dim)
        else:
            from fedml_api.model.cv.pruning_baselines.Models.tinyimagenet_vgg import vgg11_bn
            model = vgg11_bn(num_classes = output_dim)

    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    # TODO
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE", num_classes=output_dim)
    elif model_name == "efficientnet":
        model = EfficientNet()

    return model

if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format=str(process_id) + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        msg = f"_{args.client_num_in_total}|{args.client_num_per_round}_{args.model}_{args.dataset}"
        if args.prune == 0:
            msg = "Dense_D1.0" + msg
        elif args.baseline != "none":
            msg = f"{args.baseline}_D{args.density}" + msg
        else:
            msg = f"ABNS{args.ABNS}_SFt{args.SFt}_D{args.density}_T{args.T_max}_Del{args.delta_epochs}_G{args.grand}_R{args.reverse}" + msg

        wandb.init(
            # project="federated_nas",
            project="FedLottery",
            name=msg,
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key
    )

    # load data
    dataset = load_data(args, args.dataset)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    # start distributed training
    FedML_FedAvg_distributed(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
    )
