## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.


## Usage
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet50 100 20 cifar10 "./../../../data/cifar10" 1 0 0 10 0.01 none
```

## Setting ip configurations for grpc
```
1. create .csv file in the format:

    receiver_id,ip
    0,<ip_0>
    ...
    n,<ip_n>
    
    where n = client_num_per_round

2. provide path to file as argument to --grpc_ipconfig_path
```

## Running using TRPC
In order to run using TRPC set master's address and port in file trpc_master_config.csv, and use TRPC as backend option.



