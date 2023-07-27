#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
ROUND=$4
EPOCH=$5
DATASET=$6
DATA_DIR=$7
PRUNE=$8
DENSITY=$9
BUDGET=${10}
FORGET=${11}
ACT=${12}
DROPIT=${13}


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_default" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --pruning $PRUNE \
  --density $DENSITY \
  --forgetting_set $FORGET \
  --act_scaling $ACT \
  --budget_training $BUDGET \
  --dropit $DROPIT