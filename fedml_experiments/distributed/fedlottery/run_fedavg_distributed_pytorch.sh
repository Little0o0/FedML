#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
ROUND=$4
EPOCH=$5
BATCH=$6
DATASET=$7
DATA_DIR=$8
PRUNE=$9
ABNS=${10}
SFT=${11}
NUM=${12}
DENSITY=${13}
BASELINE=${14}
SHOTS=${15}
DELTA=${16}
TMAX=${17}
ADJRATE=${18}

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
  --batch_size $BATCH \
  --prune $PRUNE \
  --ABNS $ABNS \
  --SFt $SFT \
  --num_candidates $NUM \
  --density $DENSITY \
  --baseline $BASELINE \
  --n_shots $SHOTS \
  --delta_epochs $DELTA \
  --T_max $TMAX \
  --adjust_rate $ADJRATE