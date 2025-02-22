#!/usr/bin/env bash
#SBATCH --job-name=T_city
#SBATCH --gres=gpu:4           # 请求4个GPU
#SBATCH --nodes=1              # 请求1个节点
#SBATCH --ntasks-per-node=4    # 每个节点的任务数为4
#SBATCH --cpus-per-task=18      # 每个任务使用10个CPU核心
#SBATCH --mem=400G
#SBATCH --output=output_T_sbatch.txt    # 输出文件

set -x
JOB_NAME='mmseg'
GPUS=${GPUS:-4}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-18}
SRUN_ARGS=${SRUN_ARGS:-""}
OUTPUTFILE="output_T.txt"

CONFIG_FILE=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --job-name=${JOB_NAME} \
    --nodes=${NODES}\
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -o ${OUTPUTFILE}\
    -o ${OUTPUTFILE}\
    ${SRUN_ARGS} \
    python -u tools/train.py --config=${CONFIG_FILE}  --launcher="slurm"
