#!/usr/bin/env bash
#SBATCH --job-name=mmseg_test
#SBATCH --gres=gpu:4           # 请求4个GPU
#SBATCH --nodes=1              # 请求1个节点
#SBATCH --ntasks-per-node=4    # 每个节点的任务数为4
#SBATCH --cpus-per-task=10      # 每个任务使用10个CPU核心
#SBATCH --output=output_test.txt    # 输出文件

conda init
source ~/.bashrc
conda activate SegRet

set -x

JOB_NAME='mmseg_test'
CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
