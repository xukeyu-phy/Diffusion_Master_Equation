#!/bin/sh 
#SBATCH -p gpu1                # 使用gpu1队列
#SBATCH -N 1                   # 1个节点
#SBATCH --gpus-per-node=1      # 每个节点1个GPU
#SBATCH --cpus-per-gpu=1       # 每个GPU配4个CPU核
#SBATCH -o job.%j.out          # 输出文件
#SBATCH -e job.%j.err          # 错误文件


source /fs1/software/python/3.9_miniconda_4.10.3/etc/profile.d/conda.sh

# 激活 torch 环境
conda activate torch

# 可选：验证环境
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA availability: $(python -c 'import torch; print(torch.cuda.is_available())')"


# timeout 5m nvidia-smi dmon > nvi_1.log &
echo STARTED at `date`
yhrun python case.py
echo FINISHED at `date`

echo "运行完成！"