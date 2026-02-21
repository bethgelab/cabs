#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --job-name=fm_alt_0.8
#SBATCH --partition=partition_name   # Partition to run on

scontrol show job "$SLURM_JOB_ID"

source ~/.bashrc

cd /path/to/cabs
source .env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
export PYTHONPATH="$PYTHONPATH:$PWD/src"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/open_clip_train/main.py \
    --report-to wandb \
    --save-frequency 1 \
    --train-num-samples 128000000 \
    --dataset-type "webdataset" \
    --train-data "/path/to/webdataset/shards/{00000..20000}.tar" \
    --warmup 500 \
    --batch-size 5120 \
    --accum-freq 1 \
    --epochs 5 \
    --cabs-freq \
    --precision amp \
    --workers 8 \
    --filter-ratio 0.8 \
    --model ViT-B-32 \
    --captions "alt" \
    --which-sampling "filter" \
    --name "ViT-B-32_cabs-fm_alt_0.8" \
    --seed 0 \
    --local-loss \
    --grad-checkpointing \
    --wd 0.2 \
    --resume "latest" \
    --gather-with-grad
