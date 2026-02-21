#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --job-name=dm_siglip_recap_0.8
#SBATCH --partition=partition_name   # Partition to run on

scontrol show job "$SLURM_JOB_ID"

source ~/.bashrc

cd /path/to/cabs
source .env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=12
export MASTER_PORT=29500
export PYTHONPATH="$PYTHONPATH:$PWD/src"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/open_clip_train/main.py \
    --report-to wandb \
    --wandb-project-name "cabs" \
    --save-frequency 1 \
    --train-num-samples 128000000 \
    --dataset-type "webdataset" \
    --train-data "/path/to/webdataset/shards/{00000..20000}.tar" \
    --batch-size 5120 \
    --accum-freq 1 \
    --precision amp \
    --workers 8 \
    --epochs 5 \
    --log-every-n-steps 50 \
    --model ViT-B-16-SigLIP-256 \
    --siglip \
    --captions "recap" \
    --which-sampling "filter" \
    --cabs-dm \
    --filter-ratio 0.8 \
    --name "ViT-B-16-SigLIP-256_cabs-dm_recap_0.8" \
    --seed 0 \
    --local-loss \
    --grad-checkpointing \
    --wd 0.2 \
    --gather-with-grad \
    --resume "latest"
