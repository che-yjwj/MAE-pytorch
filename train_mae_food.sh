#!/bin/bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS

CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node 2 train_mae.py \
--batch_size 16 \
--num_workers 8 \
--lr 1.5e-4 \
--optimizer_name "adamw" \
--cosine 1 \
--max_epochs 300 \
--warmup_epochs 40 \
--num-classes 201 \
--crop_size 128 \
--patch_size 8 \
--color_prob 0.0 \
--calculate_val 0 \
--weight_decay 5e-2 \
--finetune 0 \
--lars 0 \
--mixup 0.0 \
--smoothing 0.0 \
--train_file /workspace/data/food/classification/KFOOD201.classification/train \
--val_file /workspace/data/food/classification/KFOOD201.classification/val \
--checkpoints-path checkpoints \
--log-dir logs