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
--max_epochs 100 \
--warmup_epochs 10 \
--num-classes 201 \
--crop_size 128 \
--patch_size 8 \
--color_prob 0.0 \
--calculate_val 1 \
--weight_decay 5e-2 \
--finetune 1 \
--lars 0 \
--mixup 0.0 \
--smoothing 0.1 \
--train_file /workspace/data/food/classification/KFOOD201.classification/train \
--val_file /workspace/data/food/classification/KFOOD201.classification/val \
--pretrained_mae_ckpt pretrained_mae_ckpts/vit-mae_losses_0.20102281799793242.pth \
--checkpoints-path checkpoints_finetune \
--log-dir logs