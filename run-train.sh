#!/bin/bash
set -uex

# You should copy this data to you local SSD for speed
DATA_DIR="/mnt/nas/Data_Neuro/ADNI3/fs-splits-matched"


# CNN with volumes
# --shape can be on one of "mask", "vol_with_bg", "vol_without_bg"
# --discriminator_net can be one of "resnet", "convnet"
python train.py \
    --train_data "${DATA_DIR}/train.h5" \
    --val_data "${DATA_DIR}/valid.h5" \
    --test_data "${DATA_DIR}/test.h5" \
    --discriminator_net "resnet" \
    --shape "vol_with_bg" \
    --task "clf" \
    --batchsize 64 \
    --epoch 15 \
    --num_classes 3 \
    --tensorboard

# Point Clouds
# --discriminator_net can be one of "pointnet", "pointnet++"
python train.py \
    --train_data "${DATA_DIR}/train.h5" \
    --val_data "${DATA_DIR}/valid.h5" \
    --test_data "${DATA_DIR}/test.h5" \
    --discriminator_net "pointnet" \
    --shape "pointcloud" \
    --task "clf" \
    --batchsize 256 \
    --epoch 15 \
    --num_classes 3 \
    --tensorboard
