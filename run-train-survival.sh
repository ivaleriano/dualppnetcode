#!/bin/bash
set -uex

# You should copy this data to you local SSD for speed
DATA_DIR="/mnt/nas/Data_Neuro/shape_continuum/ADNI/survival/cv-matched/fsl"

# CNN with volumes
# --shape can be on one of "mask", "vol_with_bg", "vol_without_bg"
# --discriminator_net can be one of "resnet", "convnet"
python train.py \
    --train_data "${DATA_DIR}/0-train.h5" \
    --val_data "${DATA_DIR}/0-valid.h5" \
    --test_data "${DATA_DIR}/0-valid.h5" \
    --discriminator_net "resnet" \
    --shape "vol_with_bg" \
    --task "surv" \
    --batchsize 64 \
    --epoch 5 \
    --num_classes 1 \
    --tensorboard

# Point Clouds
# --discriminator_net can be one of "pointnet", "pointnet++"
python train.py \
    --train_data "${DATA_DIR}/0-train.h5" \
    --val_data "${DATA_DIR}/0-valid.h5" \
    --test_data "${DATA_DIR}/0-valid.h5" \
    --discriminator_net "pointnet" \
    --shape "pointcloud" \
    --task "surv" \
    --batchsize 64 \
    --epoch 15 \
    --num_classes 1 \
    --tensorboard

# Meshes
# spiralnet
python train.py \
    --train_data "${DATA_DIR}/0-train.h5" \
    --val_data "${DATA_DIR}/0-valid.h5" \
    --test_data "${DATA_DIR}/0-test.h5" \
    --discriminator_net "spiralnet" \
    --shape "mesh" \
    --task "surv" \
    --batchsize 64 \
    --epoch 15 \
    --num_classes 1 \
    --tensorboard
