#!/usr/bin/env bash

Net_id="23"
python trainer.py training \
        --env="Vnet_${Net_id}"  \
        --notes="DiceLoss used"   \
        --datadir="../Datasets/SegTHOR/train" \
        --dataset='SegThor3'    \
        --num_classes=5   \
        --num_workers=4   \
        --batch_size=1   \
        --val_batch_size=1    \
        --epoch=200     \
        --lr=1e-2   \
        --use_truncated=True    \
        --use_parallel=True     \
        --use_dice_loss=True