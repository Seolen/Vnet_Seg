#!/usr/bin/env bash

# DeepLab, SegThor
# check DeepLab 'num_classes'
#Net_id="01"
#python trainer.py training \
#        --env="Vnet_${Net_id}"  \
#        --notes="Vnet-SegThor"   \
#        --datadir="../Datasets/SegTHOR" \
#        --num_classes=5   \
#        --num_workers=8   \
#        --batch_size=8   \
#        --val_batch_size=8    \
#        --epoch=200     \
#        --lr=7e-4   \
#        --use_truncated=False    \
#        --use_parallel=True

# --datadir="../Datasets/SegTHOR" \
# --use_parallel=True

Net_id="21"
python trainer.py training \
        --env="Vnet_${Net_id}"  \
        --notes="whole volume, resize"   \
        --datadir="../Datasets/SegTHOR/train" \
        --dataset='SegThor3'    \
        --num_classes=5   \
        --num_workers=1   \
        --batch_size=1   \
        --val_batch_size=1    \
        --epoch=200     \
        --lr=1e-2   \
        --use_truncated=False    \
        --use_parallel=True     \
        --use_balance_weight=True