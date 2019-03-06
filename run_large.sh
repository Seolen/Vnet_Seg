#!/usr/bin/env bash

# DeepLab, SegThor
# check DeepLab 'num_classes'
#Net_id="10"
#python trainer.py training \
#        --env="Vnet_${Net_id}"  \
#        --notes="Large Volume"   \
#        --datadir="../Datasets/SegTHOR" \
#        --dataset='SegThor2'    \
#        --num_classes=5   \
#        --num_workers=1   \
#        --batch_size=4   \
#        --val_batch_size=4    \
#        --epoch=200     \
#        --lr=7e-4   \
#        --use_truncated=True    \
#        --use_parallel=True

# --datadir="../Datasets/SegTHOR" \
# --use_parallel=True


### 0225: increase lr: 1e-2
Net_id="13"
python trainer.py training \
        --env="Vnet_${Net_id}"  \
        --notes="1e-2, save model"   \
        --datadir="../Datasets/SegTHOR" \
        --dataset='SegThor2'    \
        --num_classes=5   \
        --num_workers=1   \
        --batch_size=4   \
        --val_batch_size=4    \
        --epoch=200     \
        --lr=1e-2   \
        --use_truncated=False    \
        --use_parallel=True