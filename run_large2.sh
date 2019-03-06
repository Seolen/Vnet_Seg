#!/usr/bin/env bash

### 0225: lr: 1e-3
#Net_id="12"
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
#        --lr=1e-3   \
#        --use_truncated=False    \
#        --use_parallel=True

Net_id="16"
python trainer.py training \
        --env="Vnet_${Net_id}"  \
        --notes="Save the card. 1e-2, balance weights"   \
        --datadir="../Datasets/SegTHOR" \
        --dataset='SegThor2'    \
        --num_classes=5   \
        --num_workers=1   \
        --batch_size=1   \
        --val_batch_size=1    \
        --epoch=200     \
        --lr=1e-2   \
        --use_truncated=False    \
        --use_parallel=True     \
        --use_balance_weight=True