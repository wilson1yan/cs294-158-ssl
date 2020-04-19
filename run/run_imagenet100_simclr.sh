#!/bin/sh

# Trained on 4 Titan Xps. Turn down batch size to use less GPU memory. If your batch size is <= 256, then set -u 0 (no warmup)
python train_self_supervised_task.py -d imagenet100 -t simclr -b 512 -e 300 -o lars --lr 0.3 -w 1e-6 -u 10
