#!/bin/sh

# Trained on 4 Titan Xps. Turn down batch size to use less GPU memory. If your batch size is <= 256, then set -u 0 (no warmup)
python train_self_supervised_task.py -d cifar10 -t simclr -b 512 -e 1000 -o lars --lr 1.0 -w 1e-6 -u 10
