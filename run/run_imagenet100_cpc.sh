#!/bin/sh
python train_self_supervised_task.py -d imagenet100 -t cpc -b 64 -e 200 --lr 1e-3 --o adam
