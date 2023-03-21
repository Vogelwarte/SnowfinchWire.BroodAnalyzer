#!/bin/bash

N_EPOCHS=20
LR=0.0005
BS=64
N_SAMPLES=10_000
N_WORKERS=16
LOG_DIR=~/sfw-brood-work/log/

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 5.0 -t age -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --group-ages 0-5.5,6-8.5,9-11.5,12-14.5,15-30 \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s5.0-o1.0 ~/sfw-brood-work/s5-o1/s5.0-o1.0 > $LOG_DIR/train-age-resnet50-s5.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 2.0 -t age -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --group-ages 0-5.5,6-8.5,9-11.5,12-14.5,15-30 \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s2.0-o1.0 ~/sfw-brood-work/s2-o1/s2.0-o1.0 > $LOG_DIR/train-age-resnet50-s2.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 10.0 -t age -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --group-ages 0-5.5,6-8.5,9-11.5,12-14.5,15-30 \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s10.0-o1.0 ~/sfw-brood-work/s10-o1/s10.0-o1.0 > $LOG_DIR/train-age-resnet50-s10.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 5.0 -t size -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s5.0-o1.0 ~/sfw-brood-work/s5-o1/s5.0-o1.0 > $LOG_DIR/train-size-resnet50-s5.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 2.0 -t size -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s2.0-o1.0 ~/sfw-brood-work/s2-o1/s2.0-o1.0 > $LOG_DIR/train-size-resnet50-s2.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a resnet50 -d 10.0 -t size -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s10.0-o1.0 ~/sfw-brood-work/s10-o1/s10.0-o1.0 > $LOG_DIR/train-size-resnet50-s10.log 2>&1

./venv/bin/python -m sfw_brood.train \
    -a inception_v3 -d 5.0 -t age -e feeding -n $N_EPOCHS -l $LR -b $BS -w $N_WORKERS \
    --group-ages 0-5.5,6-8.5,9-11.5,12-14.5,15-30 \
    --samples-per-class $N_SAMPLES -c config/data-split-server.json \
    ~/sfw-brood-work/out/s5.0-o1.0 ~/sfw-brood-work/s5-o1/s5.0-o1.0 > $LOG_DIR/train-age-inception-s5.log 2>&1
