#!/bin/bash

DATA_CONFIG="$1.json"
SAMPLES_DIR=$2
SAMPLES_PER_CLASS=$3

echo "Generating manifests for age"
python -m sfw_brood.nemo.prepare_manifests \
  -i "/home/gardzielb/sfw-brood-work/out/$SAMPLES_DIR/brood-age.csv" \
  -o "/home/gardzielb/sfw-brood-work/out/$SAMPLES_DIR" \
  -d "/home/gardzielb/sfw-brood-work/$SAMPLES_DIR/audio" \
  -c "config/$DATA_CONFIG" -s $SAMPLES_PER_CLASS -t age

echo "Generating manifests for size"
python -m sfw_brood.nemo.prepare_manifests \
  -i "/home/gardzielb/sfw-brood-work/out/$SAMPLES_DIR/brood-size.csv" \
  -o "/home/gardzielb/sfw-brood-work/out/$SAMPLES_DIR" \
  -d "/home/gardzielb/sfw-brood-work/$SAMPLES_DIR/audio" \
  -c "config/$DATA_CONFIG" -s $SAMPLES_PER_CLASS -t size
