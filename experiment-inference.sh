#!/bin/bash

if [ "$#" -eq 4 ]; then
  ./venv/bin/python -m sfw_brood.experiment.test_inference -t $2 -p $3 -o $4 $1 > experiment-inference.log 2>&1 &
else
  ./venv/bin/python -m sfw_brood.experiment.test_inference -t $2 $1 > experiment-inference.log 2>&1 &
fi
