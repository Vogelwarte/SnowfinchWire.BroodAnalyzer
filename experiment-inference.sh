#!/bin/bash

./venv/bin/python -m sfw_brood.experiment.test_inference -t $2 $1 > experiment-inference.log 2>&1 &
