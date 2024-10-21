#!/bin/bash

./run_and_time.sh > /datasets/open-images-v6-mlperf/training.log & ./gpu_script.sh > /datasets/open-images-v6-mlperf/8gpu.txt 
