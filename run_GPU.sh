#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
module load cuda/12.8
nvcc --version
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.8.0
python3 main.py
