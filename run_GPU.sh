#!/bin/bash

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=3

# Set the correct CUDA path for XLA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.8.0

# Set CUDA to use GPU 0 (optional, if you want to specify the GPU)
export CUDA_VISIBLE_DEVICES=0