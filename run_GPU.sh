#!/bin/bash
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
module load python3/3.10.13
module load cuda/12.8

nvcc --version

export TF_CPP_MIN_LOG_LEVEL=3

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.8.0
python3 main.py
# Set CUDA to use GPU 0 (optional, if you want to specify the GPU)
