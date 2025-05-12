#!/bin/bash
sxm2sh
nvidia-smi # To see which GPUs are idle
export CUDA_VISIBLE_DEVICES=2
module load cuda/12.8
nvcc --version
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.8.0
source venv/bin/activate

# Choices
python3 manual_run.py --num_models 2 # This manually creates TakuNetModels
python3 evolutionary_run.py # This Runs the Actual Nas with Genetic Algorythm
python3 TrainBestModels.py # This was able to re-train and existing Model
python -m Retrain.createAndTrain