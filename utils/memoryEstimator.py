from typing import List, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf


import re
import numpy as np
from tensorflow.keras.models import Model

def memoryEstimation_peak_ram_only(model: Model, data_dtype_multiplier: int = 1) -> float:
    """
    Estimate peak RAM usage in KB based on TakuNet's activation dependencies.
    """
    layer_ram_kb = {}

    for layer in model.layers:
        if isinstance(layer.output, list):
            output_memory = sum(np.prod(out.shape[1:]) * data_dtype_multiplier for out in layer.output)
        else:
            output_memory = np.prod(layer.output.shape[1:]) * data_dtype_multiplier

        if isinstance(layer.input, list):
            input_memory = sum(np.prod(inp.shape[1:]) * data_dtype_multiplier for inp in layer.input)
        else:
            input_memory = np.prod(layer.input.shape[1:]) * data_dtype_multiplier

        total_ram_bytes = input_memory + output_memory
        layer_ram_kb[layer.name] = total_ram_bytes / 1024  # in KB

    peak_ram = 0.0
    stage_concat = {}
    stage_skips = {}

    for name, kb in layer_ram_kb.items():
        if "concatenate" in name:
            stage = len(stage_concat)
            stage_concat[stage] = kb
        elif "add" in name:
            stage = len(stage_skips)
            stage_skips.setdefault(stage, []).append(kb)

    for stage in stage_concat:
        concat_kb = stage_concat.get(stage, 0)
        skip_kbs = sorted(stage_skips.get(stage, []), reverse=True)[:2]
        stage_peak = concat_kb + sum(skip_kbs)
        peak_ram = max(peak_ram, stage_peak)

    # Stem block (initial conv + dwconv + bn + relu)
    stem_kb = sum(
        kb for name, kb in layer_ram_kb.items()
        if "conv2d" in name or "depthwise_conv2d" in name or "re_lu" in name or "batch_normalization" in name
    )
    peak_ram = max(peak_ram, stem_kb)

    # Refiner block
    refiner_kb = sum(
        kb for name, kb in layer_ram_kb.items()
        if "dense" in name or "global_average_pooling2d" in name
    )
    peak_ram = max(peak_ram, refiner_kb)

    return round(peak_ram, 2)



def memoryEstimation(model:tf.keras.Model,data_dtype_multiplier: int = 1)-> Tuple[float, float, float]:
    """
    ROM (Read-Only Memory) → Memory used to store layer parameters (weights & biases).
    RAM (Random-Access Memory) → Memory used to store activations (input & output tensors).
    """
    max_activation_memory: int = 0  # Peak RAM usage
    total_param_memory: int = 0      # ROM for storing weights
    layer_ram_usages: List[int] = []          # Store RAM usage of each layer

    for layer in model.layers:
        
        #  Number of parameters in the layer (weights & biases).
        #  Converts the number of parameters into bytes.
        # Adds up all the layer_param_memory of each layer
        total_param_memory += layer.count_params()  * data_dtype_multiplier 

        # Compute activation memory (RAM)
        # I wont be inside there are layer.output is  <class 'keras.src.backend.common.keras_tensor.KerasTensor'>
        
        # Compute activation memory (RAM)
        if isinstance(layer.output, list):
            output_memory: int = sum(np.prod(out.shape[1:]) * data_dtype_multiplier for out in layer.output) # I wont be inside there are layer.output is  <class 'keras.src.backend.common.keras_tensor.KerasTensor'>
        else:
            output_memory: int = np.prod(layer.output.shape[1:]) * data_dtype_multiplier # If the output shape is 30 x 30 x 32 , the output memmory is  28800 * data_size

        if isinstance(layer.input, list):
            input_memory: int = sum(np.prod(inp.shape[1:]) * data_dtype_multiplier for inp in layer.input)
        else:
            input_memory: int = np.prod(layer.input.shape[1:]) * data_dtype_multiplier


        # Track peak RAM usage
        layer_ram_usage: int = input_memory + output_memory
        layer_ram_usages.append(layer_ram_usage / 1024)
        max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

    flashModel : LinearRegression = joblib.load("utils/EstimationModels/flash_regression_model.pkl")
    ramModel : LinearRegression  = joblib.load("utils/EstimationModels/ram_regression_model.pkl")

    estimated_ram_kb:float = max_activation_memory / 1024
    estimated_flash_kb:float = flashModel.predict([[total_param_memory / 1024]])[0]
    accurate_ram_kb:float = ram_accurate(max_activation_memory=estimated_ram_kb,layer_ram_usages=layer_ram_usages)

    modelRAM:float = ramModel.predict([[accurate_ram_kb]])[0]
    return estimated_ram_kb, estimated_flash_kb, accurate_ram_kb, modelRAM


def ram_accurate(max_activation_memory:int,layer_ram_usages:List[int]) -> float:
     # Now, check for >4 consecutive layers with max RAM usage
    consecutive_max = 0
    max_ram_reached = False

    for ram_usage in layer_ram_usages:
        if ram_usage == max_activation_memory:
            consecutive_max += 1
            if consecutive_max > 4:
                max_ram_reached = True
                break
        else:
            consecutive_max = 0  # Reset if break in maximum RAM sequence

    if max_ram_reached:
        return 2 * max_activation_memory   # Double the RAM estimation
    
    return max_activation_memory