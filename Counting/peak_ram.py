import numpy as np
import tensorflow as tf # type: ignore
from typing import Tuple

def estimate_max_memory_usage(model: tf.keras.Model, dtype_size: int = 4) -> Tuple[float, float, float]:
    """
    ROM (Read-Only Memory) → Memory used to store layer parameters (weights & biases).
    RAM (Random-Access Memory) → Memory used to store activations (input & output tensors).
    """
    max_activation_memory: int = 0  # Peak RAM usage
    total_param_memory: int = 0      # ROM for storing weights

    for layer in model.layers:
        
        layer_params: int = layer.count_params() #  Number of parameters in the layer (weights & biases).
        layer_param_memory: int = layer_params * dtype_size #  Converts the number of parameters into bytes.
        total_param_memory += layer_param_memory # Adds up all the layer_param_memory of each layer

        # Compute activation memory (RAM)
        if isinstance(layer.output, list):
            output_memory: int = sum(np.prod(out.shape[1:]) * dtype_size for out in layer.output) # I wont be inside there are layer.output is  <class 'keras.src.backend.common.keras_tensor.KerasTensor'>
        else:
            output_memory: int = np.prod(layer.output.shape[1:]) * dtype_size # If the output shape is 30 x 30 x 32 , the output memmory is  28800 * data_size

        if isinstance(layer.input, list):
            input_memory: int = sum(np.prod(inp.shape[1:]) * dtype_size for inp in layer.input)
        else:
            input_memory: int = np.prod(layer.input.shape[1:]) * dtype_size

        # Track peak RAM usage
        layer_ram_usage: int = input_memory + output_memory
        max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

    # Convert bytes to KB
    max_ram_usage: float = max_activation_memory / 1024
    param_memory: float = total_param_memory / 1024
    total_memory: float = (max_activation_memory + total_param_memory) / 1024

    return max_ram_usage, param_memory, total_memory