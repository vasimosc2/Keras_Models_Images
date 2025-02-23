import numpy as np
import tensorflow as tf # type: ignore
from typing import Tuple

def estimate_max_memory_usage(model: tf.keras.Model, dtype_size: int = 4) -> Tuple[float, float, float]:
    """
    Estimates the peak RAM usage of a model during inference.

    Parameters:
    - model (tf.keras.Model): The Keras model (loaded or defined).
    - dtype_size (int): Size of data type in bytes (4 for FP32, 2 for FP16, 1 for INT8).

    Returns:
    - Tuple[float, float, float]: 
      - max_ram_usage (float): Maximum RAM required (KB).
      - param_memory (float): ROM required for storing weights (KB).
      - total_memory (float): Combined memory usage (KB).
    """
    max_activation_memory: int = 0  # Peak RAM usage
    total_param_memory: int = 0      # ROM for storing weights

    for layer in model.layers:
        # Compute memory for layer parameters (ROM)
        layer_params: int = layer.count_params()
        layer_param_memory: int = layer_params * dtype_size
        total_param_memory += layer_param_memory

        # Compute activation memory (RAM)
        if isinstance(layer.output, list):
            output_memory: int = sum(np.prod(out.shape[1:]) * dtype_size for out in layer.output)
        else:
            output_memory: int = np.prod(layer.output.shape[1:]) * dtype_size

        if isinstance(layer.input, list):
            input_memory: int = sum(np.prod(inp.shape[1:]) * dtype_size for inp in layer.input)
        else:
            input_memory: int = np.prod(layer.input.shape[1:]) * dtype_size

        # Track peak RAM usage
        layer_ram_usage: int = input_memory + output_memory
        max_activation_memory = max(max_activation_memory, layer_ram_usage)

    # Convert bytes to KB
    max_ram_usage: float = max_activation_memory / 1024
    param_memory: float = total_param_memory / 1024
    total_memory: float = (max_activation_memory + total_param_memory) / 1024

    return max_ram_usage, param_memory, total_memory