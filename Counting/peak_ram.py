import numpy as np
import tensorflow as tf  # type: ignore
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
    max_activation_memory = 0  # Peak RAM usage for activations
    total_param_memory = 0      # ROM for storing weights

    for layer in model.layers:
        # Compute memory for layer parameters (ROM)
        layer_params = layer.count_params() if hasattr(layer, "count_params") else 0
        layer_param_memory = layer_params * dtype_size
        total_param_memory += layer_param_memory

        # Compute activation memory (RAM)
        output_shape = getattr(layer, "output_shape", None)
        input_shape = getattr(layer, "input_shape", None)

        if output_shape is not None:
            if isinstance(output_shape, list):
                output_memory = sum(np.prod(shape[1:]) * dtype_size for shape in output_shape if shape)
            else:
                output_memory = np.prod(output_shape[1:]) * dtype_size if output_shape else 0
        else:
            output_memory = 0

        if input_shape is not None:
            if isinstance(input_shape, list):
                input_memory = sum(np.prod(shape[1:]) * dtype_size for shape in input_shape if shape)
            else:
                input_memory = np.prod(input_shape[1:]) * dtype_size if input_shape else 0
        else:
            input_memory = 0

        # Track peak RAM usage
        layer_ram_usage = input_memory + output_memory
        max_activation_memory = max(max_activation_memory, layer_ram_usage)

    # Convert bytes to KB
    max_ram_usage = max_activation_memory / 1024
    param_memory = total_param_memory / 1024
    total_memory = (max_activation_memory + total_param_memory) / 1024

    return max_ram_usage, param_memory, total_memory
