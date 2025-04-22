from typing import Tuple

import numpy as np
import tensorflow as tf


def memoryEstimation(model:tf.keras.Model,data_dtype_multiplier: int = 1)-> Tuple[float, float, float]:
    """
    ROM (Read-Only Memory) → Memory used to store layer parameters (weights & biases).
    RAM (Random-Access Memory) → Memory used to store activations (input & output tensors).
    """
    max_activation_memory: int = 0  # Peak RAM usage
    total_param_memory: int = 0      # ROM for storing weights

    for layer in model.layers:
        
        layer_params: int = layer.count_params() #  Number of parameters in the layer (weights & biases).
        layer_param_memory: int = layer_params * data_dtype_multiplier#  Converts the number of parameters into bytes.
        total_param_memory += layer_param_memory # Adds up all the layer_param_memory of each layer

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
        max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

    # Convert bytes to KB
    estimatedMaxRam: float = max_activation_memory / 1024
    estimatedFlash: float = total_param_memory / 1024

    return estimatedMaxRam, estimatedFlash

def FlashEstimator(model: tf.keras.Model, input_shape=(32, 32, 3)) -> float:
    """
    Simulates a quantized TFLite conversion and returns estimated flash size in KB.
    """
    # Dummy representative dataset (untrained models work fine)
    def representative_dataset():
        for _ in range(100):
            data = tf.random.uniform(shape=(1,) + input_shape, minval=0, maxval=1, dtype=tf.float32)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_model = converter.convert()
        size_kb = len(tflite_model) / 1024
        return size_kb
    except Exception as e:
        print(f"❌ TFLite conversion failed during flash estimation: {e}")
        return -1.0
