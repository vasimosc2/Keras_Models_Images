from typing import Tuple
import joblib
import numpy as np
import tensorflow as tf
from collections import defaultdict



def memoryEstimation(model:tf.keras.Model,data_dtype_multiplier: int = 1)-> Tuple[float, float, float]:
    tensor_lifetimes = defaultdict(int)
    total_param_memory = 0
    layer_tensor_ids = []
    tensor_sizes = {}

    def tensor_id(tensor):
        return id(tensor)

    for layer in model.layers:
        total_param_memory += layer.count_params() * data_dtype_multiplier

        inputs = layer.input if isinstance(layer.input, (list, tuple)) else [layer.input]
        outputs = layer.output if isinstance(layer.output, (list, tuple)) else [layer.output]

        curr_layer_ids = []

        # Track input tensor size
        for inp in inputs:
            tid = tensor_id(inp)
            shape = inp.shape
            size = np.prod(shape[1:]) * data_dtype_multiplier if shape.rank else 0
            tensor_sizes[tid] = size
            tensor_lifetimes[tid] += 1
            curr_layer_ids.append(tid)

        # Track output tensor size
        for out in outputs:
            tid = tensor_id(out)
            shape = out.shape
            size = np.prod(shape[1:]) * data_dtype_multiplier if shape.rank else 0
            tensor_sizes[tid] = size
            tensor_lifetimes[tid] += 1
            curr_layer_ids.append(tid)

        layer_tensor_ids.append(curr_layer_ids)

    # Simulate lifetime and find peak memory
    live_tensors = set()
    peak_ram = 0

    for curr_ids in layer_tensor_ids:
        for tid in curr_ids:
            live_tensors.add(tid)

        current_ram = sum(tensor_sizes[tid] for tid in live_tensors)
        peak_ram = max(peak_ram, current_ram)

        # Simulate tensors being released
        for tid in curr_ids:
            tensor_lifetimes[tid] -= 1
            if tensor_lifetimes[tid] == 0:
                live_tensors.remove(tid)

    flashModel = joblib.load("utils/flash_regression_model_smooth.pkl")

    estimated_ram_kb = peak_ram / 1024
    estimated_flash_kb = flashModel.predict([[total_param_memory / 1024]])[0]

    return estimated_ram_kb, estimated_flash_kb









# def memoryEstimation(model:tf.keras.Model,data_dtype_multiplier: int = 1)-> Tuple[float, float, float]:
#     """
#     ROM (Read-Only Memory) → Memory used to store layer parameters (weights & biases).
#     RAM (Random-Access Memory) → Memory used to store activations (input & output tensors).
#     """
#     max_activation_memory: int = 0  # Peak RAM usage
#     total_param_memory: int = 0      # ROM for storing weights

#     for layer in model.layers:
           
#         total_param_memory += layer.count_params()  * data_dtype_multiplier     #  Number of parameters in the layer (weights & biases). 
#                                                                                 #  Converts the number of parameters into bytes.
#                                                                                 #  Adds up all the layer_param_memory of each layer

#         # Compute activation memory (RAM)
#         if isinstance(layer.output, list):
#             output_memory: int = sum(np.prod(out.shape[1:]) * data_dtype_multiplier for out in layer.output) # I wont be inside there are layer.output is  <class 'keras.src.backend.common.keras_tensor.KerasTensor'>
#         else:
#             output_memory: int = np.prod(layer.output.shape[1:]) * data_dtype_multiplier # If the output shape is 30 x 30 x 32 , the output memmory is  28800 * data_size

#         if isinstance(layer.input, list):
#             input_memory: int = sum(np.prod(inp.shape[1:]) * data_dtype_multiplier for inp in layer.input)
#         else:
#             input_memory: int = np.prod(layer.input.shape[1:]) * data_dtype_multiplier

#         # Track peak RAM usage
#         layer_ram_usage: int = input_memory + output_memory
#         max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

#     estimatedMaxRam: float = max_activation_memory / 1024

#     flashModel = joblib.load("utils/flash_regression_model.pkl")
#     estimatedFlash: float = flashModel.predict([[total_param_memory / 1024]])[0]

#     return estimatedMaxRam, estimatedFlash








# def FlashEstimator(model: tf.keras.Model, input_shape=(32, 32, 3)) -> float:
#     """
#     Simulates a quantized TFLite conversion and returns estimated flash size in KB.
#     """
#     # Dummy representative dataset (untrained models work fine)
#     def representative_dataset():
#         for _ in range(100):
#             data = tf.random.uniform(shape=(1,) + input_shape, minval=0, maxval=1, dtype=tf.float32)
#             yield [data]

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.representative_dataset = representative_dataset
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.uint8
#     converter.inference_output_type = tf.uint8

#     try:
#         tflite_model = converter.convert()
#         size_kb = len(tflite_model) / 1024
#         return size_kb
#     except Exception as e:
#         print(f"❌ TFLite conversion failed during flash estimation: {e}")
#         return -1.0
