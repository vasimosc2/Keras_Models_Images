from typing import Tuple
import joblib
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
        max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

    flashModel = joblib.load("utils/flash_regression_model.pkl")

    estimated_ram_kb = max_activation_memory / 1024
    estimated_flash_kb = flashModel.predict([[total_param_memory / 1024]])[0]

    return estimated_ram_kb, estimated_flash_kb




def estimate_peak_ram_uint8(model: tf.keras.Model, input_shape=(32, 32, 3)):
    """
    Estimate peak RAM usage for a quantized (uint8) model 
    by simulating a forward pass layer-by-layer.
    """
    dummy_input = tf.zeros((1,) + input_shape, dtype=tf.uint8)
    max_memory_bytes = 0
    current_memory_bytes = 0

    x = dummy_input

    for layer in model.layers:
        try:
            # --- Handle multi-input layers properly ---
            if isinstance(layer.input, (list, tuple)):
                # If the layer expects multiple inputs, wrap x in a list
                x_new = layer([x])
            else:
                x_new = layer(x)

            # --- Memory Calculation ---
            if isinstance(x_new, (list, tuple)):
                # If output is multiple tensors (rare), sum their memory
                tensor_memory_bytes = sum(
                    np.prod(output.shape) * tf.dtypes.as_dtype(output.dtype).size
                    for output in x_new
                )
            else:
                tensor_memory_bytes = np.prod(x_new.shape) * tf.dtypes.as_dtype(x_new.dtype).size

            current_memory_bytes += tensor_memory_bytes
            max_memory_bytes = max(max_memory_bytes, current_memory_bytes)

            # --- Free previous input memory ---
            if isinstance(x, (list, tuple)):
                input_memory_bytes = sum(
                    np.prod(inp.shape) * tf.dtypes.as_dtype(inp.dtype).size
                    for inp in x
                )
            else:
                input_memory_bytes = np.prod(x.shape) * tf.dtypes.as_dtype(x.dtype).size

            current_memory_bytes -= input_memory_bytes

            # --- Update x for next layer ---
            x = x_new

        except Exception as e:
            print(f"⚠️ Skipping layer {layer.name} due to error: {e}")
            # Don't update x if failed, move to next layer
            continue

    print(f"✅ Estimated Peak RAM Usage (uint8 model): {max_memory_bytes / 1024:.2f} KB")
    return max_memory_bytes




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
