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




def estimate_tflite_ram_from_keras(model: tf.keras.Model, input_shape=(32, 32, 3)):
    """
    Estimate peak RAM usage from a Keras model by:
    1. Converting it to a quantized TFLite model (in memory).
    2. Estimating RAM usage without saving to disk.
    
    Parameters:
        model: tf.keras.Model
            The Keras model to be quantized and analyzed.
        input_shape: tuple
            Input shape of the model excluding batch dimension.
    """
    # --- Define an automatic representative dataset generator ---
    def representative_dataset():
        for _ in range(100):
            dummy_input = tf.random.uniform(shape=(1,) + input_shape, minval=0, maxval=1, dtype=tf.float32)
            yield [dummy_input]

    # --- Convert Keras model to TFLite model (in memory) ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = True  # default True in latest TF

    tflite_model_buffer = converter.convert()

    # --- Load TFLite model from buffer ---
    interpreter = tf.lite.Interpreter(model_content=tflite_model_buffer)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    total_arena_memory = 0
    for tensor in tensor_details:
        if tensor['shape_signature'] is not None and tensor['dtype'] is not None:
            shape = tensor['shape_signature']
            num_elements = np.prod([dim if dim > 0 else 1 for dim in shape])
            dtype_size = tf.dtypes.as_dtype(tensor['dtype']).size
            total_arena_memory += num_elements * dtype_size

    print(f"✅ Estimated Peak RAM Usage (from quantized TFLite model): {total_arena_memory / 1024:.2f} KB")
    return total_arena_memory




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
