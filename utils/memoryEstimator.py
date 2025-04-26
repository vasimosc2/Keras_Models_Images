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



import tensorflow as tf
import numpy as np
from typing import Dict, Set, Tuple

def build_graph(model: tf.keras.Model) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Build dependency graph from a Keras model.
    
    Returns:
        producers (dict): Maps tensor name -> input tensor names
        tensor_sizes (dict): Maps tensor name -> size in bytes (assuming uint8)
    """
    producers = {}
    tensor_sizes = {}

    for layer in model.layers:
        # Get input tensors
        if isinstance(layer.input, list):
            input_names = [inp.name for inp in layer.input]
        else:
            input_names = [layer.input.name]

        # Get output tensors
        if isinstance(layer.output, list):
            output_names = [out.name for out in layer.output]
        else:
            output_names = [layer.output.name]

        for out_name in output_names:
            producers[out_name] = set(input_names)

            # Calculate tensor size (assuming uint8, 1 byte per element)
            shape = layer.output.shape
            if None not in shape:  # Ignore dynamic shapes
                tensor_sizes[out_name] = np.prod(shape[1:])  # Exclude batch dim

    return producers, tensor_sizes

def simulate_working_set(producers: Dict[str, Set[str]], tensor_sizes: Dict[str, int], inputs: Set[str]) -> float:
    """
    Simulate execution, allocating and freeing memory dynamically.

    Args:
        producers (dict): Tensor -> set of input tensors
        tensor_sizes (dict): Tensor -> size in bytes
        inputs (set): Set of input tensor names

    Returns:
        peak_memory_kb (float): Peak RAM usage in KB
    """
    tensor_ref_count = {}
    for inputs_set in producers.values():
        for inp in inputs_set:
            tensor_ref_count[inp] = tensor_ref_count.get(inp, 0) + 1

    working_set = set(inputs)  # Inputs are initially in memory
    current_memory = sum(tensor_sizes.get(tensor, 0) for tensor in working_set)
    peak_memory = current_memory

    pending = set(producers.keys())

    while pending:
        progress = False

        for tensor in list(pending):
            input_tensors = producers[tensor]
            if all(inp in working_set for inp in input_tensors):
                # All inputs ready, execute this op
                working_set.add(tensor)
                current_memory += tensor_sizes.get(tensor, 0)

                # Update peak memory
                peak_memory = max(peak_memory, current_memory)

                # Free inputs if no longer needed
                for inp in input_tensors:
                    tensor_ref_count[inp] -= 1
                    if tensor_ref_count[inp] == 0:
                        working_set.remove(inp)
                        current_memory -= tensor_sizes.get(inp, 0)

                pending.remove(tensor)
                progress = True
                break  # Restart loop to prioritize freeing memory early

        if not progress:
            raise RuntimeError("Deadlock detected: model graph is not a DAG or missing inputs!")

    return peak_memory / 1024  # Return in KB

def estimate_model_memory(model: tf.keras.Model, input_shape=(32, 32, 3)) -> float:
    """
    Full memory estimation pipeline.

    Args:
        model: Keras model.
        input_shape: Shape of the model input (excluding batch).

    Returns:
        Peak RAM usage in KB.
    """
    producers, tensor_sizes = build_graph(model)

    # Build input tensor names manually
    input_tensor_names = set()
    for input_tensor in model.inputs:
        input_tensor_names.add(input_tensor.name)

        # Also calculate input size (important!)
        shape = input_tensor.shape
        if None not in shape:
            tensor_sizes[input_tensor.name] = np.prod(shape[1:])

    peak_ram_kb = simulate_working_set(producers, tensor_sizes, input_tensor_names)

    return peak_ram_kb
