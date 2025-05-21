import numpy as np

def compute_layer_ram_usage(model, data_dtype_multiplier=4):
    """
    Computes and prints the max RAM usage for each layer in a TensorFlow/Keras model.
    
    Parameters:
    - model (tf.keras.Model): The trained model instance.
    - data_dtype_multiplier (int): Size of each data element in bytes (e.g., 4 for float32).
    """
    print("\n=== Layer-wise Maximum RAM Usage (in KB) ===\n")
    
    for layer in model.layers:
        # Compute activation memory (RAM)
        if isinstance(layer.output, list):
            output_memory = sum(np.prod(out.shape[1:]) * data_dtype_multiplier for out in layer.output)
        else:
            output_memory = np.prod(layer.output.shape[1:]) * data_dtype_multiplier

        if isinstance(layer.input, list):
            input_memory = sum(np.prod(inp.shape[1:]) * data_dtype_multiplier for inp in layer.input)
        else:
            input_memory = np.prod(layer.input.shape[1:]) * data_dtype_multiplier

        # Compute RAM usage per layer
        layer_ram_usage = (input_memory + output_memory) / 1024  # Convert bytes to KB

        print(f"{layer.name}: {layer_ram_usage:.2f} KB")
    print("\n")
