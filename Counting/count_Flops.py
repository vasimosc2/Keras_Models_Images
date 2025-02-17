import tensorflow as tf

def count_flops(model, batch_size=1):
    """
    Count FLOPs of a TensorFlow 2.x model.
    
    Parameters:
        model (tf.keras.Model): The model whose FLOPs need to be counted.
        batch_size (int): The batch size for FLOP computation.

    Returns:
        int: The total number of FLOPs in the model.
    """
    # Create a concrete function from the model call
    input_shape = (batch_size, 32, 32, 3)  # Adjust according to your model's input shape
    dummy_input = tf.ones(input_shape)

    # Convert model to a TensorFlow function graph
    concrete_function = tf.function(model).get_concrete_function(dummy_input)
    frozen_func = concrete_function.graph

    # Count the number of float operations
    flops = 0
    for op in frozen_func.get_operations():
        for output in op.outputs:
            shape = output.shape
            if shape.is_fully_defined():
                flops += tf.reduce_prod(shape).numpy()

    return flops
