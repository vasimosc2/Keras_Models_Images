import tensorflow as tf

def count_flops(model, batch_size=1):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # Wrap the model call with tf.function to ensure graph mode execution
    @tf.function
    def run_inference(x):
        return model(x)

    # Create a dummy input for profiling (ensure the shape matches the input of your model)
    dummy_input = tf.ones((batch_size, 32, 32, 3))  # Replace with your actual input shape

    # Run the dummy input through the model to initialize graph execution
    run_inference(dummy_input)

    with tf.compat.v1.Session() as session:
        flops = tf.compat.v1.profiler.profile(graph=session.graph,
                                              run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops
