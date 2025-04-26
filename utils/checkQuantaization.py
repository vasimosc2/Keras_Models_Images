import tensorflow as tf

# ALL Those float32 are Temp, so all good 
def check_tflite_quantization(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    tensor_idx_to_dtype = {tensor['index']: tensor['dtype'] for tensor in tensor_details}

    ops = interpreter._get_ops_details()

    print("\n🔍 Model Ops:")
    for op in ops:
        input_dtypes = [tensor_idx_to_dtype.get(i, 'UNKNOWN') for i in op['inputs']]
        output_dtypes = [tensor_idx_to_dtype.get(i, 'UNKNOWN') for i in op['outputs']]
        print(f"Op {op['op_name']}")
        print(f"  Inputs: {input_dtypes}")
        print(f"  Outputs: {output_dtypes}")
        if tf.float32 in input_dtypes or tf.float32 in output_dtypes:
            print("  ⚠️ This op uses float32!")



# Example Usage
check_tflite_quantization("../Manual_Run/TfLiteModels/TakuNet_Random_0.tflite")
