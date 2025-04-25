import tensorflow as tf

def check_tflite_quantization(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    dtypes_used = set()
    for tensor in tensor_details:
        dtypes_used.add(tensor['dtype'])

    print(f"\n📦 Tensor data types used in model: {dtypes_used}")

    if dtypes_used == {tf.uint8}:
        print("✅ Model is fully quantized to uint8!")
    elif tf.int8 in dtypes_used and len(dtypes_used) == 1:
        print("✅ Model is fully quantized to int8!")
    else:
        print("⚠️ Model is not fully quantized!")
        if tf.float32 in dtypes_used:
            print("⚠️ Found float32 tensors, model is partially quantized.")
        print(f"👉 Tensors found: {dtypes_used}")

# Example Usage
check_tflite_quantization("Manual_Run/TfLiteModels/TakuNet_Random_0.tflite")
