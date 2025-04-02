import json
import os
import argparse
from typing import Dict
import data_processing
import manual_run
import tensorflow as tf
from TakuNet import TakuNetModel
from tensorflow.keras.models import load_model

# def save_tflite_model(model, x_train_sample, model_name):
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]

#     def representative_data_gen():
#         for i in range(100):
#             yield [tf.cast(x_train_sample[i:i+1], tf.float32)]
    
#     converter.representative_dataset = representative_data_gen
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.uint8
#     converter.inference_output_type = tf.uint8

#     tflite_model = converter.convert()
#     os.makedirs("continueTraining/tFlite", exist_ok=True)
#     tflite_path = f"continueTraining/tFlite/{model_name}.tflite"
#     with open(tflite_path, "wb") as f:
#         f.write(tflite_model)
#     print(f"✅ Saved quantized TFLite model to: {tflite_path}")
#     return tflite_model, tflite_path


# def tflite_to_c_array(tflite_model, model_name):
#     os.makedirs("continueTraining/headerModels", exist_ok=True)
#     header_path = f"continueTraining/headerModels/{model_name}.h"
#     c_array = ", ".join(f"0x{byte:02x}" for byte in tflite_model)
#     header = f"""#ifndef {model_name.upper()}_H
# #define {model_name.upper()}_H

# const unsigned char {model_name}_data[] = {{
#     {c_array}
# }};

# unsigned int {model_name}_len = {len(tflite_model)};

# #endif // {model_name.upper()}_H
# """
#     with open(header_path, "w") as f:
#         f.write(header)
#     print(f"✅ Saved C header file to: {header_path}")


def main(model_name: str = None):
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # GPU Setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if any(tf.config.experimental.get_memory_growth(gpu) for gpu in gpus):
                print("⚠️ GPU is already initialized! `set_memory_growth()` will fail.")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"❌ GPU Error: {e}")
    else:
        print("⚠️ No GPU found, running on CPU.")

    # Interactive model picker
    if model_name is None:
        model_files = [f for f in os.listdir("saved_models") if f.endswith(".keras")]
        if not model_files:
            print("❌ No models found in saved_models/")
            return
        
        print("\n📦 Available models:")
        for idx, file in enumerate(model_files):
            print(f"{idx}: {file}")
        
        try:
            choice = int(input("\nEnter the number of the model to continue training: "))
            model_name = model_files[choice].replace(".keras", "")
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            return

    print(f"\n📦 Loading model: {model_name}")
    model_path = f"saved_models/{model_name}.keras"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return
    
    with open("config.json", "r") as config_file:
        config = json.load(config_file)


    loaded_model:tf.keras.Model = load_model(model_path)
    training_params:Dict = manual_run.sample_from_train_and_evaluate(config["train_and_evaluate"])
    x_train, y_train, x_test, y_test = data_processing.get_dataset(output_classes= config["model_search_space"]["refiner_block"]["num_output_classes"], use_augmented_data=False)

    model = TakuNetModel(
    model_name=model_name,
    model_params=None,
    train_params=training_params,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    given_model=loaded_model
)

    print("🚀 Continuing training...")
    model.folderName = "continueTraining"
    model.epochs = 100
    model.train()


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Continue training a saved Keras model.")
    #parser.add_argument("--model_name", type=str, help="The name of the saved model (without extension)")
    #args = parser.parse_args()
    #main(args.model_name)
    main()
