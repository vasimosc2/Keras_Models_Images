import json
import os
import argparse
import pandas as pd
from typing import Dict
import tensorflow as tf
from TakuNet import TakuNetModel
from tensorflow.keras.models import load_model

from utils import getTrainingParameters

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
    training_params:Dict = getTrainingParameters.sample_from_train_and_evaluate(config["train_and_evaluate"])

    import data_processing
    use_augmented_data: bool = True
    apply_standard: bool = False
    apply_color: bool = True
    apply_geometric: bool = False
    apply_mixup: bool = False
    apply_cutmix: bool = False
    x_train, y_train, x_test, y_test = data_processing.get_dataset(output_classes= config["model_search_space"]["refiner_block"]["num_output_classes"],
                                                                   use_augmented_data=use_augmented_data,
                                                                   apply_standard=apply_standard,
                                                                   apply_color=apply_color,
                                                                   apply_geometric=apply_geometric,
                                                                   apply_mixup=apply_mixup,
                                                                   apply_cutmix=apply_cutmix
                                                                   )

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

    if model.results.train_accuracy is not None:
    
        os.makedirs("continueTraining/results", exist_ok=True)

        results_df = pd.DataFrame([{
            "Model": model.model_name,
            "Best Train Accuracy": model.results.train_accuracy,
            "Best Test Accuracy": model.results.test_accuracy,
            "TFlite Test Accuracy": model.results.tflite_accuracy,
            "Precision": model.results.precision,
            "Recall": model.results.recall,
            "F1 Score": model.results.f1_score,
            "Max RAM Usage (KB)": model.results.max_ram_usage,
            "TFlite Estimation size(KB)": model.results.tflite_size,
            "Param Memory (KB)": model.results.param_memory,
            "Total Memory (KB)": model.results.total_memory,
            "Training Time (s)": model.results.training_time
        }])

        results_path = f"continueTraining/results/{model.model_name}_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to: {results_path}")
    else:
        print(f"⚠️ Model {model.model_name} was skipped due to memory constraints.")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Continue training a saved Keras model.")
    #parser.add_argument("--model_name", type=str, help="The name of the saved model (without extension)")
    #args = parser.parse_args()
    #main(args.model_name)
    main()
