import json
import os
import random
from typing import List
import psutil  # type: ignore # For measuring memory usage

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from TakuNet import TakuNetModel
from data_processing import get_dataset
import tensorflow as tf  # type: ignore
import pandas as pd
from tensorflow.keras import backend as K  # type: ignore
import os
import time


# ✅ Ensure TensorFlow uses GPU if available
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



with open("config.json", "r") as config_file:
    config = json.load(config_file)



x_train, y_train, x_test, y_test = get_dataset(output_classes= config["model_search_space"]["refiner_block"]["num_output_classes"], use_augmented_data=False)


os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)




def sample_from_search_space(model_search_space):
    """Randomly selects model architecture hyperparameters."""
    return {
        "stem_block": {
            "filters": random.choice(model_search_space["stem_block"]["filters"]),
            "Conv_kernel": random.choice(model_search_space["stem_block"]["Conv_kernel"]),
            "strides": random.choice(model_search_space["stem_block"]["strides"]),
            "dropout": random.choice(model_search_space["stem_block"]["dropout"]),
            "dilation_rate": random.choice(model_search_space["stem_block"]["dilation_rate"]),
            "DWConv_kernel": random.choice(model_search_space["stem_block"]["DWConv_kernel"]),
        },
        "stages_block": {
            "stages_number": random.choice(model_search_space["stages_block"]["stages_number"]),
            "taku_block": {
                "taku_block_number": random.choice(model_search_space["stages_block"]["taku_block"]["taku_block_number"]),
                "dropout": random.choice(model_search_space["stages_block"]["taku_block"]["dropout"]),
                "DWConv_kernel": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_kernel"]),
            },
            "downsampler": {
                "dropout": random.choice(model_search_space["stages_block"]["downsampler"]["dropout"]),
                "Conv_kernel": random.choice(model_search_space["stages_block"]["downsampler"]["Conv_kernel"]),
            }
        },
        "refiner_block": {
            "DWConv_kernel": random.choice(model_search_space["refiner_block"]["DWConv_kernel"]),
            "dropout": random.choice(model_search_space["refiner_block"]["dropout"]),
            "num_output_classes": model_search_space["refiner_block"]["num_output_classes"]
        }
    }

def sample_from_train_and_evaluate(train_and_evaluate):
    """Randomly selects training hyperparameters."""
    return {
        "optimizer": random.choice(train_and_evaluate["model_config"]["optimizer"]),
        "loss": train_and_evaluate["model_config"]["loss"],
        "learning_rate": random.choice(train_and_evaluate["model_config"]["learning_rate"]),
        "learning_rate_patience": random.choice(train_and_evaluate["model_config"]["learning_rate_patience"]),
        "early_stopping_patience": random.choice(train_and_evaluate["model_config"]["early_stopping_patience"]),
        "num_epochs": train_and_evaluate["evaluation_config"]["num_epochs"],
        "batch_size": train_and_evaluate["evaluation_config"]["batch_size"],
        "max_ram_consumption": train_and_evaluate["evaluation_config"]["max_ram_consumption"],
        "data_dtype_multiplier": train_and_evaluate["evaluation_config"]["data_dtype_multiplier"],
        "model_dtype_multiplier": train_and_evaluate["evaluation_config"]["model_dtype_multiplier"],
    }


models_to_train:List[TakuNetModel] = []

for i in range(1, 21):  # Train 20 models with random hyperparameters
    model_params = sample_from_search_space(config["model_search_space"])
    train_params = sample_from_train_and_evaluate(config["train_and_evaluate"])

    model_name = f"TakuNet_Random_{i}"
    print(f"\n🔍 Selected hyperparameters for {model_name}:\n{json.dumps(model_params, indent=4)}")

    models_to_train.append(TakuNetModel(model_name=model_name, input_shape=(32, 32, 3), model_params=model_params, train_params=train_params, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test))


results = []
print("🚀 Starting model training...\n")

start_time = time.time()
for model in models_to_train:
    print(f"\nTraining {model.model_name}...")

    model.train()  # Train the model

    if model.results.train_accuracy is not None:
        results.append({
            "Model": model.model_name,
            "Best Train Accuracy": model.results.train_accuracy,
            "Best Test Accuracy": model.results.test_accuracy,
            "Precision": model.results.precision,
            "Recall": model.results.recall,
            "F1 Score": model.results.f1_score,
            "Max RAM Usage (KB)": model.results.max_ram_usage,
            "Param Memory (KB)": model.results.param_memory,
            "Total Memory (KB)": model.results.total_memory,
            "Training Time (s)": model.results.training_time
        })
        K.clear_session()
    else:
        print(f"⚠️ Model {model.model_name} was skipped due to excessive memory usage.")


end_time = time.time()
total_time = end_time - start_time

print(f"\n⏳ Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# **Save Results**
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/Training_Results.csv', index=False)
    print("✅ Results saved to CSV: results/Training_Results.csv")
else:
    print("⚠️ No models were trained due to memory constraints.")


