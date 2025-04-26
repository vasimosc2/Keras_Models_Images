import json
import os
import random
from typing import List
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K
import os
import time
import argparse
import gc
from utils import getSearchSpaceParameters, getTrainingParameters

parser = argparse.ArgumentParser(description="Train TakuNet models with sampled hyperparameters.")
parser.add_argument("--num_models", type=int, default=5, help="Number of models to train (default: 5)")
args = parser.parse_args()
number_of_models = args.num_models


Folder="Manual_Run"


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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



from TakuNet import TakuNetModel
from data_processing import get_dataset
from compute_ram_show import compute_layer_ram_usage




os.makedirs(f'{Folder}/saved_models', exist_ok=True)
os.makedirs(f'{Folder}/results', exist_ok=True)




models_to_train: List[TakuNetModel] = []
trainable_models_count = 0



default_augementaion_technique ={ "apply_standard":False,
                            "apply_color":False,
                            "apply_geometric":False,
                            "apply_mixup": False,
                            "apply_cutmix": False}


x_train, y_train, x_test, y_test = get_dataset( output_classes= config["model_search_space"]["refiner_block"]["num_output_classes"], 
                                                augementation_technique=default_augementaion_technique)

    

while trainable_models_count < number_of_models:  # Train number_of_models with random hyperparameters
    model_params = getSearchSpaceParameters.sample_from_search_space(config["model_search_space"])
    train_params = getTrainingParameters.sample_from_train_and_evaluate(config["train_and_evaluate"])

    # aug_type = random.choice(['standard', 'geometric', 'mixup'])

    # apply_standard = aug_type == 'standard'
    # apply_geometric = aug_type == 'geometric'
    # apply_mixup = aug_type == 'mixup'

    # augmentation_technique = {  "apply_standard":apply_standard,
    #                             "apply_color":False,
    #                             "apply_geometric":apply_geometric,
    #                             "apply_mixup": apply_mixup,
    #                             "apply_cutmix": False
    #                                 }
    # print(f"\n🎲 Randomly selected augmentation for model {trainable_models_count}: {aug_type}\n")


    
    model_name = f"TakuNet_Random_{trainable_models_count}"
    #print(f"\n🔍 Selected hyperparameters for {model_name}:\n{json.dumps(model_params, indent=4)}")
    taku_model: TakuNetModel = TakuNetModel(model_name=model_name, 
                                            input_shape=(32, 32, 3), 
                                            model_params=model_params, 
                                            train_params=train_params, 
                                            x_train=x_train,
                                            y_train=y_train, 
                                            x_test=x_test, 
                                            y_test=y_test,
                                            folder=Folder)
    if taku_model.is_trainable:
        models_to_train.append(taku_model)
        trainable_models_count += 1
        compute_layer_ram_usage(taku_model.model, data_dtype_multiplier=1)
        print(f"✅ Model {model_name} accepted for training")
    else:
        print(f"❌ Model {model_name} rejected due to memory constraints")
    
    
    #del taku_model, x_train, y_train, x_test, y_test
    tf.keras.backend.clear_session()
    gc.collect()





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
            "TFlite Test Accuracy": model.results.tflite_accuracy,
            "Precision": model.results.precision,
            "Recall": model.results.recall,
            "F1 Score": model.results.f1_score,
            "Estimated Max RAM Usage (KB)": model.results.estimatedMaxRam,
            "Accurate RAM Usage (KB)": model.results.AccurateMaxRam,
            "Estimated Flash Memory (KB)": model.results.estimatedFlash,
            "TFlite size (KB)": model.results.tflite_size,
            "Training Time (s)": model.results.training_time,
            "Flop Number": model.results.flops,
            "Epochs Trained": model.results.epochs_trained
        })

        # ✅ Save training history to CSV
        
        hist_df = pd.DataFrame(model.results.history.history)
        hist_path = f'{Folder}/results/{model.model_name}_history.csv'
        hist_df.to_csv(hist_path, index=False)
        print(f"📊 Training history saved to: {hist_path}")
        K.clear_session()
    else:
        print(f"⚠️ Model {model.model_name} was skipped due to excessive memory usage.")


end_time = time.time()
total_time = end_time - start_time

print(f"\n⏳ Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# **Save Results**
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{Folder}/results/Training_Results.csv', index=False)
    print(f"✅ Results saved to CSV: {Folder}/results/Training_Results.csv")
else:
    print("⚠️ No models were trained due to memory constraints.")


