import json
import os
import random
import psutil  # type: ignore # For measuring memory usage
from Training.train_and_evaluate import train_and_evaluate_model
from models.deeper_cnn import create_deeper_cnn
from models.simple_cnn import create_simple_cnn
from models.cnn_with_gap import create_cnn_with_gap
from models.cnn_with_batchnorm import create_cnn_with_batchnorm
from models.cnn_with_dropout import create_cnn_with_dropout
from models.resnet_like import create_resnet_like_cnn
from models.takunet import create_takunet_model
import tensorflow as tf  # type: ignore
import pandas as pd
from tensorflow.keras import layers # type: ignore
from tensorflow.keras import backend as K  # type: ignore
import os
import time 

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU




def sample_from_search_space(model_search_space:dict) -> dict:
    """Randomly select hyperparameters from the search space."""
    return {
        "stages": random.choice(model_search_space["stages"]),

        "filters_stem_1": random.choice(model_search_space["filters"]),
        "filters_stem_2": random.choice(model_search_space["filters"]),
        "filters_taku_block_1": random.choice(model_search_space["filters"]),
        "filters_taku_block_2": random.choice(model_search_space["filters"]),

        "kernel_size_stem_1": random.choice(model_search_space["kernel_size"]),
        "kernel_size_stem_2": random.choice(model_search_space["kernel_size"]),
        "kernel_size_taku_block_1": random.choice(model_search_space["kernel_size"]),
        "kernel_size_taku_block_2": random.choice(model_search_space["kernel_size"]),
        "kernel_size_refinement_block": random.choice(model_search_space["kernel_size"]),

        "dropout_rate": random.choice(model_search_space["dropout_rate"]),

        "activation": random.choice(model_search_space["activation"]),
        "strides": random.choice(model_search_space["strides"]),
        "batch_norm": random.choice(model_search_space["batch_norm"]),
        "num_units": random.choice(model_search_space["num_units"]),
        "dense_activation": random.choice(model_search_space["dense_activation"]),

        "num_output_classes":model_search_space["num_output_classes"]
    }


def sample_from_train_and_evaluate(train_and_evaluate:dict) -> dict:
    return{
        "optimizer": random.choice(train_and_evaluate["model_config"]["optimizer"]),
        "loss": train_and_evaluate["model_config"]["loss"],
        "learning_rate": random.choice(train_and_evaluate["model_config"]["learning_rate"]),
        "num_epochs": train_and_evaluate["evaluation_config"]["num_epochs"],
        "batch_size": train_and_evaluate["evaluation_config"]["batch_size"],
        "max_ram_consumption": train_and_evaluate["evaluation_config"]["max_ram_consumption"],
        "max_flash_consumption": train_and_evaluate["evaluation_config"]["max_flash_consumption"], 
        "data_dtype_multiplier": train_and_evaluate["evaluation_config"]["data_dtype_multiplier"],
        "model_dtype_multiplier": train_and_evaluate["evaluation_config"]["model_dtype_multiplier"],
    }


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

config_file = open("config.json", "r")

config:dict = json.load(config_file)
output_class:int = config["model_search_space"]["num_output_classes"]

y_train = tf.keras.utils.to_categorical(y_train, output_class)
y_test = tf.keras.utils.to_categorical(y_test, output_class)

os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

models_to_train = {}

for i in range(1, 30):
    params = sample_from_search_space(config["model_search_space"])
    stage_count = params["stages"]
    print(f"The random params selected for model_{i} are:\n{json.dumps(params, indent=4)}")
    model_name = f"TakuNet Random_{i} (Stages: {stage_count})"
    models_to_train[model_name] = create_takunet_model(params=params)


results = []
print("I am starting the training\n")
start_time = time.time()
for model_name, model in models_to_train.items():
    
    print(f"\nTraining {model_name}...")
    
    
    results_data = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name, 
                                            params=sample_from_train_and_evaluate(config["train_and_evaluate"]))

    if results_data is not None:

        test_acc, training_acc, precision, recall, model_size, flops, max_ram, param_mem, total_ram_mem, training_time = results_data
        
        results.append({
            "Model": model_name,
            "Test Accuracy": test_acc,
            "Training Accuracy": training_acc,
            "Precision": precision,
            "Recall": recall,
            "Size_MB": model_size,
            "Flops_K": flops,
            "Max_RAM_KB": max_ram,
            "Param_Memory_KB": param_mem,
            "Total_Memory_KB": total_ram_mem,
            "Training_Time": training_time
        })

        K.clear_session()
    else:
        print(f"⚠️ Model {model_name} was skipped due to excessive memory usage.")
end_time = time.time()  # Record end time
total_time = end_time - start_time

print(f"\n⏳ Total Script Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/Configurations_Big_Run_Taku_Stages_CIRA100.csv', index=False)
    print("✅ Results saved to CSV.")
else:
    print("⚠️ No models were trained due to memory constraints.")


