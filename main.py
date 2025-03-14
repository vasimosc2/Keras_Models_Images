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
from models.takunet import TakuNet
import tensorflow as tf  # type: ignore
import pandas as pd
import numpy as np
from tensorflow.keras import layers # type: ignore
from tensorflow.keras import backend as K  # type: ignore
import os
import time 

# Enable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Ensure TensorFlow uses GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ I found 1 and I will use a GPU located at: {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️ No GPU found, running on CPU.")



def sample_from_search_space(model_search_space: dict) -> dict:
    """Randomly select hyperparameters from the search space while maintaining the hierarchical structure."""
    params = {
        "stem_block": {
            "filters": random.choice(model_search_space["stem_block"]["filters"]),
            "Conv_kernel": random.choice(model_search_space["stem_block"]["Conv_kernel"]),
            "strides": random.choice(model_search_space["stem_block"]["strides"]),
            "dilation_rate": random.choice(model_search_space["stem_block"]["dilation_rate"]),
            "DWConv_kernel": random.choice(model_search_space["stem_block"]["DWConv_kernel"]),
        },
        "stages_block": {
            "stages_number": random.choice(model_search_space["stages_block"]["stages_number"]),
            "taku_block": {
                "taku_block_number": random.choice(model_search_space["stages_block"]["taku_block"]["taku_block_number"]),
                "DWConv_kernel": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_kernel"]),
            },
            "downsampler": {
                "Conv_kernel": random.choice(model_search_space["stages_block"]["downsampler"]["Conv_kernel"]),
            }
        },
        "refiner_block": {
            "DWConv_kernel": random.choice(model_search_space["refiner_block"]["DWConv_kernel"]),
            "num_output_classes": model_search_space["refiner_block"]["num_output_classes"]
        }
    }
    return params


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
output_class:int = config["model_search_space"]["refiner_block"]["num_output_classes"]

y_train = tf.keras.utils.to_categorical(y_train, output_class)
y_test = tf.keras.utils.to_categorical(y_test, output_class)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

def create_augmented_dataset(x, y):
    """Creates an augmented dataset efficiently using tf.data."""
    
    # Convert numpy arrays to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Apply augmentation only to images
    aug_dataset = dataset.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch to optimize performance
    aug_dataset = aug_dataset.batch(128).prefetch(tf.data.AUTOTUNE)  # Batching speeds up processing

    # Convert dataset back to tensors (efficient)
    x_aug_list, y_aug_list = [], []
    for img_batch, label_batch in aug_dataset:
        x_aug_list.append(img_batch)
        y_aug_list.append(label_batch)

    # Concatenate batched tensors
    x_aug = tf.concat(x_aug_list, axis=0)
    y_aug = tf.concat(y_aug_list, axis=0)

    # Concatenate original & augmented datasets
    x_combined = tf.concat([x, x_aug], axis=0)
    y_combined = tf.concat([y, y_aug], axis=0)  # Duplicate labels

    return x_combined, y_combined


print("🔄 Augmenting dataset...")
x_train_final, y_train_final = create_augmented_dataset(x_train, y_train)
#x_test = (x_test * 255).numpy().astype(np.uint8)
print(f"✅ Dataset Size Doubled: {x_train.shape[0]} → {x_train_final.shape[0]} images")



os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

models_to_train = {}

for i in range(1, 21):
    params = sample_from_search_space(config["model_search_space"])
    stages = params["stages_block"]["stages_number"]
    taku_blocks = params["stages_block"]["taku_block"]["taku_block_number"]
    print(f"The random params selected for model_{i} are:\n{json.dumps(params, indent=4)}")
    model_name = f"TakuNet Random_{i} (Stages: {stages} Blocks{taku_blocks})"
    models_to_train[model_name] = TakuNet(input_shape=(32,32,3),params=params)


results = []
print("I am starting the training\n")
start_time = time.time()
for model_name, model in models_to_train.items():
    
    print(f"\nTraining {model_name}...")
    
    
    results_data = train_and_evaluate_model(model, x_train_final, y_train_final, x_test, y_test, model_name, 
                                            params=sample_from_train_and_evaluate(config["train_and_evaluate"]))

    if results_data is not None:

        test_acc,tflite_acc, training_acc, precision, recall, keras_model_size, tf_model_size, c_array_size, flops, max_ram, param_mem, total_ram_mem, training_time = results_data
        
        results.append({
            "Model": model_name,
            "Test Accuracy": test_acc,
            "Tensor Flow Light Accuracy" : tflite_acc,
            "Training Accuracy": training_acc,
            "Precision": precision,
            "Recall": recall,
            "Keras Size_KB": keras_model_size,
            "Tensor Flow Light Size_KB": tf_model_size,
            "C array Size_KB ": c_array_size,
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
    df_results.to_csv('results/New_TakuNet_GPU_AdamW_Run.csv', index=False)
    print("✅ Results saved to CSV.")
else:
    print("⚠️ No models were trained due to memory constraints.")


