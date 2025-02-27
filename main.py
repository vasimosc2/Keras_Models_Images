import os
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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Ensure directories exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)


# List of models to train
models_to_train = {
    "TakuNet 2 stages": create_takunet_model(stages=2),
    "TakuNet 3 stages": create_takunet_model(stages=3),
    "TakuNet 4 stages": create_takunet_model(stages=4),
    "TakuNet 5 stages": create_takunet_model(stages=5),
    "TakuNet 6 stages": create_takunet_model(stages=6),
    "TakuNet 7 stages": create_takunet_model(stages=7),
    "TakuNet 8 stages": create_takunet_model(stages=8),
    "TakuNet 9 stages": create_takunet_model(stages=9),
    "TakuNet 10 stages": create_takunet_model(stages=10),
    # "TakuNet 2 stages + Normal": create_takunet_model(stages=2, extra_layer_inside_taku=None, extra_layer_outside_taku= None, l2_reg=None),
    # "TakuNet 2 stages + DropOut 0.5": create_takunet_model(stages=2, extra_layer_inside_taku=layers.Dropout(0.5), extra_layer_outside_taku= None, l2_reg=None),
    # "TakuNet 2 stages + DropOut 0.3 + DropOut 0.1": create_takunet_model(stages=2, extra_layer_inside_taku=layers.Dropout(0.3), extra_layer_outside_taku =layers.Dropout(0.1), l2_reg=None),
    # "TakuNet 2 stages + DropOut 0.2  + DropOut 0.1": create_takunet_model(stages=2, extra_layer_inside_taku=layers.Dropout(0.2), extra_layer_outside_taku = layers.Dropout(0.1), l2_reg=None),
    # "TakuNet 2 stages + L2 Regulation 0.01": create_takunet_model(stages=2, extra_layer_inside_taku=None, extra_layer_outside_taku = layers.Dropout(0.3), l2_reg=0.01),
    # "TakuNet 2 stages + DropOut 0.3 + L2 Regulation 0.01": create_takunet_model(stages=2, extra_layer_inside_taku=layers.Dropout(0.3), extra_layer_outside_taku= None, l2_reg=0.01),
    # "Simple_CNN": create_simple_cnn(),
    # "Deeper_CNN": create_deeper_cnn(),
    # "CNN_With_Dropout": create_cnn_with_dropout(),
    # "CNN_With_BatchNorm": create_cnn_with_batchnorm(),
    # "CNN_With_GAP": create_cnn_with_gap(),
    # "ResNet_Like_CNN": create_resnet_like_cnn()
}

results = []

for model_name, model in models_to_train.items():
    print(f"\nTraining {model_name}...")
    
    # Train and evaluate the original model
    test_acc, training_acc, precision, recall, model_size, flops, max_ram, param_mem, total_ram_mem, training_time = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)

    # Store original model results
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

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('results/evaluation_Taku_Stages_CIRA100.csv', index=False)

