import json
import os
import time
import random
import psutil # type: ignore
import tensorflow as tf # type: ignore
import keras_tuner as kt # type: ignore
import pandas as pd
from tensorflow.keras import layers, models, backend as K, regularizers # type: ignore
from models.takunet import create_takunet_model
from Training.train_and_evaluate import train_and_evaluate_model
from Counting.peak_ram import estimate_max_memory_usage

# Enable GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

output_class = config["model_search_space"]["num_output_classes"]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = (x_train // 255).astype("int8"), (x_test // 255).astype("int8")  # Convert to int8
y_train, y_test = y_train.astype("int8"), y_test.astype("int8")  # Convert labels to int8


# Augmentation (includes CutMix & MixUp)
def cutmix(image, label, alpha=1):
    """Applies CutMix augmentation using integer-based operations."""
    # Ensure lambda_value is an integer between 0 and 1
    lambda_value = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    # Compute cutout dimensions
    height, width = tf.shape(image)[1], tf.shape(image)[2]

    # Convert cut_ratio to int before multiplication
    cut_ratio = tf.sqrt(tf.cast(1 - lambda_value, tf.float32))  # Keep float calculations safe
    cut_ratio_int = tf.cast(cut_ratio * 100, tf.int32) // 100  # Convert safely to int

    cut_height = (height * cut_ratio_int) // 100  # Ensure integer multiplication
    cut_width = (width * cut_ratio_int) // 100  # Ensure integer multiplication

    # Ensure random center coordinates use explicit minval and maxval
    cx = tf.random.uniform(shape=(), minval=0, maxval=width, dtype=tf.int32)
    cy = tf.random.uniform(shape=(), minval=0, maxval=height, dtype=tf.int32)

    # Compute bounding box coordinates
    x1 = tf.clip_by_value(cx - cut_width // 2, 0, width)
    x2 = tf.clip_by_value(cx + cut_width // 2, 0, width)
    y1 = tf.clip_by_value(cy - cut_height // 2, 0, height)
    y2 = tf.clip_by_value(cy + cut_height // 2, 0, height)

    # Create the binary mask with integer dtype
    mask = tf.pad(tf.ones((y2 - y1, x2 - x1, 3), dtype=tf.int32),
                  [[y1, height - y2], [x1, width - x2], [0, 0]])

    # Shuffle image and label to mix with another sample
    shuffled_image = tf.random.shuffle(image)
    shuffled_label = tf.random.shuffle(label)

    # Apply CutMix using integer-based computation
    image = tf.cast(image, tf.int32) * (1 - mask) + tf.cast(shuffled_image, tf.int32) * mask
    label = tf.cast(label, tf.int32) * lambda_value + tf.cast(shuffled_label, tf.int32) * (1 - lambda_value)

    return image, label





def preprocess_images(image, label):
    """Ensures all operations stay in integer format."""
    image = tf.image.convert_image_dtype(image, tf.int32)  # Ensure image is int32
    label = tf.cast(label, tf.int32)  # Ensure label is int32

    if tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32) == 1:
        image, label = cutmix(image, label)

    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess_images).batch(128).shuffle(10000)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# Create directory to save models
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 🔥 Define Hyperparameter Search Space for TakuNet
def build_hyper_takunet(hp):
    """Dynamically builds a TakuNet model with hyperparameters selected by Keras Tuner, while ensuring RAM constraints"""
    
    params = {
        "stages": hp.Int("stages", 2, 5),  # Tune number of stages (2 to 5)
        "filters_stem_1": hp.Choice("filters_stem_1", [32, 64, 128]),
        "filters_stem_2": hp.Choice("filters_stem_2", [32, 64, 128]),
        "filters_taku_block_1": hp.Choice("filters_taku_block_1", [64, 128, 256]),
        "filters_taku_block_2": hp.Choice("filters_taku_block_2", [64, 128, 256]),
        "kernel_size_stem_1": hp.Choice("kernel_size_stem_1", [3, 5, 7]),
        "kernel_size_stem_2": hp.Choice("kernel_size_stem_2", [3, 5, 7]),
        "kernel_size_taku_block_1": hp.Choice("kernel_size_taku_block_1", [3, 5]),
        "kernel_size_taku_block_2": hp.Choice("kernel_size_taku_block_2", [3, 5]),
        "kernel_size_refinement_block": hp.Choice("kernel_size_refinement_block", [3, 5, 7]),
        "dropout_rate": hp.Float("dropout_rate", 0.1, 0.4, step=0.1),
        "activation": hp.Choice("activation", ["relu", "gelu"]),
        "num_output_classes": output_class  # From config.json
    }

    # Create model
    model = create_takunet_model(params=params)

    # 🔥 RAM Pre-Check Before Returning Model
    max_ram_usage, _, _ = estimate_max_memory_usage(model=model, data_dtype_multiplier=4)  # Assume float32
    
    if max_ram_usage * 1024 > config["train_and_evaluate"]["evaluation_config"]["max_ram_consumption"]:
        print(f"🚨 Skipping model due to RAM limit: {max_ram_usage:.2f} KB exceeds {config['train_and_evaluate']['evaluation_config']['max_ram_consumption']} KB.")
        return None  # Reject model if it exceeds RAM limit
    
    return model


# 🔥 Bayesian Optimization for Hyperparameter Tuning
tuner = kt.BayesianOptimization(
    build_hyper_takunet,
    objective="val_accuracy",
    max_trials=20,  # Try 20 different configurations
    executions_per_trial=1,
    directory="taku_tuning",
    project_name="CIRA100_HPO"
)

print("\n🚀 Starting Hyperparameter Tuning...")
start_time = time.time()
tuner.search(train_ds, epochs=10, validation_data=test_ds, verbose=2)

# Get top 3 best models
best_hps = tuner.get_best_hyperparameters(num_trials=3)

# Train and save the top 3 models
top_models = []
for i, hps in enumerate(best_hps):
    best_model = build_hyper_takunet(hps)
    best_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"\n🏆 Training Model {i+1} (Top-{i+1} Configuration)...")
    history = best_model.fit(train_ds, epochs=40, validation_data=test_ds, verbose=2)
    
    # Evaluate and save results
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    filename = f'saved_models/best_takunet_{i+1}_{test_acc:.4f}.h5'
    best_model.save(filename)
    
    top_models.append({"Model": filename, "Test Accuracy": test_acc})
    print(f"✅ Model {i+1} saved as {filename}")

end_time = time.time()
total_time = end_time - start_time

print(f"\n✅ Hyperparameter Tuning Completed in {total_time/60:.2f} minutes")

# Save results to CSV
df_results = pd.DataFrame(top_models)
df_results.to_csv("results/TakuNet_Top3_CIRA100.csv", index=False)
print("📊 Top 3 models saved to CSV!")
