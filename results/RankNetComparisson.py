import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from SurrogateComparisson.Embedding import simple_architecture_embedding


"""
Script: evaluate_ranknet_tournament.py

Description:
This script evaluates the accuracy of a surrogate ranking model (RankNet) in predicting the relative performance
of neural network architectures discovered during a Neural Architecture Search (NAS) process.

The RankNet model compares architecture embeddings and predicts which model is better. This evaluation simulates
a tournament where pairs of models are compared repeatedly, and RankNet's predictions are validated against
a ground truth fitness function based on actual model performance.

Main Components:
- Loads architecture parameter files (`*_model_params.json`) and generates fixed-length embeddings using
  `simple_architecture_embedding()`.
- Computes ground truth fitness using a weighted formula combining accuracy, RAM usage, and flash memory.
- Loads a pre-trained RankNet model and uses it to predict pairwise model comparisons.
- Runs a large number of randomized pairwise matchups (tournaments) and records how often RankNet's predicted
  winner disagrees with the actual higher-fitness model.

Fitness Function:
\[
\text{fitness} = 0.7 \times \text{accuracy} + 0.2 \times (1 - \text{RAM}/\text{MAX\_RAM}) + 0.1 \times (1 - \text{Flash}/\text{MAX\_FLASH})
\]

Outputs:
- Total number of matches
- Total number of prediction errors
- Final error rate (%), indicating how often RankNet fails to rank the models correctly

Use Case:
This script is useful for validating the quality of a learned surrogate model before using it in future
NAS iterations or model selection pipelines.
"""


# === Paths ===
day="May-27"
params_folder = f"NAS/{day}/saved_configs/model_params"
results_file = f"NAS/{day}/results/Best_Models_Results_NAS.csv"
ranknet_path = "SurrogateComparisson/ranknet_model.keras"

# === Constants for fitness ===
MAX_RAM = 197.68
MAX_FLASH = 754.921875

def fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 70 * acc + 20 * norm_ram + 10 * norm_flash

def mse_error(true1, true2):
    return (true1 - true2) ** 2

# === Load results CSV ===
results_df = pd.read_csv(results_file)

# === Load RankNet model ===
ranknet = tf.keras.models.load_model(ranknet_path)
print("📦 RankNet model loaded.")

# === Prepare model list with embeddings and fitness ===
models = []

for _, row in results_df.iterrows():
    model_name = row["Model"]
    params_path = os.path.join(params_folder, f"{model_name}_model_params.json")

    try:
        with open(params_path) as f:
            params = json.load(f)

        emb = simple_architecture_embedding(params)
        acc = row["Best Test Accuracy"]
        ram = row["Model RAM (KB)"]
        flash = row["Estimated Flash Memory (KB)"]

        models.append({
            "name": model_name,
            "embedding": emb,
            "fitness": fitness(acc, ram, flash)
        })

    except Exception as e:
        print(f"❌ Failed to load or embed {model_name}: {e}")

print(f"✅ Loaded {len(models)} models.")

# === Tournament Evaluation ===
total_runs = 1000
total_matches = 0
total_errors = 0

fitness_mse_wrong = 0.0

for run in range(total_runs):
    shuffled = random.sample(models, len(models))  # Random shuffle
    i = 0
    while i < len(shuffled) - 1:
        m1 = shuffled[i]
        m2 = shuffled[i + 1]

        # True winner by fitness
        true_winner = m1 if m1["fitness"] >= m2["fitness"] else m2

        # RankNet prediction
        pred = ranknet.predict(
            [np.expand_dims(m1["embedding"], axis=0), np.expand_dims(m2["embedding"], axis=0)],
            verbose=0
        )
        ranknet_winner = m1 if pred[0][0] > 0.5 else m2

                # Calculate MSE for this pair
        true_diff = m1["fitness"] - m2["fitness"]
        pred_diff = pred[0][0] if ranknet_winner == m1 else -pred[0][0]
       

        if ranknet_winner["name"] != true_winner["name"]:
            total_errors += 1
            mse_value = mse_error(m1["fitness"],m2["fitness"])
            fitness_mse_wrong += mse_value

        total_matches += 1
        i += 2

# === Final Report ===
print("\n📊 RankNet Tournament Evaluation")
print(f"🔁 Total runs: {total_runs}")
print(f"🎯 Total matches: {total_matches}")
print(f"❌ Incorrect predictions: {total_errors}")
print(f"⚠️ Error rate: {100 * total_errors / total_matches:.2f}%")

print(f"📐 Avg Fitness MSE (wrong predictions only): {fitness_mse_wrong / total_errors:.2e}")
print(f"ℹ️ Fitness is normalized in [0, 1]. This implies ~±{(fitness_mse_wrong / total_errors) ** 0.5:.4f} avg prediction deviation (RMSE) on incorrect predictions.")
