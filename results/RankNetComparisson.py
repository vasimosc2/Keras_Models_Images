import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from SurrogateComparisson.Embedding import simple_architecture_embedding
from TakuNet import TakuNetModel

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
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

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
        acc = row["test_accuracy"]
        ram = row["ModelRam"]
        flash = row["estimatedFlash"]

        models.append({
            "name": model_name,
            "embedding": emb,
            "fitness": fitness(acc, ram, flash)
        })

    except Exception as e:
        print(f"❌ Failed to load or embed {model_name}: {e}")

print(f"✅ Loaded {len(models)} models.")

# === Tournament Evaluation ===
total_runs = 10
total_matches = 0
total_errors = 0

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

        # Compare
        if ranknet_winner["name"] != true_winner["name"]:
            total_errors += 1

        total_matches += 1
        i += 2

# === Final Report ===
print("\n📊 RankNet Tournament Evaluation")
print(f"🔁 Total runs: {total_runs}")
print(f"🎯 Total matches: {total_matches}")
print(f"❌ Incorrect predictions: {total_errors}")
print(f"⚠️ Error rate: {100 * total_errors / total_matches:.2f}%")