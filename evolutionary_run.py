import json
import os
import time
import pandas as pd
import tensorflow as tf # type: ignore
from tensorflow.keras import backend as K # type: ignore




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

from search_strategy import EvolutionarySearch # type: ignore
# Load configuration
CONFIG_PATH = "config.json"

# Set evolutionary search parameters
POPULATION_SIZE:int = 10  # Number of models per generation
TIME:float = 3.0  # Number of hours to run
MUTATION_RATE:float = 0.2  # Probability of mutation per model
CROSSOVER_RATE:float = 0.3  # Probability of crossover between two models

# Ensure directories exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize evolutionary search
evo_search = EvolutionarySearch(CONFIG_PATH, POPULATION_SIZE, TIME, MUTATION_RATE, CROSSOVER_RATE)

# Run evolutionary search
best_models_data = []

# Run evolutionary search
for generation, best_model in enumerate(evo_search.evolve(), start=1):
    print(f"🔥 Best model of Generation {generation}: {best_model.model_name} with Accuracy: {best_model.results.test_accuracy:.4f}")

    # Save best model's data
    best_models_data.append({
        "Generation": generation,
        "Model": best_model.model_name,
        "Best Train Accuracy": best_model.results.train_accuracy,
        "Best Test Accuracy": best_model.results.test_accuracy,
        "TFlite Test Accuracy": best_model.results.tflite_accuracy,
        "Precision": best_model.results.precision,
        "Recall": best_model.results.recall,
        "F1 Score": best_model.results.f1_score,
        "Max RAM Usage (KB)": best_model.results.max_ram_usage,
        "TFlite Estimation size(KB)": best_model.results.tflite_size,
        "Param Memory (KB)": best_model.results.param_memory,
        "Total Memory (KB)": best_model.results.total_memory,
        "Training Time (s)": best_model.results.training_time
    })

print("✅ Evolutionary search complete!")

# Convert best models data to DataFrame and save to CSV
df_results = pd.DataFrame(best_models_data)
df_results.to_csv('results/Best_Models_Results.csv', index=False)
print(f"✅ All best models from each generation saved to CSV: results/Best_Models_Results.csv")
