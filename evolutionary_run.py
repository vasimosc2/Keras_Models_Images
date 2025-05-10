import json
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.backend.tensorflow.trainer")



Folder ='NAS'

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

from search_strategy import EvolutionarySearch
# Load configuration
CONFIG_PATH = "config.json"

# Set evolutionary search parameters

POPULATION_SIZE:int =6  # Number of models per generation
TIME:float = 5.0  # Number of hours to run
MUTATION_RATE:float = 0.2  # Probability of mutation per model
CROSSOVER_RATE:float = 0.3  # Probability of crossover between two models

# Ensure directories exist
os.makedirs(f'{Folder}/saved_models', exist_ok=True)
os.makedirs(f'{Folder}/results', exist_ok=True)


default_augementaion_technique ={ "apply_standard":False,
                            "apply_color":False,
                            "apply_geometric":False,
                            "apply_mixup": False,
                            "apply_cutmix": False}

# Initialize evolutionary search
evo_search = EvolutionarySearch(config_path=CONFIG_PATH, population_size=POPULATION_SIZE, time=TIME,
                                mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE, augmentation_techinque=default_augementaion_technique)

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
        "Estimated Max RAM Usage (KB)": best_model.results.estimatedMaxRam,
        "Accurate Max RAM Usage (KB)": best_model.results.AccurateMaxRam,
        "Model RAM (KB)": best_model.results.ModelRam,
        "Estimated Flash Memory (KB)": best_model.results.estimatedFlash,        
        "TFlite size(KB)": best_model.results.tflite_size,
        "Flop Number": best_model.results.flops,
        "Training Time (s)": best_model.results.training_time,
        "Epochs Trained": best_model.results.epochs_trained
    })

    hist_df = pd.DataFrame(best_model.results.history.history)
    hist_path = f'{Folder}/results/{best_model.model_name}_history.csv'
    hist_df.to_csv(hist_path, index=False)
    print(f"📊 Training history saved to: {hist_path}")

print("✅ Evolutionary search complete!")

# Convert best models data to DataFrame and save to CSV
df_results = pd.DataFrame(best_models_data)
df_results.to_csv(f'{Folder}/results/Best_Models_Results_NAS.csv', index=False)
print(f"✅ All best models from each generation saved to CSV: {Folder}/results/Best_Models_Results_NAS.csv")
