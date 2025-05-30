import os
import argparse
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.backend.tensorflow.trainer")



# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run Evolutionary Search for TakuNet Models")
parser.add_argument("--time", type=float, default=2.0, help="Total time to run the evolutionary search (in hours)")
parser.add_argument("--population_size", type=int, default=6, help="Number of models in each generation")

parser.add_argument(
    "--lr_strategy",
    type=str,
    default="cosine",
    choices=["cosine", "linear", "step"],
    help="Learning Rate strategy to be used to train TakuNet models: 'cosine', 'linear', or 'step'"
)
args = parser.parse_args()

today = datetime.now().strftime("%b-%d")
Folder ='NAS'
Folder = os.path.join(Folder, today)

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

POPULATION_SIZE:int= int(args.population_size)
TIME:float = float(args.time)
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
evo_search = EvolutionarySearch(config_path=CONFIG_PATH, 
                                population_size=POPULATION_SIZE, 
                                time=TIME,
                                mutation_rate=MUTATION_RATE, 
                                crossover_rate=CROSSOVER_RATE, 
                                augmentation_techinque=default_augementaion_technique,
                                folder=Folder,
                                strategy=args.lr_strategy)

# Run evolutionary search
models_data = []

# Run evolutionary search
for generation, model in enumerate(evo_search.evolve(), start=1):

    # Save  model's data
    models_data.append({
        "Model": model.model_name,
        "Best Train Accuracy": model.results.train_accuracy,
        "Best Test Accuracy": model.results.test_accuracy,
        "Swa Test Accuracy": model.results.SWA_test_accuracy,
        "TFlite Test Accuracy": model.results.tflite_accuracy,
        "Optimizer":model.model_params["optimizer"],
        "Precision": model.results.precision,
        "Recall": model.results.recall,
        "F1 Score": model.results.f1_score,
        "Model RAM (KB)": model.results.ModelRam,
        "Estimated Flash Memory (KB)": model.results.estimatedFlash,        
        "TFlite size(KB)": model.results.tflite_size,
        "Flop Number": model.results.flops,
        "Fitness Score": model.results.fitness_score,
        "Training Time (min)": round(model.results.training_time/60,2),
        "Epochs Trained": model.results.epochs_trained
    })

    hist_df = pd.DataFrame(model.results.history.history)

    os.makedirs(f'{Folder}/results/History', exist_ok=True)
    
    hist_path = f'{Folder}/results/History/{model.model_name}_history.csv'
    hist_df.to_csv(hist_path, index=False)
    print(f"📊 Training history saved to: {hist_path}")

print("✅ Evolutionary search complete!")

# Convert best models data to DataFrame and save to CSV
df_results = pd.DataFrame(models_data)
df_results.to_csv(f'{Folder}/results/Best_Models_Results_NAS.csv', index=False)
print(f"✅ All best models from each generation saved to CSV: {Folder}/results/Best_Models_Results_NAS.csv")


# --- Pareto Filtering ---

def is_pareto_efficient(models):
    pareto_set = []
    for i, m_i in enumerate(models):
        dominated = False
        for j, m_j in enumerate(models):
            if i != j:
                if (
                    m_j["Best Test Accuracy"] >= m_i["Best Test Accuracy"] and
                    m_j["Model RAM (KB)"] <= m_i["Model RAM (KB)"] and
                    m_j["Estimated Flash Memory (KB)"] <= m_i["Estimated Flash Memory (KB)"]
                ):
                    if (
                        m_j["Best Test Accuracy"] > m_i["Best Test Accuracy"] or
                        m_j["Model RAM (KB)"] < m_i["Model RAM (KB)"] or
                        m_j["Estimated Flash Memory (KB)"] < m_i["Estimated Flash Memory (KB)"]
                    ):
                        dominated = True
                        break
        if not dominated:
            pareto_set.append(m_i)
    return pareto_set

# Filter only Pareto-optimal models
pareto_models = is_pareto_efficient(models_data)

# Save to CSV
df_results = pd.DataFrame(pareto_models)
df_results.to_csv(f'{Folder}/results/Pareto_Optimal_Models.csv', index=False)
print(f"✅ Pareto-optimal models saved to: {Folder}/results/Pareto_Optimal_Models.csv")
