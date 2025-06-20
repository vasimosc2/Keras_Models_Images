import argparse
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
import glob
import random


"""
optimizationFindPerformance.py

This script performs a comprehensive grid search to identify the best early stopping configuration 
to reduce training time in Neural Architecture Search (NAS), while preserving the fidelity of 
model selection based on accuracy and resource constraints.

🔍 Purpose:
----------
The script tests combinations of two performance-based early stopping strategies:
1. **No-improvement Rule**: Stops training if validation accuracy does not improve after 
   a number of consecutive epochs (patience).
2. **Diminishing-returns Rule**: Stops training if the relative improvement over a 
   window (defined as a fraction alpha of total epochs) is less than a minimum threshold (epsilon).

📈 Methodology:
--------------
For each model's training history:
- Simulates early stopping using both rules and records the earliest stopping epoch.
- Calculates the validation accuracy at the stopping point and compares it to full training.
- Computes how much training time would be saved if this early stopping had been applied.

💡 Evaluation:
-------------
- Fitness is defined as a weighted combination of validation accuracy, RAM usage, and Flash usage.
- A randomized tournament is performed to compare the model rankings using full training vs early stopping.
- Misranking error rate and RMSE are computed to quantify the impact of early stopping.

📦 Output:
---------
- Best configuration (patience, alpha, epsilon) that achieves the lowest misranking error 
  under a user-defined threshold and saves the most training time.
- Summary statistics including error rate, RMSE, MSE, time saved, and percentage improvement.
- Resulting performance-stopped dataset is saved for future use.

⚙️ Args:
-------
--partialEpochs             : Number of epochs in the current training run (default: 30)
--max_error_allowed         : Maximum acceptable ranking error in percentage (default: 25.0)

📂 Files:
--------
- Reads model history CSVs (e.g., `TakuNet_Init_1_history.csv`) from the appropriate results folder.
- Loads baseline accuracies from `Retraining_70.csv`.
- Saves best-config results to `PerformanceStoppage_{EPOCHS}.csv`.

🔁 Grid Search Details:
----------------------
Explores multiple values of:
- Patience: [4 to 11]
- Alpha: [0.1, 0.15, 0.2, 0.25, 0.3]
- Epsilon: [0.01, 0.03, 0.05, 0.07]

💬 Example:
----------
python optimizationFindPerformance.py --partialEpochs 30 --max_error_allowed 20

"""

parser = argparse.ArgumentParser()
parser.add_argument("--partialEpochs", type=int, default=70, help="Partial Training Epochs (Strategy 1)")
parser.add_argument("--max_error_allowed", type=float, default=25.0, help="Maximun Allowed Error Rate in Comparissons (Strategy 1)")

args = parser.parse_args()

# --- Settings ---
results = "results"
epochNumbers = args.partialEpochs
epochTruth = 70
max_error_rate = args.max_error_allowed # Maximum allowed misranking error (%)

history_folder = os.path.join(results, f"{epochNumbers}-epochs")
history_files = glob.glob(os.path.join(history_folder, "*_history.csv"))
val_acc_path = os.path.join(history_folder, f"PerformanceStoppage_{epochNumbers}.csv")
nas_results_path = os.path.join(history_folder, f"Retraining_{epochNumbers}.csv")

# --- Utility for formatted training time ---
def print_training_time(total_minutes: float, label: str) -> None:
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n⏱️ Total Training Time ({label}):")
    print(f"🕒 {total_minutes:.2f} minutes")
    print(f"🕓 ≈ {hours}h {minutes}m")

# --- Load 70-epoch reference results ---
nas_70_path = os.path.join(results, f"{epochTruth}-epochs")
df_70 = pd.read_csv(os.path.join(nas_70_path, f"Retraining_{epochTruth}.csv"))
df_70 = df_70[["Model", "Best Test Accuracy"]].rename(columns={"Best Test Accuracy": "Original_Val_Accuracy"})

# --- Load runtime + memory stats ---
df_nas = pd.read_csv(nas_results_path)
if "Time Per Epoch (sec)" not in df_nas.columns and "Training Time (min)" in df_nas.columns:
    df_nas["Time Per Epoch (sec)"] = (df_nas["Training Time (min)"] * 60) / epochNumbers
df_nas = df_nas[["Model", "Model RAM (KB)", "Estimated Flash Memory (KB)", "Time Per Epoch (sec)"]]

# --- Early stopping (no improvement) ---
def simulate_performance_early_stopping(df: pd.DataFrame, monitor: str = 'val_accuracy', patience: int = 10) -> Tuple[int, str]:
    best_val = float('-inf')
    best_epoch = 0
    patience_counter = 0
    for epoch in range(len(df)):
        current_val = df.loc[epoch, monitor]
        if current_val > best_val:
            best_val = current_val
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            return best_epoch, "early_stopping"
    return len(df) - 1, "none"

# --- Diminishing-returns stopping ---
def simulate_diminishing_return_performance_stopping(df: pd.DataFrame, monitor: str = 'val_accuracy', alpha: float = 0.2, epsilon: float = 0.1) -> Tuple[int, str]:
    total_epochs = len(df)
    window_size = int(alpha * total_epochs)
    if window_size < 1:
        return total_epochs - 1, "none"

    for start_epoch in range(0, total_epochs - window_size):
        acc_start = df.loc[start_epoch, monitor]
        acc_end = df.loc[start_epoch + window_size, monitor]
        if acc_start == 0:
            continue  # Avoid division by zero
        relative_improvement = (acc_end - acc_start) / acc_start
        if relative_improvement < epsilon:
            return start_epoch + window_size, "performance_stopping"
    
    return total_epochs - 1, "none"

# --- Fitness Function ---
MAX_RAM = 197.68
MAX_FLASH = 754.921875

def compute_fitness(acc: float, ram: float, flash: float) -> float:
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 70 * acc + 20 * norm_ram + 10 * norm_flash

# --- Grid Search ---
total_runs = 1000
best_candidate: Optional[Dict] = None
candidate_list: List[Dict] = []

for patience in range(4, 12):
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for epsilon in [0.01, 0.03, 0.05, 0.07]:
            combined_results = []
            for file_path in history_files:
                try:
                    df = pd.read_csv(file_path)
                    model_name = os.path.basename(file_path).replace("_history.csv", "")
                    total_epochs = len(df)

                    early_epoch, _ = simulate_performance_early_stopping(df, patience=patience)
                    perf_epoch, _ = simulate_diminishing_return_performance_stopping(df, alpha=alpha, epsilon=epsilon)

                    if early_epoch < total_epochs - 1 or perf_epoch < total_epochs - 1:
                        stop_epoch = min(early_epoch, perf_epoch)
                        stop_reason = "early_stopping" if early_epoch <= perf_epoch else "performance_stopping"
                    else:
                        stop_epoch = total_epochs - 1
                        stop_reason = "none"

                    stopped_val = df.loc[stop_epoch, 'val_accuracy']
                    best_val = df['val_accuracy'].max()

                    combined_results.append({
                        'Model': model_name,
                        f'Val_Accuracy_{epochNumbers}_Epochs': best_val,
                        'Stopped_Val_Accuracy': stopped_val,
                        'Stop_Reason': stop_reason,
                        'Epochs_Saved': total_epochs - stop_epoch - 1
                    })
                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {e}")

            df_val = pd.DataFrame(combined_results)
            df_val = pd.merge(df_val, df_70, on="Model", how="left")
            df_val["Original_Val_Accuracy"] = df_val["Original_Val_Accuracy"].fillna(df_val[f'Val_Accuracy_{epochNumbers}_Epochs'])
            df_val = pd.merge(df_val, df_nas, on="Model", how="left")
            df_val["Time_Saved_sec"] = df_val["Epochs_Saved"] * df_val["Time Per Epoch (sec)"]

            # Misranking error evaluation
            total_errors = 0
            total_matches = 0
            fitness_mse_wrong = 0.0
            records = df_val.to_dict('records')

            for _ in range(total_runs):
                shuffled = random.sample(records, len(records))
                for i in range(0, len(shuffled) - 1, 2):
                    m1, m2 = shuffled[i], shuffled[i + 1]
                    f1r = compute_fitness(m1["Original_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
                    f2r = compute_fitness(m2["Original_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])
                    f1e = compute_fitness(m1["Stopped_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
                    f2e = compute_fitness(m2["Stopped_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])
                    if (f1r > f2r and f1e <= f2e) or (f2r > f1r and f2e <= f1e):
                        total_errors += 1
                        fitness_mse_wrong += (f1r - f2r) ** 2
                    total_matches += 1

            error_rate = 100 * total_errors / total_matches if total_matches else 0.0
            rmse = (fitness_mse_wrong / total_errors) ** 0.5 if total_errors else 0.0
            total_time_saved_min = df_val["Time_Saved_sec"].sum() / 60

            if error_rate <= max_error_rate and total_time_saved_min > 0:
                candidate_list.append({
                    "patience": patience,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "error_rate": error_rate,
                    "rmse": rmse,
                    "mse": fitness_mse_wrong / total_errors if total_errors else 0.0,
                    "time_saved_min": total_time_saved_min,
                    "df": df_val
                })

# --- Best Config Output ---

    desired_columns = [
    'Model',
    'Original_Val_Accuracy',
    f'Val_Accuracy_{epochNumbers}_Epochs',
    'Stopped_Val_Accuracy',
    'Stop_Reason',
    'Epochs_Saved',
    'Model RAM (KB)',
    'Estimated Flash Memory (KB)',
    'Time Per Epoch (sec)',
    'Time_Saved_sec'  # Make sure time saved is last
]
    
if candidate_list:
    best_candidate = max(candidate_list, key=lambda x: x["time_saved_min"])
    best_df = best_candidate["df"]
    ordered_columns = [col for col in desired_columns if col in best_df.columns]
    best_df = best_df[ordered_columns]
    total_models = len(df_nas)
    total_time_saved_min = best_candidate["time_saved_min"]
    total_possible_time_min = (df_nas["Time Per Epoch (sec)"] * epochNumbers).sum() / 60
    percentage_saved = (total_time_saved_min / total_possible_time_min) * 100 if total_possible_time_min > 0 else 0.0

    print("\n✅ Best configuration under", max_error_rate, "% error rate:")
    print(f"🔁 Patience: {best_candidate['patience']}")
    print(f"🪟 Alpha (Window Size %): {best_candidate['alpha']} (Epoch Diminishing Return: {best_candidate['alpha'] * total_epochs})")
    print(f"📉 Epsilon (Min Relative Improvement): {best_candidate['epsilon']}")
    print(f"❗ Error Rate: {best_candidate['error_rate']:.2f}%")
    print(f"📉 RMSE: {best_candidate['rmse']:.2e}")
    print(f"📉 MSE: {best_candidate['mse']:.2e}")
    print(f"⏱️ Time Saved: {total_time_saved_min:.2f} minutes")
    print(f"🧠 Total Models: {total_models}")
    print(f"💸 Percentage Time Saved: {percentage_saved:.2f}%")

    print_training_time(total_possible_time_min, label="Original (Full 30 Epochs)")
    print_training_time(total_possible_time_min - total_time_saved_min, label="With Early/Performance Stopping")
    print(f"\n💡 Training Time Saved: {total_time_saved_min:.2f} minutes")
    print(f"📉 Percentage Saved: {percentage_saved:.2f}%")

    # print("\n📋 Sample of Resulting DataFrame with Best Config:")
    # print(best_df.head(20).to_string(index=False))

    # Optional: Export to CSV


    # Save the reordered DataFrame
    best_df.to_csv(val_acc_path, index=False)

else:
    print(f"❌ No configurations found with error rate ≤ {max_error_rate}%")
