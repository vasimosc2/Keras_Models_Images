import os
import random
import pandas as pd

"""
Script: evaluate_Performance_stopping_vs_full_training.py

Description:
This script evaluates the effectiveness of Performance stopping strategies by comparing Performance validation accuracy
(e.g., after X epochs) with the final accuracy from full training (e.g., 70 epochs).

Key Features:
- Loads a comparison CSV containing validation accuracies at Performance and full training stages.
- Computes the total training time saved due to Performance stopping.
- Defines a fitness function based on validation accuracy, RAM usage, and flash memory consumption.
- Performs tournament-style evaluations to assess how well early or partial training accuracy predicts
  final model performance.
- Reports error rates for two comparisons:
    • Performance stopping accuracy vs. full training accuracy -> (So optimization Based on Learning Rate + Perfomance Stop) 🛑 Stopped Accuracy
    • Full training at X epochs vs. 70-epoch accuracy -> (Only optimization of Learning Rate)

Intended Use:
    • See the time and error rate by incomperating this Performance Based optimization, in the already used Learning Rate Optimization

"""


# Load the CSV
results = "results"
epochNumbers:int = 70
history_folder = os.path.join(results, f"{str(epochNumbers)}-epochs")
csvPath = os.path.join(history_folder, f"val_accuracy_comparison_{str(epochNumbers)}.csv")
df = pd.read_csv(csvPath)

# --- Compute time saved from early stopping ---
if "Epochs_Saved" in df.columns and "Time Per Epoch (sec)" in df.columns:
    df["Time_Saved_sec"] = df["Epochs_Saved"] * df["Time Per Epoch (sec)"]
    total_time_saved_sec = df["Time_Saved_sec"].sum()
    total_time_saved_min = total_time_saved_sec / 60
    total_time_saved_hr = total_time_saved_min / 60

    total_time = (df["Time Per Epoch (sec)"] * epochNumbers).sum()
    print(f"\n💡 Estimated Time Savings from Early Stopping:")
    print(f"📦 Total models: {len(df)}\n")
    print(f"⏱️ Total time saved: {total_time_saved_sec:.2f} seconds")
    print(f"🕒 Total time saved: {total_time_saved_min:.2f} minutes")
    print(f"⏳ Total time saved: {total_time_saved_hr:.2f} hours\n")

    print(f"⏱️ Total time: {total_time / 3600 :.2f} hours")
    print(f"⏱️ Percentage Saved: {total_time_saved_sec * 100 / total_time:.2f} %")

# Constants
MAX_RAM = 197.68
MAX_FLASH = 754.921875

# Fitness function
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 70 * acc + 20 * norm_ram + 10 * norm_flash

# Comparison columns
comparison_columns = {
    "Stopped_Val_Accuracy": "🛑 Stopped Accuracy",
    f"Val_Accuracy_{epochNumbers}_Epochs": f"📈 Best {epochNumbers}-Epoch Accuracy"
}

# Run tournament evaluation for each column
total_runs = 1000

for comparison_col, label in comparison_columns.items():
    total_errors = 0
    total_matches = 0
    fitness_mse_wrong = 0.0

    for run in range(total_runs):
        shuffled = random.sample(list(df.to_dict('records')), len(df))
        i = 0
        while i < len(shuffled) - 1:
            m1 = shuffled[i]
            m2 = shuffled[i + 1]

            f1_real = compute_fitness(m1["Original_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_real = compute_fitness(m2["Original_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

            f1_est = compute_fitness(m1[comparison_col], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_est = compute_fitness(m2[comparison_col], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

            real_winner = 1 if f1_real > f2_real else 2
            est_winner = 1 if f1_est > f2_est else 2

            if real_winner != est_winner:
                total_errors += 1
                fitness_mse_wrong += (f1_real - f2_real) ** 2

            total_matches += 1
            i += 2

    mse = fitness_mse_wrong / total_errors if total_errors > 0 else 0.0
    rmse = mse ** 0.5

    print(f"\n🔍 {label}")
    print(f"🔁 Total runs: {total_runs}")
    print(f"🎯 Total matches: {total_matches}")
    print(f"❌ Total mismatches: {total_errors}")
    print(f"⚠️ Error rate: {100 * total_errors / total_matches:.2f}%")
    print(f"✏️ MSE (Wrong Predictions): {mse:.2e}")
    print(f"🔢 RMSE (Wrong Predictions): ±{rmse:.4f}")

