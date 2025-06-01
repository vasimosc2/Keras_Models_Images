import os
import random
import pandas as pd

# Load the CSV
results = "results"
epochNumbers = 20
history_folder = os.path.join(results, f"{epochNumbers}-epochs")
csvPath = os.path.join(history_folder, f"val_accuracy_comparison_{epochNumbers}.csv")
df = pd.read_csv(csvPath)

# --- Compute time saved from early stopping ---
if "Epochs_Saved" in df.columns and "Time Per Epoch (sec)" in df.columns:
    df["Time_Saved_sec"] = df["Epochs_Saved"] * df["Time Per Epoch (sec)"]
    total_time_saved_sec = df["Time_Saved_sec"].sum()
    total_time_saved_min = total_time_saved_sec / 60
    total_time_saved_hr = total_time_saved_min / 60

    print(f"\n💡 Estimated Time Savings from Early Stopping:")
    print(f"📦 Total models: {len(df)}")
    print(f"⏱️ Total time saved: {total_time_saved_sec:.2f} seconds")
    print(f"🕒 Total time saved: {total_time_saved_min:.2f} minutes")
    print(f"⏳ Total time saved: {total_time_saved_hr:.2f} hours")

# Constants
MAX_RAM = 197.68
MAX_FLASH = 754.921875

# Fitness function
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

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

    for run in range(total_runs):
        shuffled = random.sample(list(df.to_dict('records')), len(df))
        i = 0
        while i < len(shuffled) - 1:
            m1 = shuffled[i]
            m2 = shuffled[i + 1]

            # Real fitness using 70-epoch accuracy
            f1_real = compute_fitness(m1["Original_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_real = compute_fitness(m2["Original_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

            # Estimated fitness using comparison column
            f1_est = compute_fitness(m1[comparison_col], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_est = compute_fitness(m2[comparison_col], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

            real_winner = 1 if f1_real > f2_real else 2
            est_winner = 1 if f1_est > f2_est else 2

            if real_winner != est_winner:
                total_errors += 1

            total_matches += 1
            i += 2

    # Output for this comparison
    print(f"\n🔍 {label}")
    print(f"🔁 Total runs: {total_runs}")
    print(f"🎯 Total matches: {total_matches}")
    print(f"❌ Total mismatches: {total_errors}")
    print(f"⚠️ Error rate: {100 * total_errors / total_matches:.2f}%")
