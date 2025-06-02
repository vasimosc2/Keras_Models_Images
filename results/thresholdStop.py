import pandas as pd
import os
import random
from pathlib import Path
import numpy as np

# --- Settings ---
epochs_total = 70
results_dir = Path("results") / f"{epochs_total}-epochs"
history_files = list(results_dir.glob("*_history.csv"))
ground_truth_file = results_dir / f"Retraining_{epochs_total}.csv"
cutoff_epochs = list(range(5, 30, 5))  # 5, 10, ..., 25
thresholds = np.round(np.arange(0.15, 0.4, 0.02), 3)  # val_acc thresholds

MAX_RAM = 197.68
MAX_FLASH = 754.921875
total_runs = 1000  # Tournament simulation count
max_error_allowed = 35.0  # Max accepted error rate in percent

# --- Load Ground Truth and Epoch Timing ---
ground_truth = pd.read_csv(ground_truth_file).set_index("Model")
ground_truth["Time Per Epoch (sec)"] = (ground_truth["Training Time (min)"] * 60) / epochs_total

# --- Fitness Function ---
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

# --- Tournament Evaluation ---
def evaluate_tournament(df, acc_col):
    total_errors = 0
    total_matches = 0
    records = list(df.to_dict("records"))

    for _ in range(total_runs):
        shuffled = random.sample(records, len(records))
        for i in range(0, len(shuffled) - 1, 2):
            m1 = shuffled[i]
            m2 = shuffled[i + 1]

            f1_real = compute_fitness(m1["Original_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_real = compute_fitness(m2["Original_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])
            f1_est = compute_fitness(m1[acc_col], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
            f2_est = compute_fitness(m2[acc_col], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

            if (f1_real > f2_real and f1_est <= f2_est) or (f2_real > f1_real and f2_est <= f1_est):
                total_errors += 1
            total_matches += 1

    return total_errors / total_matches

# --- Grid Search Over (Epoch, Threshold) ---
all_results = []

for cutoff in cutoff_epochs:
    for threshold in thresholds:
        rows = []

        for file in history_files:
            model = file.stem.replace("_history", "")
            try:
                df = pd.read_csv(file)
                if model not in ground_truth.index or len(df) <= cutoff:
                    continue

                stop_epoch = cutoff if df.loc[cutoff, "val_accuracy"] < threshold else len(df) - 1
                val_acc = df.loc[stop_epoch, "val_accuracy"]
                orig_acc = df["val_accuracy"].max()
                epochs_saved = max(0, epochs_total - stop_epoch - 1)
                time_per_epoch = ground_truth.loc[model, "Time Per Epoch (sec)"]

                rows.append({
                    "Model": model,
                    "Stopped_Val_Accuracy": val_acc,
                    "Original_Val_Accuracy": orig_acc,
                    "Epochs_Saved": epochs_saved,
                    "Model RAM (KB)": ground_truth.loc[model, "Model RAM (KB)"],
                    "Estimated Flash Memory (KB)": ground_truth.loc[model, "Estimated Flash Memory (KB)"],
                    "Time Per Epoch (sec)": time_per_epoch
                })

            except Exception as e:
                print(f"⚠️ Error processing {model}: {e}")

        df_simulated = pd.DataFrame(rows)
        if df_simulated.empty:
            continue

        # Evaluate tournament error
        error_rate = evaluate_tournament(df_simulated, "Stopped_Val_Accuracy")

        # Use real model training time to compute total time saved and % saved
        time_saved_total = 0
        total_full_time = 0

        for row in rows:
            model = row["Model"]
            epochs_saved = row["Epochs_Saved"]
            full_training_time_sec = ground_truth.loc[model, "Training Time (min)"] * 60
            time_per_epoch = full_training_time_sec / epochs_total
            time_saved_model = epochs_saved * time_per_epoch

            time_saved_total += time_saved_model
            total_full_time += full_training_time_sec

        percent_saved = (time_saved_total / total_full_time) * 100 if total_full_time > 0 else 0

        # Append results
        all_results.append({
            "Cutoff_Epoch": cutoff,
            "Val_Threshold": threshold,
            "Error_Rate (%)": round(100 * error_rate, 2),
            "Total_Time_Saved_min": round(time_saved_total / 60, 2),
            "Time_Saved (%)": round(percent_saved, 2)
        })

# --- Output Results ---
df_summary = pd.DataFrame(all_results).sort_values(by="Error_Rate (%)")

# 📊 Top 10 by lowest error
print("\n📊 Top 10 Configurations by Lowest Error Rate:")
print(df_summary.head(10))

# 💰 Best config under allowed error
df_filtered = df_summary[df_summary["Error_Rate (%)"] <= max_error_allowed]

if not df_filtered.empty:
    best_time_saved_row = df_filtered.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True]).iloc[0]
    print(f"\n💰 Best Config (≤ {max_error_allowed:.0f}% Error):")
    print(best_time_saved_row.to_string())

    # Save filtered configurations
    #df_filtered.to_csv(results_dir / "best_configs_under_25_percent_error.csv", index=False)
else:
    print(f"\n⚠️ No configurations found with error rate ≤ {max_error_allowed:.0f}%.")

# ✅ Show only top 5 valid configs (≤ 10% error), sorted by time saved
df_valid = df_summary[df_summary["Error_Rate (%)"] <= max_error_allowed]
df_valid_sorted = df_valid.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True])

if not df_valid_sorted.empty:
    top_5_configs = df_valid_sorted.head(5)
    print(f"\n✅ Top 5 Configurations with ≤ {max_error_allowed:.0f}% Error Rate, Sorted by Time Saved:")
    print(top_5_configs.to_string(index=False))
    # Save to file
    #top_5_configs.to_csv(results_dir / f"top_5_configs_under_{int(max_error_allowed)}_percent_error_sorted.csv", index=False)
else:
    print(f"\n⚠️ No configurations found with error rate ≤ {max_error_allowed:.0f}%.")
