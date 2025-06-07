import pandas as pd
import random
from pathlib import Path
import numpy as np

# --- Settings ---
epochs_total = 30
ground_truth_epochs = 70
max_error_allowed = 18.0  # %

test_dir = Path("results") / f"{epochs_total}-epochs"
ground_dir = Path("results") / f"{ground_truth_epochs}-epochs"

history_files = list(test_dir.glob("*_history.csv"))
ground_truth_file = ground_dir / f"Retraining_{ground_truth_epochs}.csv"
runtime_file = test_dir / f"Retraining_{epochs_total}.csv"

cutoff_epochs = list(range(3, 15, 1))
thresholds = np.round(np.arange(0.15, 0.45, 0.02), 3)

MAX_RAM = 197.68
MAX_FLASH = 754.921875
total_runs = 1000

# --- Load Ground Truth Accuracy and Runtime Info ---
ground_truth_accuracy = pd.read_csv(ground_truth_file).set_index("Model")
runtime_info = pd.read_csv(runtime_file).set_index("Model")
runtime_info["Time Per Epoch (sec)"] = (runtime_info["Training Time (min)"] * 60) / epochs_total

# --- Fitness Function ---
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

# --- MSE Evaluation ---
def evaluate_tournament(df, acc_col):
    total_errors = 0
    total_matches = 0
    fitness_mse_wrong = 0
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
                fitness_mse_wrong += (f1_real - f2_real) ** 2

            total_matches += 1

    error_rate = total_errors / total_matches
    mse = fitness_mse_wrong / total_errors if total_errors > 0 else 0.0
    return error_rate, mse

# --- Time Formatting ---
def print_training_time(total_minutes):
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n ⏱️ Total Training Time :")
    print(f"🕓 ≈  {total_minutes:.2f} minutes")
    print(f"🕓 ≈  {hours}h {minutes}m")

# --- Grid Search ---
all_results = []

for cutoff in cutoff_epochs:
    for threshold in thresholds:
        rows = []

        for file in history_files:
            model = file.stem.replace("_history", "")
            try:
                df = pd.read_csv(file)
                if model not in ground_truth_accuracy.index or model not in runtime_info.index:
                    continue
                if len(df) <= cutoff:
                    continue

                cutoff_index = cutoff - 1
                stop_epoch = cutoff_index if df.loc[cutoff_index, "val_accuracy"] < threshold else len(df) - 1
                val_acc = df.loc[stop_epoch, "val_accuracy"]
                orig_acc = ground_truth_accuracy.loc[model, "Best Test Accuracy"]
                epochs_saved = max(0, epochs_total - stop_epoch - 1)
                time_per_epoch = runtime_info.loc[model, "Time Per Epoch (sec)"]

                rows.append({
                    "Model": model,
                    "Stopped_Val_Accuracy": val_acc,
                    "Original_Val_Accuracy": orig_acc,
                    "Epochs_Saved": epochs_saved,
                    "Model RAM (KB)": runtime_info.loc[model, "Model RAM (KB)"],
                    "Estimated Flash Memory (KB)": runtime_info.loc[model, "Estimated Flash Memory (KB)"],
                    "Time Per Epoch (sec)": time_per_epoch
                })

            except Exception as e:
                print(f"\u26a0\ufe0f Error processing {model}: {e}")

        df_simulated = pd.DataFrame(rows)
        if df_simulated.empty:
            continue

        error_rate, mse = evaluate_tournament(df_simulated, "Stopped_Val_Accuracy")

        time_saved_total = 0
        total_full_time = 0
        total_saved_epochs = 0

        for row in rows:
            model = row["Model"]
            epochs_saved = row["Epochs_Saved"]
            full_training_time_sec = runtime_info.loc[model, "Training Time (min)"] * 60
            time_per_epoch = full_training_time_sec / epochs_total
            time_saved_model = epochs_saved * time_per_epoch
            time_saved_total += time_saved_model
            total_saved_epochs += epochs_saved
            total_full_time += full_training_time_sec

        print_training_time(total_full_time / 60)

        percent_saved = (time_saved_total / total_full_time) * 100 if total_full_time > 0 else 0

        all_results.append({
            "Cutoff_Epoch": cutoff,
            "Val_Threshold": threshold,
            "Error_Rate (%)": round(100 * error_rate, 2),
            "Avg Fitness RMSE": f"{np.sqrt(mse):.2e}",
            "MSE": f"{mse:.2e}",
            "Saved Epochs": total_saved_epochs,
            "Total_Time_Saved_min": round(time_saved_total / 60, 2),
            "Time_Saved (%)": round(percent_saved, 2)
        })

# --- Output Results ---
df_summary = pd.DataFrame(all_results).sort_values(by="Time_Saved (%)")
df_filtered = df_summary[df_summary["Error_Rate (%)"] <= max_error_allowed]

if not df_filtered.empty:
    best_time_saved_row = df_filtered.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True]).iloc[0]
    print(f"✅ Best Config (≤ {max_error_allowed:.0f}% Error):")
    print(best_time_saved_row.to_string())
else:
    print(f"⚠️ No configurations found with error rate ≤ {max_error_allowed:.0f}%.")

if not df_filtered.empty:
    top_5_configs = df_filtered.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True]).head(5)
    print(f"\n ✅ Top 5 Configurations with ≤ {max_error_allowed:.0f}% Error Rate, Sorted by Time Saved:")
    print(top_5_configs.to_string(index=False))
else:
    print(f"⚠️ No configurations found with error rate ≤ {max_error_allowed:.0f}%.")
