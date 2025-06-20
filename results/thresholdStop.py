import argparse
import pandas as pd
import random
from pathlib import Path
import numpy as np



"""
thresholdStop.py

This script simulates early stopping strategies for evaluating candidate models during
Neural Architecture Search (NAS), aiming to reduce training time while maintaining accurate
model ranking based on validation accuracy and resource usage (RAM, Flash).

✨ Overview:
-----------
Two early stopping strategies are supported:
1. Strategy 1 (Grid Search): Finds the best (cutoff_epoch, accuracy_threshold) combination 
   that maximizes training time savings while keeping model misranking within an acceptable error rate.
2. Strategy 2 (Fixed Evaluation): Evaluates a user-defined cutoff point (as a fraction of total epochs) 
   and a minimum accuracy threshold to determine whether training should stop early.

📦 Features:
-----------
- Simulates early stopping on each model’s training history (CSV format).
- When using performance-aware runtimes (PerformanceStoppage), only evaluates history up to the actual stopping point.
- Computes "stopped" validation accuracy and estimates how many epochs and time could be saved.
- Measures model fitness using a weighted function of accuracy, RAM, and Flash memory.
- Evaluates fitness ranking consistency using a randomized tournament approach and computes error rate + RMSE.
- Automatically loads runtime data and model metrics (RAM, Flash, training time).
- Reports the best configuration (or warns if none found within error bounds).

⚙️ Parameters (via argparse):
-----------------------------
--strategy [1|2]             : 1 = Grid Search, 2 = Fixed Threshold Evaluation
--partialEpochs             : Number of training epochs per run (default: 30)
--max_error_allowed         : Maximum tolerable misranking error (percent)
--cutoff_fraction           : Fraction of total epochs to use as cutoff (Strategy 2)
--threshold                 : Accuracy threshold at the cutoff (Strategy 2)
--run_file                  : Prefix of the runtime file used to read validation results.
                              If set to "PerformanceStoppage", the script reads from the 
                              output of a previous performance stopping + learning rate 
                              optimization routine. In this case, for each model, only the
                              portion of the history up to the early stopping epoch is
                              considered. This simulates the real scenario where the model
                              was stopped early due to poor performance trends.

📈 Output:
---------
- Best configuration found (cutoff epoch + threshold) and performance metrics
- Error rate, RMSE, total/percentage training time saved
- Top-5 configurations (Strategy 1 only)

🔍 Example Usage:
-----------------
python thresholdStop.py --strategy 1 --partialEpochs 30 --max_error_allowed 20
python thresholdStop.py --strategy 2 --partialEpochs 30 --cutoff_fraction 0.25 --threshold 0.35 --run_file PerformanceStoppage
"""


parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=int, default=1, choices=[1, 2], help="Strategy: 1=Grid Search, 2=Fixed Evaluation")

parser.add_argument("--partialEpochs", type=int, default=70, help="Partial Training Epochs (Strategy 1)")
parser.add_argument("--max_error_allowed", type=float, default=20.0, help="Maximun Allowed Error Rate in Comparissons (Strategy 1)")

parser.add_argument("--cutoff", type=int, default=10, help="Fraction of total epochs to use as cutoff (Strategy 2)")
parser.add_argument("--threshold", type=float, default=0.35,help="Accuracy threshold at cutoff (Strategy 2)")

parser.add_argument("--run_file", type=str, default="Retraining",choices=["Retraining", "PerformanceStoppage"], help="Base name of the runtime file (default: Retraining)")

args = parser.parse_args()

# --- Settings ---
epochs_total = args.partialEpochs
ground_truth_epochs = 70
max_error_allowed = args.max_error_allowed # %

test_dir = Path("results") / f"{epochs_total}-epochs"
ground_dir = Path("results") / f"{ground_truth_epochs}-epochs"

history_files = list(test_dir.glob("*_history.csv"))
ground_truth_file = ground_dir / f"Retraining_{ground_truth_epochs}.csv"
runtime_file = test_dir / f"{args.run_file}_{epochs_total}.csv"

cutoff_epochs = list(range(3, 15, 1))
thresholds = np.round(np.arange(0.15, 0.45, 0.02), 3)

MAX_RAM = 197.68
MAX_FLASH = 754.921875
total_runs = 1000

# --- Load Ground Truth Accuracy and Runtime Info ---
ground_truth_accuracy = pd.read_csv(ground_truth_file).set_index("Model")
runtime_info = pd.read_csv(runtime_file).set_index("Model")

if "Time Per Epoch (sec)" not in runtime_info.columns:
    if "Training Time (min)" in runtime_info.columns:
        runtime_info["Time Per Epoch (sec)"] = (runtime_info["Training Time (min)"] * 60) / epochs_total
    else:
        raise ValueError("Missing both 'Time Per Epoch (sec)' and 'Training Time (min)' columns in runtime info CSV.")

# --- Fitness Function ---
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 70 * acc + 20 * norm_ram + 10 * norm_flash

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


def get_full_training_time_sec(model: str) -> float:
    if "Training Time (min)" in runtime_info.columns:
        return runtime_info.loc[model, "Training Time (min)"] * 60
    elif "Time Per Epoch (sec)" in runtime_info.columns:
        return runtime_info.loc[model, "Time Per Epoch (sec)"] * args.partialEpochs
    else:
        raise ValueError(f"Missing timing columns for model: {model}")



# --- Grid Search ---
if args.strategy ==1:
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
                    if args.run_file == "PerformanceStoppage" and model in runtime_info.index:
                        stop_epoch = runtime_info.loc[model, "Epochs_Saved"]
                        trim_index = epochs_total - stop_epoch
                        df = df.iloc[:trim_index]
                        # print(f"The history file of the {model}")
                        # print(df)
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
                full_training_time_sec = get_full_training_time_sec(model=model)
                time_per_epoch = full_training_time_sec / epochs_total
                time_saved_model = epochs_saved * time_per_epoch
                time_saved_total += time_saved_model
                total_saved_epochs += epochs_saved
                total_full_time += full_training_time_sec

            # This can be uncommented to see the Time of the full Partial Train. ( The total time of the 30-epochs training for example)
            #print_training_time(total_full_time / 60)

            percent_saved = (time_saved_total / total_full_time) * 100 if total_full_time > 0 else 0

            all_results.append({
                "Cutoff_Epoch": cutoff,
                "Val_Threshold": threshold,
                "Error_Rate (%)": round(100 * error_rate, 2),
                "Avg Fitness RMSE": f"{round(np.sqrt(mse),2)}",
                "MSE": f"{round(mse,2)}",
                "Saved Epochs": total_saved_epochs,
                "Training Initial Time (min)": total_full_time / 60,
                "Total_Time_Saved_min": round(time_saved_total / 60, 2),
                "Time_Saved (%)": round(percent_saved, 2)
            })

    # --- Output Results ---
    df_summary = pd.DataFrame(all_results).sort_values(by="Time_Saved (%)")
    df_filtered = df_summary[df_summary["Error_Rate (%)"] <= max_error_allowed]

    if not df_filtered.empty:
        best_time_saved_row = df_filtered.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True]).iloc[0]
        print(f"\n ✅ Best Config (≤ {max_error_allowed:.0f}% Error):")
        print(best_time_saved_row.to_string())
    else:
        print(f"⚠️ No configurations found with error rate ≤ {max_error_allowed:.0f}%.")

    if not df_filtered.empty:
        top_5_configs = df_filtered.sort_values(by=["Total_Time_Saved_min", "Error_Rate (%)"], ascending=[False, True]).head(5)
        print(f"\n ✅ Top 5 Configurations with ≤ {max_error_allowed:.0f}% Error Rate, Sorted by Time Saved:")
        print(top_5_configs.to_string(index=False))

if args.strategy ==2:
    if args.cutoff is None or args.threshold is None:
        print("❌ Strategy 2 requires both --cutoff and --threshold.")
        exit(1)

    cutoff = int(args.cutoff)
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
            stop_epoch = cutoff_index if df.loc[cutoff_index, "val_accuracy"] < args.threshold else len(df) - 1
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
            print(f"⚠️ Error processing {model}: {e}")

    df_simulated = pd.DataFrame(rows)
    if df_simulated.empty:
        print("❌ No configurations found with the given threshold and cutoff.")
    else:
        error_rate, mse = evaluate_tournament(df_simulated, "Stopped_Val_Accuracy")
        total_time_saved = (df_simulated["Epochs_Saved"] * df_simulated["Time Per Epoch (sec)"]).sum() / 60
        if "Training Time (min)" in runtime_info.columns:
            total_time_full = runtime_info.loc[df_simulated["Model"], "Training Time (min)"].sum()
        elif "Time Per Epoch (sec)" in runtime_info.columns:
            total_time_full = round((runtime_info.loc[df_simulated["Model"], "Time Per Epoch (sec)"] / (60/args.partialEpochs)).sum(),2)
            print(f"Total Time {total_time_full} minutes")
        else:
            total_time_full = 0  # Or raise an error depending on your needs
       
        percent_saved = (total_time_saved / total_time_full) * 100 if total_time_full > 0 else 0

        print(f"\n✅ Strategy 2 Evaluation:")
        print(f"🔢 Cutoff Epoch: {cutoff} / {epochs_total} ({cutoff / epochs_total:.1%})")
        print(f"🎯 Threshold: {args.threshold}")
        print(f"❗ Error Rate: {100 * error_rate:.2f}%")
        print(f"📉 RMSE: {np.sqrt(mse):.2e}")
        print(f"⏱️ Time Saved: {total_time_saved:.2f} min ({percent_saved:.2f}%)")