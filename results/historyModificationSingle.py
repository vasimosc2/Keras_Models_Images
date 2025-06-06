import pandas as pd
import os
import glob

# Set paths
results: str = "results"
epochNumbers: int = 70
history_folder = os.path.join(results, f"{epochNumbers}-epochs")

history_files = glob.glob(os.path.join(history_folder, "*_history.csv"))
val_acc_path = os.path.join(history_folder, f"val_accuracy_comparison_{epochNumbers}.csv")
nas_results_path = os.path.join(history_folder,  f"Retraining_{epochNumbers}.csv")

combined_results = []

def simulate_performance_early_stopping(df, monitor='val_accuracy', patience=10, mode='max'):
    if mode == 'max':
        best_val = float('-inf')
        compare = lambda a, b: a > b
    else:
        best_val = float('inf')
        compare = lambda a, b: a < b

    best_epoch = 0
    patience_counter = 0

    for epoch in range(len(df)):
        current_val = df.loc[epoch, monitor]
        if compare(current_val, best_val):
            best_val = current_val
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            return best_epoch, "early_stopping"
    return len(df) - 1, "none"




def simulate_performance_stopping(df, monitor='val_accuracy', min_improvement=0.05):
    best_val_acc = 0.0
    wait = 0
    patience = int(0.2 * len(df))

    for epoch in range(len(df)):
        current_val = df.loc[epoch, monitor]
        if current_val > best_val_acc * (1 + min_improvement):
            best_val_acc = current_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return epoch, "performance_stopping"
    return len(df) - 1, "none"

# First pass to determine global best validation accuracy
global_best_accuracy = 0.0
for file_path in history_files:
    try:
        df = pd.read_csv(file_path)
        global_best_accuracy = max(global_best_accuracy, df["val_accuracy"].max())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Second pass to collect model performance and stopping metrics
for file_path in history_files:
    try:
        df = pd.read_csv(file_path)
        model_name = os.path.basename(file_path).replace("_history.csv", "")
        total_epochs = len(df)

        early_epoch, early_reason = simulate_performance_early_stopping(df, patience=8)
        perf_epoch, perf_reason = simulate_performance_stopping(df)

        if early_epoch <= perf_epoch:
            stop_epoch = early_epoch
            stop_reason = early_reason
        else:
            stop_epoch = perf_epoch
            stop_reason = perf_reason

        stopped_val_acc = df.loc[stop_epoch, 'val_accuracy']
        orig_val_acc = df['val_accuracy'].max()
        epochs_saved = total_epochs - stop_epoch - 1

        combined_results.append({
            'Model': model_name,
            'Stopped_Val_Accuracy': stopped_val_acc,
            'Original_Val_Accuracy': orig_val_acc,
            'Stop_Reason': stop_reason,
            'Epochs_Saved': epochs_saved
        })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save initial results to CSV
df_val_acc = pd.DataFrame(combined_results)

# Load NAS file and compute time per epoch (sec)
df_nas = pd.read_csv(nas_results_path)
df_nas["Time Per Epoch (sec)"] = (df_nas["Training Time (min)"] * 60) / 70

# Merge RAM, Flash, and Time per Epoch into results
df_merged = pd.merge(
    df_val_acc,
    df_nas[['Model', 'Model RAM (KB)', 'Estimated Flash Memory (KB)', 'Time Per Epoch (sec)']],
    on='Model',
    how='left'
)

# Save final file
df_merged.to_csv(val_acc_path, index=False)

# Compute total time saved per model in seconds
df_merged["Time_Saved (sec)"] = df_merged["Time Per Epoch (sec)"] * df_merged["Epochs_Saved"]

# Compute totals
total_time_saved_sec = df_merged["Time_Saved (sec)"].sum()
total_time_saved_min = total_time_saved_sec / 60
total_epochs_saved = df_merged["Epochs_Saved"].sum()

# Print summary
print(f"🕒 Total time saved across all models: {total_time_saved_sec:.1f} seconds (~{total_time_saved_min:.1f} minutes)")
print(f"📉 Total number of epochs saved across all models: {total_epochs_saved}")