import pandas as pd
import os
import glob


"""
Description:
This script evaluates the impact of early stopping for models trained for fewer epochs (e.g., 20 or 30).
It simulates two stopping strategies (early and performance-based) and compares the resulting accuracy
to the final accuracy achieved by the same models trained for 70 epochs. The goal is to estimate how
much training time could be saved with minimal loss in validation performance.

Output: A CSV summary with per-model early stopping accuracy, original accuracy, and time saved.
"""


# Set paths
results: str = "results"

epochNumbers: int = 30
epochTruth:int = 70

history_folder = os.path.join(results, f"{epochNumbers}-epochs")
history_files = glob.glob(os.path.join(history_folder, "*_history.csv"))
val_acc_path = os.path.join(history_folder, f"val_accuracy_comparison_{epochNumbers}.csv")
nas_results_path = os.path.join(history_folder, f"Retraining_{epochNumbers}.csv")

# Load 70-epoch reference results
nas_70_path = os.path.join(results, f"{str(epochTruth)}-epochs")
df_70 = pd.read_csv(os.path.join(nas_70_path, f"Retraining_{str(epochTruth)}.csv"))
df_70 = df_70[["Model", "Best Test Accuracy"]].rename(columns={"Best Test Accuracy": "Original_Val_Accuracy"})

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

# Process all history files (30-epoch training)
for file_path in history_files:
    try:
        df = pd.read_csv(file_path)
        model_name = os.path.basename(file_path).replace("_history.csv", "")
        total_epochs = len(df)

        early_epoch, early_reason = simulate_performance_early_stopping(df, patience=8)
        perf_epoch, perf_reason = simulate_performance_stopping(df)

        stop_epoch = early_epoch if early_epoch <= perf_epoch else perf_epoch

        combined_results.append({
            'Model': model_name,
            f'Val_Accuracy_{epochNumbers}_Epochs': df['val_accuracy'].max(),
            'Stopped_Val_Accuracy': df.loc[stop_epoch, 'val_accuracy'],
            'Stop_Reason': "early_stopping" if early_epoch <= perf_epoch else "performance_stopping",
            'Epochs_Saved': total_epochs - stop_epoch - 1
        })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Create DataFrame and merge with 70-epoch accuracy
df_val_acc = pd.DataFrame(combined_results)
df_val_acc = pd.merge(df_val_acc, df_70, on="Model", how="left")

# Fill missing 70-epoch accuracy with 30-epoch best if not found
df_val_acc["Original_Val_Accuracy"] = df_val_acc["Original_Val_Accuracy"].fillna(df_val_acc[f'Val_Accuracy_{epochNumbers}_Epochs'])

# Reorder columns to have the 3 accuracies side-by-side
df_val_acc = df_val_acc[[
    'Model',
    'Original_Val_Accuracy',
    f'Val_Accuracy_{epochNumbers}_Epochs',
    'Stopped_Val_Accuracy',
    'Stop_Reason',
    'Epochs_Saved'
]]

# Load NAS info and add time/flash/ram stats
df_nas = pd.read_csv(nas_results_path)

"""

This Represents the time per epoch in the Retrained version

"""
df_nas["Time Per Epoch (sec)"] = (df_nas["Training Time (min)"] * 60) / epochNumbers


df_merged = pd.merge(
    df_val_acc,
    df_nas[['Model', 'Model RAM (KB)', 'Estimated Flash Memory (KB)', 'Time Per Epoch (sec)']],
    on='Model',
    how='left'
)

# Save result
df_merged.to_csv(val_acc_path, index=False)

# Time saved summary
df_merged["Time_Saved (sec)"] = df_merged["Time Per Epoch (sec)"] * df_merged["Epochs_Saved"]
total_time_saved_sec = df_merged["Time_Saved (sec)"].sum()
total_epochs_saved = df_merged["Epochs_Saved"].sum()

print(f"🕒 Total time saved: {total_time_saved_sec:.1f} sec (~{total_time_saved_sec / 60:.1f} min)")
print(f"📉 Total epochs saved: {total_epochs_saved}")
