import argparse
import pandas as pd
import os



"""
Script: retraining_vs_nas_times.py

Description:
This script compares the total training time between:
1. Models trained via Neural Architecture Search (NAS) for 70 epochs.
2. Retrained models using fewer epochs (e.g., 30 epochs), with an optional additional speed-up adjustment.

It loads two CSV files containing training times (in minutes) for both sets of models, then:
- Computes and prints the total training time for each set.
- Converts the time into hours and minutes for readability.
- Calculates the relative improvement (%) in training time between the original NAS process and the retrained version.
- Optionally incorporates an additional fixed speed-up to account for external efficiency gains (e.g., from hardware tuning or early pruning).

Configurable Parameters:
- `epochNumber`: Number of epochs used in the retrained set (default: "30")
- `additionalSpeedUp`: This is an ofset, which can be used as a percentage of speed-up in this learning Rate Optimization. If 0, then no other optimization took place. Please add the % Number

Use Case:
This script is helpful in quantifying the training efficiency gained through retraining models with fewer epochs 
or improved procedures. It provides a direct comparison in both absolute (minutes/hours) and relative (%) terms.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--partialEpochs", type=int, default=30, help="Partial Training Epochs (Strategy 1)")
parser.add_argument("--additionalSpeedUp", type=float, default=21.83 , help="Saved Time in the Partial Training from another optimization (Like Early Stoppage or Performance Stoppage)")

args = parser.parse_args()

additionalSpeedUp:float = args.additionalSpeedUp

# File paths
results = "results"

file_retrain = f"Retraining_{str(args.partialEpochs)}.csv"
file_originalRun = f"Retraining_70.csv"

folder_retrain:str = os.path.join(results,f"{str(args.partialEpochs)}-epochs")
folder_original:str = os.path.join(results,"70-epochs")

retraining_csv = os.path.join(folder_retrain, file_retrain)
original_csv = os.path.join(folder_original, file_originalRun)

# Load CSVs
df_retrain = pd.read_csv(retraining_csv)
df_original = pd.read_csv(original_csv)

# Helper function to compute total training time
def compute_training_time(df):
    return df["Training Time (min)"].sum()

# Helper function to print training time
def print_training_time(total_minutes, label):
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n⏱️ Total Training Time ({label}):")
    print(f"🕒 {total_minutes:.2f} minutes")
    print(f"🕓 ≈ {hours}h {minutes}m")

# Compute training times
time_retrain = compute_training_time(df_retrain) 
time_original = compute_training_time(df_original)

# Print results
print_training_time(time_retrain, "Retrained Models")
print_training_time(time_retrain-additionalSpeedUp, "Retrained Models")
print_training_time(time_original, "Original NAS Models")

# Calculate improvement
improvement = ( time_original - (time_retrain * ( 1 - ( additionalSpeedUp /100 ) ) ) ) / time_original * 100
print(f"\n📈 Training Time Improvement from NAS to Retraining:")
print(f"✅ {improvement:.2f}% faster")
