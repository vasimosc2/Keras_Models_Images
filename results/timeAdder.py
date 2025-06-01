import pandas as pd
import os




#


additionalSpeedUp:str = 15

# File paths
results = "results"
epochNumber:str = "30"

file_retrain = f"Retraining_{epochNumber}.csv"
file_originalRun = f"Retraining_70.csv"

folder_retrain:str = os.path.join(results,f"{epochNumber}-epochs")
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
print_training_time(time_original, "Original NAS Models")

# Calculate improvement
improvement = (time_original - (time_retrain - additionalSpeedUp) ) / time_original * 100
print(f"\n📈 Training Time Improvement from NAS to Retraining:")
print(f"✅ {improvement:.2f}% faster")
