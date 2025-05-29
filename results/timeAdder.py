import pandas as pd
import os

# Path to your retraining results
file_retrain10:str = "Retraining_10.csv"
file_originalRun:str = "Best_Models_Results_NAS.csv"
retraining_csv = os.path.join("results", file_originalRun)

# Load the CSV
df = pd.read_csv(retraining_csv)

# Sum the training time column
total_minutes = df["Training Time (min)"].sum()
hours = int(total_minutes // 60)
minutes = int(total_minutes % 60)

# Print results
print("⏱️ Total Training Time (Retrained Models):")
print(f"🕒 {total_minutes:.2f} minutes")
print(f"🕓 ≈ {hours}h {minutes}m")
