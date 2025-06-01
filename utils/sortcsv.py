import os
import pandas as pd
from pathlib import Path

# Get the path to this script
script_dir = Path(__file__).resolve().parent
epochNumber:int = 70
results = os.path.join(script_dir.parent,"results")
epochFolder = os.path.join(results,f"{str(epochNumber)}-epochs")
csv_path = os.path.join(epochFolder,f"Retraining_{str(epochNumber)}.csv")
# Load the CSV
df = pd.read_csv(csv_path)

# Sort alphabetically by the 'Model' column
df_sorted = df.sort_values(by="Model")

# Save it back in the same location
df_sorted.to_csv(csv_path, index=False)

print(f"CSV file sorted by 'Model' and saved to: {csv_path}")