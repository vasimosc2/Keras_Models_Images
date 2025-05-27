import os
import random
import pandas as pd
import numpy as np

# Load the CSV
csvPath = os.path.join("results", "val_accuracy_comparison.csv")
df = pd.read_csv(csvPath)

# Constants
MAX_RAM = 197.68
MAX_FLASH = 754.921875
# Fitness function
# Constants
MAX_RAM = 197.68
MAX_FLASH = 754.921875

# Fitness function
def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

# Tournament logic
total_runs = 1000
total_errors = 0
total_matches = 0

for run in range(total_runs):
    shuffled = random.sample(list(df.to_dict('records')), len(df))  # Random order as list of dicts
    i = 0

    while i < len(shuffled) - 1:
        m1 = shuffled[i]
        m2 = shuffled[i + 1]

        # Compute real fitness
        f1_real = compute_fitness(m1["Original_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
        f2_real = compute_fitness(m2["Original_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

        # Compute estimated fitness
        f1_est = compute_fitness(m1["Stopped_Val_Accuracy"], m1["Model RAM (KB)"], m1["Estimated Flash Memory (KB)"])
        f2_est = compute_fitness(m2["Stopped_Val_Accuracy"], m2["Model RAM (KB)"], m2["Estimated Flash Memory (KB)"])

        real_winner = 1 if f1_real > f2_real else 2
        est_winner = 1 if f1_est > f2_est else 2

        if real_winner != est_winner:
            total_errors += 1

        total_matches += 1
        i += 2  # Move to next pair

# Final output
print(f"🔁 Total runs: {total_runs}")
print(f"🎯 Total matches: {total_matches}")
print(f"❌ Total mistakes using stopped accuracy: {total_errors}")
print(f"⚠️ Error rate: {100 * total_errors / total_matches:.2f}%")