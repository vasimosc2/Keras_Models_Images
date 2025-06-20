import pandas as pd
import os
import random

"""
Script: evaluate_misranking_partial_vs_full.py

Description:
This script compares model rankings between two training durations (e.g., 30 and 70 epochs) to assess whether 
early-stopped models (trained partially) can correctly approximate the ranking order of fully trained models.

It performs tournament-style comparisons between models using two ranking strategies:
1. Accuracy-Based: Using the best test accuracy of partially vs. fully trained models.
2. Fitness-Based: Using a composite fitness score that combines accuracy, RAM usage, and flash memory usage.

The script:
- Loads two CSV files containing model metadata and performance for different training durations.
- Merges them by model name and runs randomized pairwise matchups.
- For each matchup, checks if the estimated winner (from partial training) matches the true winner (from full training).
- Records the number and rate of misranked predictions based on both accuracy and fitness.
- Optionally prints detailed logs of misranked pairs for debugging or reporting purposes.


Outputs:
- Total matchups and number of incorrect predictions
- Misranking error rate for both accuracy-based and fitness-based evaluations
- Optional detailed logs of misranked model comparisons

Config Flags:
- `PRINT_ACCURACY_MISTAKES`: Set to True to print incorrect accuracy-based matchups
- `PRINT_FITNESS_MISTAKES`: Set to True to print incorrect fitness-based matchups

Use Case:
This script is valuable for validating whether early training signals are reliable predictors of final model quality,
and helps determine if early stopping strategies can safely guide model selection.
"""


# === Config flags ===
PRINT_ACCURACY_MISTAKES = False
PRINT_FITNESS_MISTAKES = False

# Constants for normalization
MAX_RAM = 197.68
MAX_FLASH = 754.921875

def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 70 * acc + 20 * norm_ram + 10 * norm_flash

def mse_error(true1, true2):
    return (true1 - true2) ** 2

# Load both CSVs
results = "results"
fullTrainEpochs = 70
partiallyTrainEpochs = 10

history_fullTrainEpochs_folder = os.path.join(results, f"{fullTrainEpochs}-epochs")
full_train_path = os.path.join(history_fullTrainEpochs_folder, f"Retraining_{fullTrainEpochs}.csv")

history_partiallyTrainEpochs_folder = os.path.join(results, f"{partiallyTrainEpochs}-epochs")
partially_trained = os.path.join(history_partiallyTrainEpochs_folder, f"Retraining_{partiallyTrainEpochs}.csv")

original_df = pd.read_csv(full_train_path)
retrained_df = pd.read_csv(partially_trained)

# Merge by model name
merged = pd.merge(original_df, retrained_df, on="Model", suffixes=("_original", "_retrained"))
models = merged.to_dict("records")

# Tournament-style comparison
total_runs = 1000
accuracy_errors = 0
fitness_errors = 0
total_matches = 0

accuracy_mse_total = 0.0
fitness_mse_total = 0.0

accuracy_error_log = []
fitness_error_log = []

for _ in range(total_runs):
    shuffled = random.sample(models, len(models))
    for i in range(0, len(shuffled) - 1, 2):
        m1 = shuffled[i]
        m2 = shuffled[i + 1]

        # --- Accuracy-based comparison ---
        est_1 = m1["Best Test Accuracy_retrained"]
        est_2 = m2["Best Test Accuracy_retrained"]
        true_1 = m1["Best Test Accuracy_original"]
        true_2 = m2["Best Test Accuracy_original"]

        est_acc_winner = 1 if est_1 > est_2 else 2
        true_acc_winner = 1 if true_1 > true_2 else 2

        if est_acc_winner != true_acc_winner:
            accuracy_errors += 1
            accuracy_error_log.append(
                f"❌ ACC: Estimated {m1['Model']}({est_1:.4f}) vs {m2['Model']}({est_2:.4f}) "
                f"≠ True {true_1:.4f} vs {true_2:.4f}"
            )
        accuracy_mse_total += mse_error(true_1, true_2)

        # --- Fitness-based comparison ---
        f1_est = compute_fitness(est_1, m1["Model RAM (KB)_original"], m1["Estimated Flash Memory (KB)_original"])
        f2_est = compute_fitness(est_2, m2["Model RAM (KB)_original"], m2["Estimated Flash Memory (KB)_original"])

        f1_true = compute_fitness(true_1, m1["Model RAM (KB)_original"], m1["Estimated Flash Memory (KB)_original"])
        f2_true = compute_fitness(true_2, m2["Model RAM (KB)_original"], m2["Estimated Flash Memory (KB)_original"])

        est_fit_winner = 1 if f1_est > f2_est else 2
        true_fit_winner = 1 if f1_true > f2_true else 2

        if est_fit_winner != true_fit_winner:
            fitness_errors += 1
            fitness_error_log.append(
                f"❌ FIT: Estimated {m1['Model']} (acc={est_1:.4f}, ram={m1['Model RAM (KB)_original']:.2f}, "
                f"flash={m1['Estimated Flash Memory (KB)_original']:.2f}, fitness={f1_est:.4f}) vs "
                f"{m2['Model']} (acc={est_2:.4f}, ram={m2['Model RAM (KB)_original']:.2f}, "
                f"flash={m2['Estimated Flash Memory (KB)_original']:.2f}, fitness={f2_est:.4f}) \n"
                f"≠ True Fitness {m1['Model']} (acc={true_1:.4f}, ram={m1['Model RAM (KB)_original']:.2f}, "
                f"flash={m1['Estimated Flash Memory (KB)_original']:.2f}, fitness={f1_true:.4f}) vs "
                f"{m2['Model']} (acc={true_2:.4f}, ram={m2['Model RAM (KB)_original']:.2f}, "
                f"flash={m2['Estimated Flash Memory (KB)_original']:.2f}, fitness={f2_true:.4f})"
            )

        fitness_mse_total += mse_error(f1_true, f2_true)

        total_matches += 1

if PRINT_ACCURACY_MISTAKES:
    print("\n🔍 Misranked Accuracy Pairs:")
    for line in accuracy_error_log:
        print(line)

# Final output
print(f"\n📊 Accuracy-Based Misranking Evaluation for {partiallyTrainEpochs}")
print(f"🔁 Total runs: {total_runs}")
print(f"🎯 Total model matchups: {total_matches}")
print(f"❌ Misranked pairs (Accuracy only): {accuracy_errors}")
print(f"⚠️ Misranking rate (Accuracy): {100 * accuracy_errors / total_matches:.2f}%")
print(f"ℹ️ Note: Fitness values are normalized in the range [0, 1].")
print(f"📐 Average Accuracy MSE: {accuracy_mse_total / total_matches:.2e}")

if PRINT_FITNESS_MISTAKES:
    print("\n🔍 Misranked Fitness Pairs:")
    for line in fitness_error_log:
        print(line)

print(f"\nFor the Partial Training  of {partiallyTrainEpochs} :")
print(f"\n📊 Fitness-Based Misranking Evaluation (Accuracy + RAM + Flash) for {partiallyTrainEpochs}")
print(f"❌ Misranked pairs (Fitness): {fitness_errors}")
print(f"⚠️ Misranking rate (Fitness): {100 * fitness_errors / total_matches:.2f}%")
print(f"ℹ️ Note: Fitness values are normalized in the range [0, 1].")
print(f"📐 Average Fitness MSE: {fitness_mse_total / total_matches:.2e}")


