import pandas as pd
import os
import random

# === Config flags ===
PRINT_ACCURACY_MISTAKES = False
PRINT_FITNESS_MISTAKES = True

# Constants for normalization
MAX_RAM = 197.68
MAX_FLASH = 754.921875

def compute_fitness(acc, ram, flash):
    norm_ram = max(0.0, 1.0 - ram / MAX_RAM)
    norm_flash = max(0.0, 1.0 - flash / MAX_FLASH)
    return 0.7 * acc + 0.2 * norm_ram + 0.1 * norm_flash

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
total_runs = 10
accuracy_errors = 0
fitness_errors = 0
total_matches = 0

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
                f"flash={m2['Estimated Flash Memory (KB)_original']:.2f}, fitness={f2_est:.4f}) "
                f"≠ True Fitness {f1_true:.4f} vs {f2_true:.4f}"
            )


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

if PRINT_FITNESS_MISTAKES:
    print("\n🔍 Misranked Fitness Pairs:")
    for line in fitness_error_log:
        print(line)


print(f"\n📊 Fitness-Based Misranking Evaluation (Accuracy + RAM + Flash) for {partiallyTrainEpochs}")
print(f"❌ Misranked pairs (Fitness): {fitness_errors}")
print(f"⚠️ Misranking rate (Fitness): {100 * fitness_errors / total_matches:.2f}%")

