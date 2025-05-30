import pandas as pd
import os
import random

# Load both CSVs
history_folder = "results"
full_train_path = os.path.join(history_folder, "Best_Models_Results_NAS.csv")
partially_trained = os.path.join(history_folder, "Retraining_50.csv")

original_df = pd.read_csv(full_train_path)
retrained_df = pd.read_csv(partially_trained)

# Merge by model name
merged = pd.merge(original_df, retrained_df, on="Model", suffixes=("_original", "_retrained"))

# Prepare list of records
models = merged.to_dict("records")

# Tournament-style comparison
total_runs = 1000
total_errors = 0
total_matches = 0
error_log = []

for _ in range(total_runs):
    shuffled = random.sample(models, len(models))
    for i in range(0, len(shuffled) - 1, 2):
        m1 = shuffled[i]
        m2 = shuffled[i + 1]

        est_1 = m1["Best Test Accuracy_retrained"]
        est_2 = m2["Best Test Accuracy_retrained"]
        true_1 = m1["Best Test Accuracy_original"]
        true_2 = m2["Best Test Accuracy_original"]

        est_winner = 1 if est_1 > est_2 else 2
        true_winner = 1 if true_1 > true_2 else 2

        if est_winner != true_winner:
            total_errors += 1

            if est_winner == 1:
                error_log.append(
                    f"❌ Estimated model '{m1['Model']}' (acc={est_1:.4f}, true={true_1:.4f}) > "
                    f"'{m2['Model']}' (acc={est_2:.4f}, true={true_2:.4f}) — ❌ Misranking"
                )
            else:
                error_log.append(
                    f"❌ Estimated model '{m2['Model']}' (acc={est_2:.4f}, true={true_2:.4f}) > "
                    f"'{m1['Model']}' (acc={est_1:.4f}, true={true_1:.4f}) — ❌ Misranking"
                )

        total_matches += 1

#To show which are models are MisRanked uncomment the following

# for mistake in error_log:
#     print(mistake)

# Final output
print("\n📊 Accuracy-Based Misranking Evaluation")
print(f"🔁 Total runs: {total_runs}")
print(f"🎯 Total model matchups: {total_matches}")
print(f"❌ Misranked pairs: {total_errors}")
print(f"⚠️ Misranking rate: {100 * total_errors / total_matches:.2f}%\n")
