import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Folder where the history CSV files are stored
folder = '../Manual_Run/results'  # Adjust this if needed

# Find all relevant CSVs
csv_files = sorted(glob.glob(os.path.join(folder, 'TakuNet_Random_*_history.csv')))

plt.figure(figsize=(10, 6))

# Process and plot each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    if 'val_accuracy' not in df.columns:
        print(f"Skipping {file}: 'val_accuracy' not found.")
        continue

    val_acc = df['val_accuracy']
    normalized_val_acc = val_acc / val_acc.max()

    label = os.path.basename(file).replace('_history.csv', '')
    plt.plot(normalized_val_acc, label=label)

# Plot formatting
plt.xlabel('Epoch')
plt.ylabel('Normalized Validation Accuracy')
plt.title('Normalized Val Accuracy Over Epochs')
plt.axvline(x=10, color='gray', linestyle='--', label='Epoch 10')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to file
output_path = os.path.join(folder, 'val_accuracy_comparison.png')
plt.savefig('val_accuracy_comparison.png')
print(f"Plot saved to: {output_path}")

