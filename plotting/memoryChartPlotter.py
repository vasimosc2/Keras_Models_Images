import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print("Current working directory:", os.getcwd())

# Load CSV
df = pd.read_csv("utils/EstimationModels/ram_inference_flash.csv")  # Replace with your actual filename

# Optional: Rename columns for consistency
df.columns = ["StartingRam", "AccurateRam", "MeasuredRam","InferenceTime", "FlashMeasured(KB)"]

# Drop rows where MeasuredRam is NaN (to avoid plotting errors)
df = df.dropna(subset=["MeasuredRam"])

# --- Plot 1: Grouped Bar Chart ---
x = np.arange(len(df))  # Row indices
width = 0.25  # Width of each bar

fig, ax = plt.subplots(figsize=(14, 6))
bar1 = ax.bar(x - width, df["StartingRam"], width, label="Starting RAM")
bar2 = ax.bar(x, df["AccurateRam"], width, label="Accurate RAM")
bar3 = ax.bar(x + width, df["MeasuredRam"], width, label="Measured RAM")

ax.set_ylabel("RAM (KB)")
ax.set_xlabel("Model Index")
ax.set_title("Comparison of Starting, Accurate, and Measured RAM")
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Absolute Error Comparison ---
df["Error_Start"] = abs(df["StartingRam"] - df["MeasuredRam"])
df["Error_Accurate"] = abs(df["AccurateRam"] - df["MeasuredRam"])

fig, ax = plt.subplots(figsize=(14, 6))
bar1 = ax.bar(x - width/2, df["Error_Start"], width, label="|Starting - Measured|")
bar2 = ax.bar(x + width/2, df["Error_Accurate"], width, label="|Accurate - Measured|")

ax.set_ylabel("Absolute Error (KB)")
ax.set_xlabel("Model Index")
ax.set_title("Estimation Error Compared to Measured RAM")
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()
plt.tight_layout()
plt.show()
