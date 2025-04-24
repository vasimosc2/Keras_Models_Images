# Re-run after kernel reset
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# === CONFIG ===
CSV_PATH = "../Manual_Run/results/Old/Training_Results.csv"
ESTIMATE_COL = "Estimated Flash Memory (KB)"
TFLITE_COL = "TFlite Estimation size(KB)"
MODEL_SAVE_PATH = "flash_regression_model_poly.pkl"
PLOT = False

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)

# === SPLIT DATA ===
threshold = 400
df_small = df[df[ESTIMATE_COL] <= threshold]
df_large = df[df[ESTIMATE_COL] > threshold]

# === POLYNOMIAL REGRESSION FOR SMALL MODELS ===
X_small = df_small[ESTIMATE_COL].values.reshape(-1, 1)
y_small = df_small[TFLITE_COL].values
model_small = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model_small.fit(X_small, y_small)

# === LINEAR REGRESSION FOR LARGE MODELS ===
X_large = df_large[ESTIMATE_COL].values.reshape(-1, 1)
y_large = df_large[TFLITE_COL].values
model_large = LinearRegression()
model_large.fit(X_large, y_large)

# === SAVE MODELS ===
joblib.dump((model_small, model_large, threshold), MODEL_SAVE_PATH)

# === OPTIONAL: PLOT ===
if PLOT:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[ESTIMATE_COL], df[TFLITE_COL], color='blue', label="Actual TFLite Sizes")

    # Plot predictions
    x_vals = df[ESTIMATE_COL].values.reshape(-1, 1)
    predictions = [model_small.predict([[x[0]]])[0] if x[0] <= threshold else model_large.predict([[x[0]]])[0] for x in x_vals]
    plt.plot(df[ESTIMATE_COL], predictions, color='red', label="Piecewise Fit")

    plt.xlabel("Estimated Flash Memory (KB)")
    plt.ylabel("True TFLite Size (KB)")
    plt.title("Flash Estimation: Piecewise (Poly + Linear)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("flash_piecewise_model_plot.png")

print("✅ Saved improved piecewise model as flash_regression_model_poly.pkl and plot as flash_piecewise_model_plot.png")
