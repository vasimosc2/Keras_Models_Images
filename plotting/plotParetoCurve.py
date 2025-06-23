import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.lines import Line2D

def plot_optimal_models(csvs, labels, colors, markers):
    plt.figure(figsize=(12, 6))

    for i, csv_path in enumerate(csvs):
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: x.strip())
        assert 'Best Test Accuracy' in df.columns
        assert 'Model RAM (KB)' in df.columns
        assert 'TFlite size(KB)' in df.columns

        plt.scatter(
            df['Model RAM (KB)'],
            df['Best Test Accuracy'],
            s=df['TFlite size(KB)'] * 1.5,
            alpha=0.7,
            label=labels[i],
            color=colors[i],
            marker=markers[i],
            edgecolors='black'
        )

    plt.xlabel('RAM Consumption (KB)', fontsize=12)
    plt.ylabel('Val. Accuracy', fontsize=12)
    plt.title('Optimal Models (Flexible Runs)', fontsize=14)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    os.makedirs("plotting", exist_ok=True)
    name = f"pareto_plot_{len(csvs)}runs.png"
    plt.savefig(os.path.join("plotting", name), dpi=300)
    print(f"✅ Plot saved to: plotting/{name}")
    plt.close()

def findParetoOptimalCsv(month, day, lr_strategy):
    date = f"{month}-{day}"
    return os.path.join("NAS", date, "Retraining", lr_strategy, "ParetoOptimals", "results", "ParetoOptimalFullTrain.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot up to 4 Pareto Optimal Runs")

    # 30 Epochs runs (green)
    parser.add_argument("--epoch_30_month_run1", type=str)
    parser.add_argument("--epoch_30_day30_run1", type=str)
    parser.add_argument("--epoch_30_month30_run2", type=str)
    parser.add_argument("--epoch_30_day30_run2", type=str)

    # 20 Epochs runs (blue)
    parser.add_argument("--epoch_20_month_run1", type=str)
    parser.add_argument("--epoch_20_day_run1", type=str)
    parser.add_argument("--epoch_20_month_run2", type=str)
    parser.add_argument("--epoch_20_day_run2", type=str)

    parser.add_argument("--lr_strategy", type=str, default="cosine")

    args = parser.parse_args()

    csvs, labels, colors, markers = [], [], [], []

    if args.epoch_30_month_run1 and args.epoch_30_day30_run1:
        csvs.append(findParetoOptimalCsv(args.epoch_30_month_run1, args.epoch_30_day30_run1, args.lr_strategy))
        labels.append("30 Epochs - Run 1 (●)")
        colors.append("green")
        markers.append("o")

    if args.epoch_30_month30_run2 and args.epoch_30_day30_run2:
        csvs.append(findParetoOptimalCsv(args.epoch_30_month30_run2, args.epoch_30_day30_run2, args.lr_strategy))
        labels.append("30 Epochs - Run 2 (▲)")
        colors.append("green")
        markers.append("^")

    if args.epoch_20_month_run1 and args.epoch_20_day_run1:
        csvs.append(findParetoOptimalCsv(args.epoch_20_month_run1, args.epoch_20_day_run1, args.lr_strategy))
        labels.append("20 Epochs - Run 1 (■)")
        colors.append("blue")
        markers.append("s")

    if args.epoch_20_month_run2 and args.epoch_20_day_run2:
        csvs.append(findParetoOptimalCsv(args.epoch_20_month_run2, args.epoch_20_day_run2, args.lr_strategy))
        labels.append("20 Epochs - Run 2 (◆)")
        colors.append("blue")
        markers.append("D")

    if not csvs:
        print("⚠️ No valid run data provided. Please provide at least one run using the expected --epoch_* arguments.")
        sys.exit(1)
    else:
        plot_optimal_models(csvs, labels, colors, markers)
