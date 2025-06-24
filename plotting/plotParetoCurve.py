import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.lines import Line2D

def plot_optimal_models(csvs, labels, colors, markers, marker_scale):
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
            s=df['TFlite size(KB)'] * marker_scale,
            alpha=0.7,
            label=labels[i],
            color=colors[i],
            marker=markers[i],
            edgecolors='black'
        )

    plt.xlabel('RAM Consumption (KB)', fontsize=12)
    plt.ylabel('Val. Accuracy', fontsize=12)
    plt.title('Optimal Models', fontsize=14)

    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='30 Epochs - Run 1',
               markerfacecolor='green', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='30 Epochs - Run 2',
               markerfacecolor='green', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='20 Epochs - Run 1',
               markerfacecolor='blue', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', label='20 Epochs - Run 2',
               markerfacecolor='blue', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='P', color='w', label='70 Epochs - Run 1',
               markerfacecolor='red', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='X', color='w', label='70 Epochs - Run 2',
               markerfacecolor='red', markersize=8, markeredgecolor='black'),
    ]

    plt.legend(handles=custom_legend, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
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

def originalParetoCsv(month,day):
    date = f"{month}-{day}"
    return os.path.join("NAS", date, "results", "Pareto_Optimal_Models.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot up to 6 Pareto Optimal Runs")

    # 30 Epochs runs (green)
    parser.add_argument("--epoch_30_month_run1", type=str)
    parser.add_argument("--epoch_30_day_run1", type=str)
    parser.add_argument("--epoch_30_month_run2", type=str)
    parser.add_argument("--epoch_30_day_run2", type=str)

    # 20 Epochs runs (blue)
    parser.add_argument("--epoch_20_month_run1", type=str)
    parser.add_argument("--epoch_20_day_run1", type=str)
    parser.add_argument("--epoch_20_month_run2", type=str)
    parser.add_argument("--epoch_20_day_run2", type=str)

    # 70 Epochs runs (red)
    parser.add_argument("--epoch_70_month_run1", type=str)
    parser.add_argument("--epoch_70_day_run1", type=str)
    parser.add_argument("--epoch_70_month_run2", type=str)
    parser.add_argument("--epoch_70_day_run2", type=str)

    # Optional marker size scaling
    parser.add_argument("--marker_scale", type=float, default=0.3, help="Scale factor for marker size")

    parser.add_argument("--lr_strategy", type=str, default="cosine")

    args = parser.parse_args()

    csvs, labels, colors, markers = [], [], [], []

    if args.epoch_30_month_run1 and args.epoch_30_day_run1:
        csvs.append(findParetoOptimalCsv(args.epoch_30_month_run1, args.epoch_30_day_run1, args.lr_strategy))
        labels.append("30 Epochs - Run 1")
        colors.append("green")
        markers.append("o")

    if args.epoch_30_month_run2 and args.epoch_30_day_run2:
        csvs.append(findParetoOptimalCsv(args.epoch_30_month_run2, args.epoch_30_day_run2, args.lr_strategy))
        labels.append("30 Epochs - Run 2")
        colors.append("green")
        markers.append("^")

    if args.epoch_20_month_run1 and args.epoch_20_day_run1:
        csvs.append(findParetoOptimalCsv(args.epoch_20_month_run1, args.epoch_20_day_run1, args.lr_strategy))
        labels.append("20 Epochs - Run 1")
        colors.append("blue")
        markers.append("s")

    if args.epoch_20_month_run2 and args.epoch_20_day_run2:
        csvs.append(findParetoOptimalCsv(args.epoch_20_month_run2, args.epoch_20_day_run2, args.lr_strategy))
        labels.append("20 Epochs - Run 2")
        colors.append("blue")
        markers.append("D")

    if args.epoch_70_month_run1 and args.epoch_70_day_run1:
        csvs.append(originalParetoCsv(args.epoch_70_month_run1, args.epoch_70_day_run1))
        labels.append("70 Epochs - Run 1")
        colors.append("red")
        markers.append("P")

    if args.epoch_70_month_run2 and args.epoch_70_day_run2:
        csvs.append(originalParetoCsv(args.epoch_70_month_run2, args.epoch_70_day_run2))
        labels.append("70 Epochs - Run 2")
        colors.append("red")
        markers.append("X")

    if not csvs:
        print("⚠️ No valid run data provided. Please provide at least one run using the expected --epoch_* arguments.")
        sys.exit(1)
    else:
        plot_optimal_models(csvs, labels, colors, markers, marker_scale=args.marker_scale)
