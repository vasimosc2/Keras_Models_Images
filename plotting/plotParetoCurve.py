import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D 

def plot_optimal_models(csv1_path, csv2_path=None):
    dfs = [pd.read_csv(csv1_path)]
    labels = ['30-epochs Run (● Circle)']
    colors = ['tab:blue']
    markers = ['o']

    name = "pareto_plot_single.png"
    if csv2_path:
        dfs.append(pd.read_csv(csv2_path))
        labels.append('20-epochs Run (■ Square)')
        colors.append('tab:green')
        markers.append('s')
        name = "pareto_plot_comparison.png"

    plt.figure(figsize=(12, 6))  # Wider for better horizontal spacing

    for i, df in enumerate(dfs):
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
    plt.title('Optimal Models', fontsize=14)

    # Custom legend with clear shape indications
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='30-epochs Run (● Circle)',
               markerfacecolor='tab:blue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='20-epochs Run (■ Square)',
               markerfacecolor='tab:green', markersize=10, markeredgecolor='black'),
    ]
    plt.legend(handles=custom_legend, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend

    os.makedirs("plotting", exist_ok=True)
    plt.savefig(os.path.join("plotting", name), dpi=300)
    print(f"✅ Plot saved to: plotting/{name}")
    plt.close()

def findParetoOptimalCsv(month, day, learning_rate_strategy):
    date = f"{month}-{day}"
    return os.path.join("NAS", date, "Retraining", learning_rate_strategy, "ParetoOptimals", "results", "ParetoOptimalFullTrain.csv")

if __name__ == "__main__":
    """
    Jun-09: Run for the 30 epochs
    Jun-08: Run for the 20 epochs
    """
    parser = argparse.ArgumentParser(description="Plot Pareto Optimal Models")
    parser.add_argument("--month1", type=str, default="Jun")
    parser.add_argument("--day1", type=str, default="09")
    parser.add_argument("--learning_rate_strategy1", type=str, default="cosine")

    parser.add_argument("--month2", type=str, default=None)
    parser.add_argument("--day2", type=str, default=None)
    parser.add_argument("--learning_rate_strategy2", type=str, default="cosine")

    args = parser.parse_args()

    pareto1 = findParetoOptimalCsv(args.month1, args.day1, args.learning_rate_strategy1)
    pareto2 = findParetoOptimalCsv(args.month2, args.day2, args.learning_rate_strategy2) if args.month2 else None

    plot_optimal_models(pareto1, csv2_path=pareto2)
