import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_optimal_models(csv1_path, csv2_path=None):
    dfs = [pd.read_csv(csv1_path)]
    labels = ['Model Set 1']
    colors = ['tab:blue']
    markers = ['o']

    if csv2_path:
        dfs.append(pd.read_csv(csv2_path))
        labels.append('Model Set 2')
        colors.append('tab:green')
        markers.append('s')

    plt.figure(figsize=(8, 6))

    for i, df in enumerate(dfs):
        df = df.rename(columns=lambda x: x.strip())
        assert 'TFlite Test Accuracy' in df.columns
        assert 'Model RAM (KB)' in df.columns
        assert 'TFlite size(KB)' in df.columns

        plt.scatter(
            df['Model RAM (KB)'],
            df['TFlite Test Accuracy'],
            s=df['TFlite size(KB)'],
            alpha=0.7,
            label=labels[i],
            color=colors[i],
            marker=markers[i],
            edgecolors='black'
        )

    plt.xlabel('RAM Consumption (KB)')
    plt.ylabel('TFLite Accuracy')
    plt.title('Optimal Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join("plotting","pareto_plot.png"))
    print(f"Plot saved to  pareto_plot.png")
    plt.close()

def findParetoOptimalCsv(month,day,learning_rate_strategy):
    date = f"{month}-{day}"
    Folder = os.path.join("NAS", date , "Retraining", learning_rate_strategy, "ParetoOptimals", "results", "ParetoOptimalFullTrain.csv" )
    return Folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pareto Optimal Models")
    parser.add_argument("--month1", type=str, default="Jun", help="Folder of first run (CSV must be in there)")
    parser.add_argument("--day1", type=str, default="09", help="Folder of second run (optional)")
    parser.add_argument("--learning_rate_strategy1", type=str, default="cosine", help="Folder of second run (optional)")

    parser.add_argument("--month2", type=str, default=None, help="Folder of first run (CSV must be in there)")
    parser.add_argument("--day2", type=str, default=None, help="Folder of second run (optional)")
    parser.add_argument("--learning_rate_strategy2", type=str, default=None, help="Folder of second run (optional)")

    args = parser.parse_args()

    pareto1 = findParetoOptimalCsv( month = args.month1, day = args.day1, learning_rate_strategy = args.learning_rate_strategy1 )
    if args.month2:
        pareto2 = findParetoOptimalCsv( month = args.month2, day = args.day2, learning_rate_strategy = args.learning_rate_strategy2 )
    else:
        pareto2 = None
        
    plot_optimal_models(pareto1, csv2_path=pareto2)
