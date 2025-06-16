import argparse
import glob
import os
import time
from typing import Optional

import pandas as pd

from Retrain.createAndTrain import train_from_saved_config
from TakuNet import TrainingResults
from utils.str2bool import str2bool


def main(folder, month, day, epochs, dropout, train, learning_strategy, performanceStoppage:bool, early_stopping_acc:bool, midway_callback:bool):

    config_path = os.path.join(folder, f"{month}-{day}", "saved_configs", "train_params")
    train_files = glob.glob(f"{config_path}/*_train_params.json")
    paretoFront = os.path.join(folder, f"{month}-{day}", "results", "Pareto_Optimal_Models.csv")
    if not os.path.exists(paretoFront):
        raise FileNotFoundError(f"❌ Pareto front file not found at: {paretoFront}")
    
    df_pareto = pd.read_csv(paretoFront)
    if "Model" not in df_pareto.columns:
        raise ValueError("❌ The file must contain a 'Model' column.")
    
    models_to_include = set(df_pareto["Model"].astype(str).str.strip())

    models_data = []


    for file in train_files:
        model_name = os.path.basename(file).replace("_train_params.json", "")

        # ✅ Only retrain if in Pareto front
        if model_name not in models_to_include:
            print(f"⏩ Skipping model {model_name} (not in Pareto front)")
            continue

        print(f"\n🔁 Retraining model: {model_name}\n")

        try:
            trainingResult: Optional[TrainingResults]
            optimizer: Optional[str]
            trainingResult, optimizer = train_from_saved_config(
                model_name=model_name,
                epochs=epochs,
                dropout=dropout,
                train=train,
                folder=folder,
                month=month,
                day=day,
                performaceStoppage=performanceStoppage,
                early_stopping_acc=early_stopping_acc,
                midway_callback=midway_callback,
                learning_rate_strategy=learning_strategy
            )
            if trainingResult is not None:
                models_data.append({
                    "Model": model_name,
                    "Best Train Accuracy": trainingResult.train_accuracy,
                    "Best Test Accuracy": trainingResult.test_accuracy,
                    "Swa Test Accuracy": trainingResult.SWA_test_accuracy,
                    "TFlite Test Accuracy": trainingResult.tflite_accuracy,
                    "Optimizer": optimizer,
                    "Precision": trainingResult.precision,
                    "Recall": trainingResult.recall,
                    "F1 Score": trainingResult.f1_score,
                    "Model RAM (KB)": trainingResult.ModelRam,
                    "Estimated Flash Memory (KB)": trainingResult.estimatedFlash,
                    "TFlite size(KB)": trainingResult.tflite_size,
                    "Flop Number": trainingResult.flops,
                    "Fitness Score": trainingResult.fitness_score,
                    "Training Time (min)": round(trainingResult.training_time / 60, 2),
                    "Epochs Trained": trainingResult.epochs_trained
                })

        except Exception as e:
            print(f"❌ Failed to retrain model {model_name}: {e}")

    # ✅ Save results
    results_folder = os.path.join(folder, f"{month}-{day}", "Retraining", learning_strategy, f"ParetoOptimals", "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_path = os.path.join(results_folder, f"ParetoOptimalFullTrain.csv")

    df_results = pd.DataFrame(models_data).sort_values(by="Model")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ All retrained model results saved to: {csv_path}")




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Retrain all models from saved configs and log results")
    parser.add_argument("--folder", type=str, default="NAS", help="The Run folder")
    parser.add_argument("--month", type=str, default="Jun", help="The Month a run was made")
    parser.add_argument("--day", type=str, default="09", help="The day a run was made")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs to run")
    parser.add_argument("--dropout", type=lambda x: x.lower() == "true", default=True, help="Enable dropout (True/False)")
    parser.add_argument("--train", type=lambda x: x.lower() == "true", default=True, help="Enable training (True/False)")
    parser.add_argument("--performaceStoppage", type=str2bool, default=False, help="Enable/Disable the PerformanceStoppage during Retraining")
    parser.add_argument("--early_stopping_acc", type=str2bool, default=False, help="Enable/Disable the EarlyStoppingAcc during Retraining")
    parser.add_argument("--midway_callback", type=str2bool, default=False, help="Enable/Disable the MidwayCallback during Retraining")
    parser.add_argument("--lr", type=str, default="cosine", help="The day a run was made")
    args = parser.parse_args()

    main(folder = args.folder, 
         month = args.month, 
         day = args.day, 
         epochs = args.epochs, 
         dropout = args.dropout, 
         train = args.train, 
         performanceStoppage = args.performaceStoppage, 
         early_stopping_acc = args.early_stopping_acc,
         midway_callback = args.midway_callback,
         learning_strategy = args.lr )