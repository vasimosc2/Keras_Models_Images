import os
import glob
import argparse
from typing import Optional
import pandas as pd
from createAndTrain import train_from_saved_config
from TakuNet import TrainingResults  # Assumes this contains .results after training

def main(folder, month, day, epochs, dropout, train):
    config_path = os.path.join(folder, f"{month}-{day}", "saved_configs", "train_params")
    train_files = glob.glob(f"{config_path}/*_train_params.json")

    models_data = []

    for file in train_files:
        model_name = os.path.basename(file).replace("_train_params.json", "")
        print(f"\n🔁 Retraining model: {model_name}\n")

        try:
            trainingResult: Optional[TrainingResults]
            optimizer: Optional[str]
            trainingResult,optimizer = train_from_saved_config(model_name=model_name,
                                                     epochs=epochs,
                                                     dropout=dropout,
                                                     train=train,
                                                     folder=folder,
                                                     month=month,
                                                     day=day)
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

    # Save all results to a single CSV
    results_folder = os.path.join(folder, f"{month}-{day}", "Retraining", f"{epochs}-epochs", "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_path = os.path.join(results_folder, f"Retraining_{epochs}.csv")

    df_results = pd.DataFrame(models_data)
    df_results.to_csv(csv_path, index=False)

    print(f"\n✅ All retrained model results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain all models from saved configs and log results")
    parser.add_argument("--folder", type=str, default="NAS", help="The Run folder")
    parser.add_argument("--month", type=str, default="May", help="The Month a run was made")
    parser.add_argument("--day", type=str, default="27", help="The day a run was made")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to run")
    parser.add_argument("--dropout", type=lambda x: x.lower() == "true", default=True, help="Enable dropout (True/False)")
    parser.add_argument("--train", type=lambda x: x.lower() == "true", default=True, help="Enable training (True/False)")
    
    args = parser.parse_args()

    main(folder=args.folder,month=args.month,day=args.day,epochs=args.epochs,dropout=args.dropout,train=args.train)
