import os
import glob
import argparse
from typing import Optional
import pandas as pd
import tensorflow as tf

from utils.str2bool import str2bool

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        if any(tf.config.experimental.get_memory_growth(gpu) for gpu in gpus):
            print("⚠️ GPU is already initialized! `set_memory_growth()` will fail.")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️ No GPU found, running on CPU.")





from .createAndTrain import train_from_saved_config
from TakuNet import TrainingResults  # Assumes this contains .results after training

def main(folder, month, day, epochs, dropout, train, learning_strategy):
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
                                                     day=day,
                                                     learning_rate_strategy=learning_strategy)
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
    results_folder = os.path.join(folder, f"{month}-{day}", "Retraining", learning_strategy, f"{epochs}-epochs", "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_path = os.path.join(results_folder, f"Retraining_{epochs}.csv")

    df_results = pd.DataFrame(models_data)
    # ✅ Sort models alphabetically by "Model" column
    df_results = df_results.sort_values(by="Model")
    
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
          performaceStoppage = args.performaceStoppage,
          early_stopping_acc = args.early_stopping_acc,
          midway_callback = args.midway_callback,
          learning_strategy = args.lr )
