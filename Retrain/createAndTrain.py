import argparse
import json
import os
from typing import Optional, Tuple
import pandas as pd
from TakuNet import TakuNetModel, TrainingResults
from compute_ram_show import compute_layer_ram_usage
from data_processing import get_dataset
from utils import memoryEstimator
import time
from utils.str2bool import str2bool


def load_config(model_name: str, folder:str):
    with open(f"{folder}/saved_configs/model_params/{model_name}_model_params.json" ,"r") as f:
        model_params = json.load(f)

    with open(f"{folder}/saved_configs/train_params/{model_name}_train_params.json", "r") as f:
        train_params = json.load(f)

    return model_params, train_params


def train_from_saved_config(model_name: str, epochs:int, dropout:bool, train:bool, folder:str, month:str, day:str, performaceStoppage:bool = False, early_stopping_acc:bool = False, midway_callback:bool = False, learning_rate_strategy:str = "cosine") -> Tuple[Optional[TrainingResults], Optional[str]] :
    
    date = f"{month}-{day}"
    
    Folder = os.path.join(folder, date)
    print(f"🔍 Loading saved configs for model: {model_name}\n")

    model_params, train_params = load_config(model_name=model_name,folder=Folder)
    Folder = os.path.join(Folder, "Retraining")

    Folder = os.path.join(Folder,learning_rate_strategy)


    Folder = os.path.join(Folder, f"{epochs}-epochs")
    
    os.makedirs(f'{Folder}/results', exist_ok=True)
    

    print("🧠 Creating new TakuNet model\n")
    taku_model = TakuNetModel(
        model_name=model_name,
        input_shape=(32, 32, 3),
        model_params=model_params,
        train_params=train_params,
        x_train=None,
        y_train=None,
        x_test= None,
        y_test=None,
        folder=Folder,
        epochs=epochs,
        enable_dropout=dropout,
        performaceStoppage=performaceStoppage,
        early_stopping_acc=early_stopping_acc,
        midway_callback=midway_callback,
        lr_schedule_strategy=learning_rate_strategy
    )
    
    default_augementaion_technique ={"apply_standard":False,
                                     "apply_color":False,
                                     "apply_geometric":False,
                                     "apply_mixup": False,
                                     "apply_cutmix": False}
    
    x_train, y_train, x_test, y_test = get_dataset(output_classes= model_params["refiner_block"]["num_output_classes"], 
                                                   augementation_technique=default_augementaion_technique)
    print(f"LAYERS MEMORY CONSUMPTION:\n")

    compute_layer_ram_usage(taku_model.model, data_dtype_multiplier=1)
    flash,ram = memoryEstimator.memoryEstimation(model=taku_model.model,data_dtype_multiplier=1)

    print(f"The estimated Max Ram is {ram} whereas Flash Memory is {flash}")

    if train:
        print("🚀 Starting training\n")
        
        taku_model.train(x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test)  # Train the model
        
        hist_df = pd.DataFrame(taku_model.results.history.history)
        os.makedirs(f'{Folder}/results/History', exist_ok=True)
        hist_path = f'{Folder}/results/History/{taku_model.model_name}_history.csv'
        hist_df.to_csv(hist_path, index=False)
        print(f"📊 Training history saved to: {hist_path}\n")

        print("✅ Training completed!\n")
        return (taku_model.results, model_params["optimizer"])
    else:
        print("❌ Training was skipped (train=False). Model initialized but not trained.\n")
        return (None, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model with optional SAM support")
    parser.add_argument("--name", type=str, default="TakuNet_Random_0", help="Model file name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to be run")
    parser.add_argument("--dropout", type=lambda x: x.lower() == "true", default=True, help="Enable dropout (True/False)")
    parser.add_argument("--train", type=lambda x: x.lower() == "true", default=True, help="Enable training (True/False)")
    parser.add_argument("--folder", type=str, default="NAS", help="The Run folder")
    parser.add_argument("--month", type=str, default="May", help="The Month a run was made")
    parser.add_argument("--day", type=str, default="24", help="The day a run was made")
    parser.add_argument("--performaceStoppage", type=str2bool, default=False, help="Enable/Disable the PerformanceStoppage during Retraining")
    parser.add_argument("--early_stopping_acc", type=str2bool, default=False, help="Enable/Disable the EarlyStoppingAcc during Retraining")
    parser.add_argument("--midway_callback", type=str2bool, default=False, help="Enable/Disable the MidwayCallback during Retraining")
    parser.add_argument("--lr", type=str, default="linear", help="The day a run was made")
    args = parser.parse_args()

    train_from_saved_config(model_name = args.name, 
                            epochs = args.epochs, 
                            dropout = args.dropout,
                            Folder=args.folder, 
                            train = args.train, 
                            month = args.month, 
                            day = args.day,
                            performaceStoppage = args.performaceStoppage,
                            early_stopping_acc = args.early_stopping_acc,
                            midway_callback = args.midway_callback,
                            learning_rate_strategy = args.lr)
