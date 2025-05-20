import argparse
import json
import os
import pandas as pd
from TakuNet import TakuNetModel
from compute_ram_show import compute_layer_ram_usage
from data_processing import get_dataset
from utils import memoryEstimator
Folder="Manual_Run/Retraining"
os.makedirs(f'{Folder}/results', exist_ok=True)

def load_config(model_name: str):
    with open(f"{Folder}/saved_configs/model_params/{model_name}_model_params.json" ,"r") as f:
        model_params = json.load(f)

    with open(f"{Folder}/saved_configs/train_params/{model_name}_train_params.json", "r") as f:
        train_params = json.load(f)

    return model_params, train_params


def train_from_saved_config(model_name: str, epochs:int, dropout:bool, train:bool):

    print(f"🔍 Loading saved configs for model: {model_name}\n")

    model_params, train_params = load_config(model_name)

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
        enable_dropout=dropout
    )
    
    default_augementaion_technique ={ "apply_standard":False,
                                "apply_color":False,
                                "apply_geometric":False,
                                "apply_mixup": False,
                                "apply_cutmix": False}
    x_train, y_train, x_test, y_test = get_dataset( output_classes= model_params["refiner_block"]["num_output_classes"], 
                                                augementation_technique=default_augementaion_technique)
    print(f"LAYERS MEMORY CONSUMPTION:\n")

    compute_layer_ram_usage(taku_model.model, data_dtype_multiplier=1)
    print(f"The estimated Max Ram is {memoryEstimator.memoryEstimation_peak_ram_only(model=taku_model.model,data_dtype_multiplier=1)}")
    if train:
        print("🚀 Starting training\n")
        
        taku_model.train(x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test)  # Train the model
        hist_df = pd.DataFrame(taku_model.results.history.history)
        hist_path = f'{Folder}/results/{taku_model.model_name}_history.csv'
        hist_df.to_csv(hist_path, index=False)
        print(f"📊 Training history saved to: {hist_path}\n")

        print("✅ Training completed!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model with optional SAM support")
    parser.add_argument("--name", type=str, default="TakuNet_Random_0", help="Model file name")
    parser.add_argument("--epochs", type=int, default=50, help="Model folder")
    parser.add_argument("--dropout", type=lambda x: x.lower() == "true", default=True, help="Enable dropout (True/False)")
    parser.add_argument("--train", type=lambda x: x.lower() == "true", default=True, help="Enable training (True/False)")
    args = parser.parse_args()

    train_from_saved_config(model_name=args.name,epochs=args.epochs, dropout = args.dropout, train = args.train)
