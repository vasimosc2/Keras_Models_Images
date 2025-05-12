import os
import json
import tensorflow as tf
from TakuNet import TakuNetModel
from data_processing import get_dataset
Folder="Manual_Run"
def load_config(model_name: str):
    with open(f"{Folder}/saved_configs/model_params/{model_name}_model_params.json" ,"r") as f:
        model_params = json.load(f)

    with open(f"{Folder}/saved_configs/train_params/{model_name}_train_params.json", "r") as f:
        train_params = json.load(f)

    return model_params, train_params


def train_from_saved_config(model_name: str):
    print(f"🔍 Loading saved configs for model: {model_name}")
    model_params, train_params = load_config(model_name)
    print("🧠 Creating new TakuNet model")
    model = TakuNetModel(
        model_name=model_name,
        input_shape=(32, 32, 3),
        model_params=model_params,
        train_params=train_params,
        x_train=None,
        y_train=None,
        x_test= None,
        y_test=None,
        folder=Folder
    )
    
    default_augementaion_technique ={ "apply_standard":False,
                                "apply_color":False,
                                "apply_geometric":False,
                                "apply_mixup": False,
                                "apply_cutmix": False}
    x_train, y_train, x_test, y_test = get_dataset( output_classes= model_params["refiner_block"]["num_output_classes"], 
                                                augementation_technique=default_augementaion_technique)

    print("🚀 Starting training")
    
    model.train(x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test)  # Train the model

    print("✅ Training completed!")
    model.summary()

if __name__ == "__main__":
    train_from_saved_config(model_name="TakuNet_Random_0")
