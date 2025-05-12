import os
import json
import tensorflow as tf
from TakuNet import TakuNetModel
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

    print("🚀 Starting training")
    model.train()

    print("✅ Training completed!")
    model.summary()

if __name__ == "__main__":
    train_from_saved_config(model_name="TakuNet_Init_0")
