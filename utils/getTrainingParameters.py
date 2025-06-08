import random
from typing import Dict


def sample_from_train_and_evaluate(train_and_evaluate)->Dict:
    """Randomly selects training hyperparameters."""
    return {
        "label_smothing": train_and_evaluate["model_config"]["label_smothing"],

        "stop_patience": random.choice(train_and_evaluate["model_config"]["early_stop"]["stop_patience"]),

        "learning_rate": random.choice(train_and_evaluate["model_config"]["learning"]["rate"]),
        "learning_factor": train_and_evaluate["model_config"]["learning"]["factor"],
        "learning_rate_patience": train_and_evaluate["model_config"]["learning"]["patience"],
        
        "max_dropout": train_and_evaluate["model_config"]["adaptive_dropout"]["max_dropout"],
        "incrementFactor": train_and_evaluate["model_config"]["adaptive_dropout"]["incrementFactor"],
        "overfitting": train_and_evaluate["model_config"]["adaptive_dropout"]["overfitting"],
        
        "num_epochs": train_and_evaluate["evaluation_config"]["num_epochs"],
        "batch_size": train_and_evaluate["evaluation_config"]["batch_size"],

        "max_ram_consumption": train_and_evaluate["evaluation_config"]["max_ram_consumption"],
        "additional_ram_consumption": train_and_evaluate["evaluation_config"]["additional_ram_consumption"],

        "max_flash_consumption": train_and_evaluate["evaluation_config"]["max_flash_consumption"],
        "additional_flash_consumption": train_and_evaluate["evaluation_config"]["additional_flash_consumption"], 
        
        "data_dtype_multiplier": train_and_evaluate["evaluation_config"]["data_dtype_multiplier"],
        "model_dtype_multiplier": train_and_evaluate["evaluation_config"]["model_dtype_multiplier"],
    }