import random
from typing import Dict


def sample_from_search_space(model_search_space)->Dict:
    """Randomly selects model architecture hyperparameters."""
    return {
        "stem_block": {
            "filters": random.choice(model_search_space["stem_block"]["filters"]),
            "Conv_kernel": random.choice(model_search_space["stem_block"]["Conv_kernel"]),
            "l2_weight_decay": random.choice(model_search_space["stem_block"]["l2_weight_decay"]),
            "Conv_strides": random.choice(model_search_space["stem_block"]["Conv_strides"]),
            "dropout": random.choice(model_search_space["stem_block"]["dropout"]),
            "dilation_rate": random.choice(model_search_space["stem_block"]["dilation_rate"]),
            "DWConv_kernel": random.choice(model_search_space["stem_block"]["DWConv_kernel"]),
            "DWConv_strides": random.choice(model_search_space["stem_block"]["DWConv_strides"])
        },
        "stages_block": {
            "stages_number": random.choice(model_search_space["stages_block"]["stages_number"]),
            "taku_block": {
                "taku_block_number": random.choice(model_search_space["stages_block"]["taku_block"]["taku_block_number"]),
                "dropout": random.choice(model_search_space["stages_block"]["taku_block"]["dropout"]),
                "DWConv_kernel": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_kernel"]),
                "DWConv_strides": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_strides"])
            },
            "downsampler": {
                "dropout": random.choice(model_search_space["stages_block"]["downsampler"]["dropout"]),
                "pool_size": random.choice(model_search_space["stages_block"]["downsampler"]["pool_size"]),
                "Conv_kernel": random.choice(model_search_space["stages_block"]["downsampler"]["Conv_kernel"]),
                "l2_weight_decay": random.choice(model_search_space["stages_block"]["downsampler"]["l2_weight_decay"]),
                "strides": random.choice(model_search_space["stages_block"]["downsampler"]["strides"]),
            }
        },
        "refiner_block": {
            "DWConv_kernel": random.choice(model_search_space["refiner_block"]["DWConv_kernel"]),
            "l2_weight_decay": random.choice(model_search_space["refiner_block"]["l2_weight_decay"]),
            "DWConv_strides": random.choice(model_search_space["refiner_block"]["DWConv_strides"]),
            "dropout": random.choice(model_search_space["refiner_block"]["dropout"]),
            "num_output_classes": model_search_space["refiner_block"]["num_output_classes"]
        }
    }