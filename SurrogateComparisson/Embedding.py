from typing import Dict
import numpy as np

def simple_architecture_embedding(model_params:Dict) -> np.ndarray:
    """
    Given a TakuNetModel instance, extract a fixed-size vector
    summarizing its architecture.
    (Mock Graph2Vec-like embedding)

    Parameters used:
    - Number of stages
    - Number of blocks
    - Kernel sizes
    - Strides

    Output: np.ndarray of shape (13,)
    """
    params = model_params
    features = []

    # Stem block features
    features.append(params['stem_block']['filters'])
    features.append(params['stem_block']['Conv_kernel'])
    features.append(params['stem_block']['Conv_strides'])
    features.append(params['stem_block']['DWConv_kernel'])
    features.append(params['stem_block']['DWConv_strides'])

    # Stages block features
    features.append(params['stages_block']['stages_number'])
    features.append(params['stages_block']['taku_block']['taku_block_number'])
    features.append(params['stages_block']['taku_block']['DWConv_kernel'])
    features.append(params['stages_block']['taku_block']['DWConv_strides'])
    # Downsampler features
    features.append(params['stages_block']['downsampler']['pool_size'])
    features.append(params['stages_block']['downsampler']['strides'])

    # Refiner block features
    features.append(params['refiner_block']['DWConv_kernel'])
    features.append(params['refiner_block']['DWConv_strides'])

    #Optimaizer One Hot encoding
    optimizer = params['optimizer'].lower()

    # Define available optimizers (order matters)
    available_optimizers = ["sgd", "adamw"]

    # Create one-hot encoding
    optimizer_onehot = [1.0 if optimizer == opt else 0.0 for opt in available_optimizers]

    # Append to feature vector
    features.extend(optimizer_onehot)


    features = np.array(features, dtype=np.float32)
    features /= np.max(features) + 1e-8 # Neural networks perform best when their inputs are normalized to similar scales.

    return features  # Shape (15,)