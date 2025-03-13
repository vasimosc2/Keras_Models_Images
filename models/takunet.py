from typing import Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models, regularizers  # type: ignore



def create_takunet_model(
    params:dict,
    extra_layer_outside_taku: Union[layers.Layer, None] = None,
    l2_reg: Union[float, None] = None
):
    """Build the Takunet model using modular blocks"""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Stem Block
    x = stem_block(inputs,params=params)

    if extra_layer_outside_taku is not None:
        x = extra_layer_outside_taku(x)

    # Stages with Taku Blocks
    for _ in range(params["stages"]):
        x = stage_block(x,params=params)
        # x = taku_block(x=x, params=params, l2_reg=l2_reg)
        # x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # Downsampling
        # x = layers.BatchNormalization()(x)

    # Refinement Block
    x = refinement_block(x,kernel_size=params["kernel_size_refinement_block"])

    # Classification Head
    x = classification_head(x,output_classes=params["num_output_classes"])

    # Create model
    model = models.Model(inputs, x)
    return model




# Blocks

def stem_block(inputs,params:dict):
    """Stem block of the Takunet model"""
    x = layers.Conv2D(params["filters_stem_1"], (params["kernel_size_stem_1"], params["kernel_size_stem_1"]), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(params["filters_stem_2"], (params["kernel_size_stem_2"], params["kernel_size_stem_2"]), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def refinement_block(x,kernel_size:int):
    """Refinement block using Depthwise Convolution"""
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def classification_head(x,output_classes:int):
    """Classification head for the Takunet model"""
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(output_classes, activation='softmax')(x)
    return x


def taku_block(inputs, params: dict, l2_reg: Union[float, None] = None):
    reg = regularizers.l2(l2_reg) if l2_reg else None
   # Conditionally pass kernel_regularizer
    if reg:
        x = layers.DepthwiseConv2D(kernel_size=params["kernel_size_taku_block_1"], padding='same', use_bias=False, kernel_regularizer=reg)(inputs)
    else:
        x = layers.DepthwiseConv2D(kernel_size=params["kernel_size_taku_block_1"], padding='same', use_bias=False)(inputs)
    
    x = layers.ReLU(6.0)(x)
    x = layers.BatchNormalization()(x)
    if params["dropout_rate"] > 0:
        x = layers.Dropout(params["dropout_rate"])(x)
    return x


def downsampler_block(inputs):
    x = layers.Conv2D(filters=inputs.shape[-1], kernel_size=1, use_bias=False)(inputs)
    x = layers.ReLU(6.0)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # Assuming GRN as Global Response Normalization
    return x

def stage_block(inputs, params):
    x = inputs
    for _ in range(params["stages"]):
        x = taku_block(x, params)
    
    concat = layers.Concatenate()([inputs, x])
    downsampled = downsampler_block(concat)
    return downsampled

# def taku_block(x, params:dict, l2_reg: Union[float, None] = None):
#     res = x
#     reg = regularizers.l2(l2_reg) if l2_reg else None  # Apply L2 if given
#     x = layers.Conv2D(params["filters_taku_block_1"], (params["kernel_size_taku_block_1"], params["kernel_size_taku_block_1"]), padding='same', activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     if params["dropout_rate"] > 0:
#         x = layers.Dropout(params["dropout_rate"])(x)
#     x = layers.Conv2D(params["filters_taku_block_2"], (params["kernel_size_taku_block_2"], params["kernel_size_taku_block_2"]), padding='same', activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     if res.shape[-1] != x.shape[-1]:  # Match channels if needed
#         res = layers.Conv2D(params["filters_taku_block_2"], (1, 1), padding="same")(res)
#     x = layers.Add()([x, res])
#     return x
