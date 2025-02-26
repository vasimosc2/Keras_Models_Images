from typing import Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models, regularizers  # type: ignore

def taku_block(x, filters:int = 64, extra_layer_inside_taku:Union[layers.Layer, None] = None , l2_reg: Union[float, None] = None):
    res = x
    reg = regularizers.l2(l2_reg) if l2_reg else None  # Apply L2 if given
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    if extra_layer_inside_taku is not None:
        x = extra_layer_inside_taku(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    return x

def create_takunet_model(stages:int = 4, extra_layer_inside_taku:Union[layers.Layer, None] = None, extra_layer_outside_taku:Union[layers.Layer, None] = None, l2_reg: Union[float, None] = None):
    # Define model architecture
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Stem
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = extra_layer_outside_taku(x)
    for _ in range(stages):  # 4 Stages
        x = taku_block(x=x, filters=64, extra_layer_inside_taku=extra_layer_inside_taku, l2_reg=l2_reg)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # Downsampling
        x = layers.BatchNormalization()(x)

    # Refinement
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs, x)
    return model