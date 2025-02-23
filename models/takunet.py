# takunet.py

import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

class TakuBlock(layers.Layer):
    # Define the TakuBlock as per your implementation or requirements
    def __init__(self, resolution, in_channels, hidden_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.conv = layers.Conv2D(hidden_channels, kernel_size, strides=stride, padding=padding)
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DownSampler(layers.Layer):
    # Define the DownSampler layer as per your implementation or requirements
    def __init__(self, resolution, in_channels, hidden_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, strides=stride, padding='same')

    def call(self, inputs):
        return self.conv(inputs)

def create_takunet(input_shape=(32, 32, 3), num_classes=10):
    """
    Create TakuNet model with given input shape and number of classes.
    
    Args:
    - input_shape: Shape of the input images.
    - num_classes: Number of output classes.
    
    Returns:
    - model: Compiled TakuNet model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Define the TakuNet architecture
    x = TakuBlock(resolution=input_shape[0], in_channels=input_shape[2], hidden_channels=32)(inputs)
    x = DownSampler(resolution=input_shape[0], in_channels=32, hidden_channels=64, out_channels=64)(x)
    x = TakuBlock(resolution=input_shape[0] // 2, in_channels=64, hidden_channels=128)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model
