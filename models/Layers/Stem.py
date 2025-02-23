import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

class Stem(layers.Layer):
    def __init__(self, resolution: int, in_channels: int, out_channels: int, reduction: int = 1):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        self.stride1 = 2 if reduction % 2 == 0 else 1
        self.stride2 = 2 if reduction % 4 == 0 else 1

        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=self.stride1, padding='same', dilation_rate=2)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(max_value=6)

        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=3, strides=self.stride2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU(max_value=6)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        return x

    def get_output_resolution(self):
        return (((self.resolution - 1) // self.stride1) + 1) // self.stride2 + 1
