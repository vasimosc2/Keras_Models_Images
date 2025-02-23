import tensorflow as tf # type: ignore
from tensorflow.keras import layers, Model # type: ignore
import gc
import logging

# Assuming Stem, TakuBlock, and DownSampler classes are defined elsewhere
from models.Layers import Stem, TakuBlock
from models.Layers import DownSampler

class TakuNet(Model):
    """ TakuNet
    """
    def __init__(self, net_kwargs: dict, criterion: layers.Layer, optim_kwargs: dict):
        super().__init__() # In the given example it uses the criterion and optim, there
        logging.info(f"Building TakuNet with {net_kwargs}")

        self.input_channels = net_kwargs["input_channels"]  # e.g., 3
        self.output_classes = net_kwargs["output_classes"]  # e.g., 4
        self.depths = net_kwargs["depths"]  # e.g., [5, 5, 5, 4]
        self.widths = net_kwargs["widths"]  # e.g., [40, 80, 160, 240]
        self.heads = net_kwargs["heads"]  # e.g., [1, 1, 1, 1]
        self.poolings = net_kwargs["poolings"]  # e.g., [1, 1, 1, 1]
        self.dense = net_kwargs["dense"]  # e.g., True
        self.net_modules = net_kwargs["modules"]  # e.g., [TakuBlock, TakuBlock, TakuBlock, TakuBlock]
        self.reduction = net_kwargs["stem_reduction"]  # e.g., 4
        self.resolution = net_kwargs["resolution"]  # e.g., 224

        assert len(self.depths) == len(self.widths) == len(self.heads), "depths, heads, and widths must have the same length"

        self.stages = []
        self.stages.append(Stem(self.resolution, self.input_channels, self.widths[0], reduction=self.reduction))
        curr_resolution = self.stages[0].get_output_resolution()
        prev_channel_dim = self.widths[0]

        for i in range(len(self.depths)):
            depth = self.depths[i] if i < len(self.depths) - 1 else self.depths[i] - 1
            hidden_channels = self.widths[i]
            out_channels = self.widths[i + 1] if i < len(self.depths) - 1 else self.widths[i]

            self.stages.append(Stage(self.net_modules[i], curr_resolution, prev_channel_dim, hidden_channels, out_channels, depth, self.poolings[i], dense=self.dense))
            curr_resolution = self.stages[i + 1].downsampler.get_output_resolution()
            prev_channel_dim = out_channels  # Update previous channel dimension

        self.refiner = tf.keras.Sequential([
            layers.Conv2D(prev_channel_dim, kernel_size=3, strides=1, padding='same', groups=prev_channel_dim),
            layers.BatchNormalization(),
        ])

        self.classifier = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.output_classes)
        ])

        gc.collect()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.refiner(x)
        x = self.classifier(x)
        return x

class Stage(tf.keras.layers.Layer):
    def __init__(self, module, resolution: int, in_channels: int, hidden_channels: int, out_channels: int, depth: int, pooling=None, dense: bool=False) -> None:
        super(Stage, self).__init__()
        self.layers = []
        self.downsample = True  # Assuming downsample is true by default

        for i in range(depth):
            cur_in_channels = in_channels if i == 0 else hidden_channels
            self.layers.append(module(resolution, cur_in_channels, hidden_channels, kernel_size=3, stride=1, padding='same'))

        self.stage = tf.keras.Sequential(self.layers)
        self.downsampler = DownSampler(resolution, in_channels, hidden_channels, out_channels, kernel_size=2, stride=2, pooling=pooling, dense=dense)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = self.stage(x)
        if self.downsample:
            out = self.downsampler(out, dense_x=x)
        return out

def create_takunet(net_kwargs: dict, criterion: layers.Layer, optim_kwargs: dict, ckpt_path: str = None):
    net_kwargs['modules'] = [TakuBlock, TakuBlock, TakuBlock, TakuBlock]
    net_kwargs['depths'] = [5, 5, 5, 4]
    net_kwargs['widths'] = [40, 80, 160, 240]
    net_kwargs['heads'] = [1, 1, 1, 1]
    net_kwargs['poolings'] = [layers.MaxPooling2D, layers.MaxPooling2D, layers.MaxPooling2D, layers.AveragePooling2D]

    model = TakuNet(net_kwargs, criterion, optim_kwargs)
    if ckpt_path is not None:
        model.load_weights(ckpt_path)

    return model
