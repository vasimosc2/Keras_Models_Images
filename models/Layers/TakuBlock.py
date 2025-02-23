from tensorflow.keras import layers # type: ignore
class TakuBlock(layers.Layer):
    def __init__(self, resolution: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
        super(TakuBlock, self).__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if in_channels != out_channels:
            self.skip_conn = layers.Conv2D(out_channels, kernel_size=1, strides=stride, padding="same")
        else:
            self.skip_conn = lambda x: x  # Identity function

        self.bn = layers.BatchNormalization()
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding="same", dilation_rate=dilation)
        self.act = layers.ReLU(max_value=6)

    def call(self, inputs, training=False):
        skip = self.skip_conn(inputs)
        x = self.depthwise_conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x + skip