
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore

class DownSampler(tf.keras.layers.Layer):
    def __init__(self, resolution, in_channels, hidden_channels, out_channels, kernel_size, stride, pooling=None, dense=False):
        super(DownSampler, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.dense = dense
        self.dense_channels = in_channels + hidden_channels
        self.downsampler_channels = out_channels if self.dense else hidden_channels

        self.dense_fc = None
        self.batch_norm = None
        self.activation = None

        if self.dense:
            self.dense_fc = layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, groups=self.dense_channels // 4)
            self.activation = layers.ReLU(max_value=6)
            self.batch_norm = layers.BatchNormalization()

        self.downsampler = None
        if pooling is None:
            self.downsampler = layers.Lambda(lambda x: x)  # Identity layer
        elif pooling == 'conv':
            self.downsampler = layers.Conv2D(filters=out_channels, kernel_size=2, strides=2, groups=out_channels if self.dense else 1, use_bias=False)
        elif pooling == 'maxpool':
            self.downsampler = tf.keras.Sequential([
                layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, use_bias=False) if not self.dense else layers.Lambda(lambda x: x),
                layers.MaxPooling2D(pool_size=kernel_size, strides=stride),
            ])
        elif pooling == 'avgpool':
            self.downsampler = tf.keras.Sequential([
                layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, use_bias=False) if not self.dense else layers.Lambda(lambda x: x),
                layers.AveragePooling2D(pool_size=kernel_size, strides=stride),
            ])

        self.grn = GRN(self.out_channels)  # Assuming GRN is another TensorFlow layer

    def get_output_resolution(self):
        if self.pooling in ['maxpool', 'avgpool']:
            output_resolution = (self.resolution - self.kernel_size) // self.stride + 1
        else:
            raise NotImplementedError(f"Pooling layer {self.pooling} not implemented")
        return output_resolution

    def call(self, x, dense_x=None):
        if self.dense and dense_x is not None:
            batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x = tf.concat([tf.reshape(x, [batch_size, h, w, -1]), tf.reshape(dense_x, [batch_size, h, w, -1])], axis=-1)
            x = self.activation(self.batch_norm(self.dense_fc(x)))

        x = self.grn(self.downsampler(x))
        return x






class GRN(tf.keras.layers.Layer):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = tf.Variable(tf.zeros([1, dim, 1, 1]), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, dim, 1, 1]), trainable=True)

    def call(self, x):
        Gx = tf.norm(x, ord='euclidean', axis=[2, 3], keepdims=True)
        Nx = Gx / (tf.reduce_mean(Gx, axis=1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x