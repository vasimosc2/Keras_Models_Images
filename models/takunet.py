from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras import layers, Model, Input, regularizers # type: ignore
import tensorflow as tf # type: ignore

class GlobalResponseNormalization(Layer):
    """Custom Keras Layer for Global Response Normalization (GRN)"""
    def __init__(self, epsilon=1e-6, **kwargs):
        super(GlobalResponseNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[1, 2], keepdims=True)
        return inputs / tf.sqrt(variance + self.epsilon)

def stem_block(inputs:tuple, params: dict):
    """Stem Block: Initial feature extraction"""
    x = layers.Conv2D(filters = params["filters"], kernel_size = params["Conv_kernel"], strides=2, padding='same', dilation_rate= params["dilation_rate"], activation='relu6')(inputs)
    print(x.shape)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.DepthwiseConv2D(kernel_size = params["DWConv_kernel"], strides=2, padding='same', activation='relu6')(x)
    print(x.shape)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    #x = layers.Add()([x, inputs])  # Residual connection
    print(x.shape)
    return x

def taku_block(inputs:tuple, params: dict):
    """Taku Block: Depthwise Convolution with Residual Connection"""
    

    x = layers.DepthwiseConv2D(kernel_size = params["DWConv_kernel"], strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Add()([x, inputs])  # Residual connection

    return x

def global_response_normalization(x:tuple, epsilon=1e-6):
    """Global Response Normalization (GRN)"""
    mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
    return x / tf.sqrt(variance + epsilon)

def downsampler_block(inputs: tuple, params: dict, number_of_stages: int, curr_stage_number: int):
    """Downsampler Block: Reduces spatial dimensions and expands channels"""
    
    filters = inputs.shape[-1]  # Number of input channels
    
    # Ensure num_groups is a valid divisor of filters
    num_groups = max(1, min(number_of_stages, filters))  # Ensure it does not exceed filters
    
    # If num_groups is not a valid divisor, set it to 1
    if filters % num_groups != 0:
        num_groups = 1  
    
    kernel_size = params["Conv_kernel"]
    if inputs.shape[1] < kernel_size or inputs.shape[2] < kernel_size:  # If H or W < kernel size
        kernel_size = 1
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                      groups=num_groups, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    if curr_stage_number < number_of_stages:
        x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    else:
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalResponseNormalization()(x)

    return x


def stage_block(inputs:tuple, params:dict , curr_stage_number:int):
    """Stage Block: Multiple Taku Blocks followed by Downsampler"""
    x = inputs
    for _ in range(params["taku_block"]["taku_block_number"]):
        x = taku_block(inputs = x, params = params["taku_block"])

    concat = layers.Concatenate()([inputs, x])
    downsampled = downsampler_block(inputs=concat, params=params["downsampler"], number_of_stages = params["stages_number"], curr_stage_number = curr_stage_number )
    
    return downsampled

def refiner_block(inputs:tuple, params: dict ):
    """Refiner Block: Final feature aggregation and classification"""
   

    x = layers.DepthwiseConv2D(kernel_size = params["DWConv_kernel"], strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # AdaptiveAvgPool reducing spatial dimensions to 1x1
    x = layers.Dense(params["num_output_classes"], activation='softmax')(x)  # Output probabilities

    return x

def TakuNet(input_shape, params):
    """Builds the TakuNet model"""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Stem Block
    x = stem_block(inputs=inputs, params=params["stem_block"])

    for curr_stage_number in range(params["stages_block"]["stages_number"]):
        x = stage_block(inputs = x, params = params["stages_block"], curr_stage_number = curr_stage_number)

    outputs = refiner_block(x, params["refiner_block"])

    model = Model(inputs, outputs)
    model.summary()
    return model