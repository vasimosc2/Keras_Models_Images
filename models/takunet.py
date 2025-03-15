from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras import layers, Model # type: ignore
import tensorflow as tf # type: ignore

def stem_block(inputs:tuple, params: dict):
    """Stem Block: Initial feature extraction"""
    x = layers.Conv2D(filters=params["filters"], kernel_size=params["Conv_kernel"],
                strides=2, padding='same', activation=None, use_bias=False)(inputs)
    print(x.shape)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    if params["dropout"]>0:
        x = layers.Dropout(params["dropout"])(x)
    x = layers.DepthwiseConv2D(kernel_size=params["DWConv_kernel"], strides=2, padding='same', use_bias=False)(x)
    print(x.shape)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    print(x.shape)
    #x = layers.Add()([x, inputs])  # Residual connection
    return x

def taku_block(inputs:tuple, params: dict):
    """Taku Block: Depthwise Convolution with Residual Connection"""
    x = layers.DepthwiseConv2D(kernel_size = params["DWConv_kernel"], strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    if params["dropout"] > 0:
        x = layers.Dropout(params["dropout"])(x)
    x = layers.Add()([x, inputs])  # Residual connection

    return x

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
    if params["dropout"] > 0:
        x = layers.Dropout(params["dropout"])(x)

    if curr_stage_number < number_of_stages:
        x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    else:
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)

    x = layers.LayerNormalization()(x)

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
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)  # AdaptiveAvgPool reducing spatial dimensions to 1x1
    if params["dropout"] > 0:
        x = layers.Dropout(params["dropout"])(x)
    x = layers.Dense(params["num_output_classes"], activation='softmax')(x)  # Output probabilities

    return x

def TakuNet(input_shape, params):
    """Builds the TakuNet model"""
    inputs = tf.keras.Input(shape=(input_shape))

    # Stem Block
    x = stem_block(inputs=inputs, params=params["stem_block"])

    for curr_stage_number in range(params["stages_block"]["stages_number"]):
        x = stage_block(inputs = x, params = params["stages_block"], curr_stage_number = curr_stage_number)

    outputs = refiner_block(x, params["refiner_block"])

    model = Model(inputs, outputs)
    model.summary()
    return model
