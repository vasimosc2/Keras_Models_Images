import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

def taku_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    return x

def create_takunet_model():
    # Define model architecture
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Stem
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Stages with Taku Blocks
    for _ in range(4):  # 4 Stages
        x = taku_block(x, 64)
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