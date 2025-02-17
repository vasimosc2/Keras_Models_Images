from tensorflow.keras import layers, models

def create_resnet_like_cnn():
    inputs = layers.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    shortcut = x  # Skip connection
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Add()([x, shortcut])  # Element-wise sum
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    shortcut = x
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model