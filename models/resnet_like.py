from tensorflow.keras import layers, models # type: ignore


"""
Main Difference than the previous ones:

Conv2D(64, (3,3)): This applies 64 different 3×3 filters to the input image.

Each filter slides over the image and detects different features, such as edges, textures, or patterns.
The output of this layer is called a feature map, and since we have 64 filters, we get 64 feature maps.

Shortcut = x: We save this output (the feature maps) to use later.

This is the skip connection in a ResNet block.
The idea is that instead of just passing information sequentially, we allow it to skip over layers, which helps improve training.


This is called a residual connection, and it does two important things:
Prevents information loss – If the new layers don’t learn useful features, the model can fall back on the original shortcut.
Improves gradient flow – During training, gradients can pass through the shortcut directly, avoiding the vanishing gradient problem.



Issues?
In very deep networks, the vanishing gradient problem occurs (gradients get too small during backpropagation, making training slow or unstable).
The model must learn a completely new transformation at each step (no option to keep useful features from earlier layers).


The first convolution extracts features.
The shortcut saves this information.
The second convolution further processes it.
Instead of overwriting previous features, we add the original features back (from the shortcut).

"""
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
    outputs = layers.Dense(100, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model