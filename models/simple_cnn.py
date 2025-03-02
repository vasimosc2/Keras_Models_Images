from tensorflow.keras import layers, models # type: ignore


"""
A Simple CNN model:
We use as input Layer 32x32 pixels for the image and 3 colors (RGB)
Conv2D, this used applies learnable filters (kernels) to extract spatial features from the image.

MaxPolling2D, this layer is used  : Pooling layers reduce the spatial dimensions (downsampling).
                                    Max pooling takes the maximum value in a given window (e.g., 2×2).
                                    Helps to:
                                    Reduce computation (fewer pixels).
                                    Prevent overfitting (removes unnecessary details).
                                    Increase spatial invariance (small changes in position don't affect the output).

Conv2D, same as before but with higher filter size, Deeper layers capture more complex features like shapes, textures, and object parts.


Flatten, this layer transforms the 2D array into 1D array  This is needed because Dense layers expect a 1D input. No learnable parameters, just reshaping.

Dense(64, activation='relu'),  A fully connected (FC) layer connects every neuron to the next layer. Each neuron applies a weighted sum followed by an activation function:

Dense(10, activation='softmax'), 10 neurons, one for each class (if classifying CIFAR-10).  Each neuron connects to all 64 neurons from the previous layer.

"""
def create_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(100, activation='softmax')
    ])
    return model