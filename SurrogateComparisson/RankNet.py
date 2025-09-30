import tensorflow as tf
from tensorflow.keras import layers, Model


def build_ranknet(input_dim: int) -> Model:
    """
    Builds a simple RankNet model.

    RankNet takes two architecture embeddings,
    predicts which architecture is better.
    
    Input: Two vectors of shape (input_dim,)
    Output: Probability (0 to 1)
    """

    # Inputs
    input_a = layers.Input(shape=(input_dim,), name="input_a")
    input_b = layers.Input(shape=(input_dim,), name="input_b")

    # Shared MLP (same for both inputs)
    shared_mlp = tf.keras.Sequential([ # Dense = Fully connected layer
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu')
    ], name="shared_mlp")

    # Process both embeddings
    processed_a = shared_mlp(input_a)
    processed_b = shared_mlp(input_b)

    # Subtract embeddings
    diff = layers.Subtract()([processed_a, processed_b])

    # Pass through final layer to get probability
    out = layers.Dense(1, activation='sigmoid', name="rank_output")(diff)

    # Build model
    model = Model(inputs=[input_a, input_b], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model