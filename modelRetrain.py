import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.backend.tensorflow.trainer")

# ==============================
# 🔁 Load model and reset weights
# ==============================
def load_architecture_and_reset_weights(h5_path):
    full_model = load_model(h5_path)
    json_config = full_model.to_json()
    model = model_from_json(json_config)  # Reset weights
    return model

# ==============================
# 🧪 Example: Use dummy data (replace with your dataset)
# ==============================
def load_dummy_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    return x_train, y_train, x_test, y_test

# ==============================
# 🚀 Retrain from scratch
# ==============================
def retrain_model(model, x_train, y_train, x_test, y_test, save_path=None):
    warmup_epochs = 5
    initial_lr = 0.05
    total_epochs = 50
    batchSize = 16
    def cosine_annealing_with_warmup(epoch)->float:
        if epoch < warmup_epochs:
            return float(initial_lr * (epoch + 1) / warmup_epochs)
        else:
            cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return float(initial_lr * cosine_decay)
        
    model.compile(optimizer=SGD(learning_rate=initial_lr, momentum=0.9),
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
    ]

    if save_path:
        callbacks.append(
            ModelCheckpoint(filepath=save_path, save_best_only=True, monitor='val_accuracy', mode='max')
        )

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=batchSize,
        epochs=total_epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"✅ Final test accuracy: {acc:.4f}")
    return model

# ==============================
# 🔧 MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The name of model to retrain")
    parser.add_argument("--name", type=str, default="TakuNet_Init_2.keras", help="The name of the model")
    parser.add_argument("--folder", type=str, default="NAS", help="The folder of the model")
    args = parser.parse_args()
    number_of_models = args.num_models

    h5_path = f"/zhome/02/e/181021/Desktop/Keras_Models_Images/{args.folder}/saved_models/{args.name}"
    #save_path = "retrained_model.keras"         # 👈 Optional: Save final model

    x_train, y_train, x_test, y_test = load_dummy_cifar100()  # 👈 Replace if needed

    model = load_architecture_and_reset_weights(h5_path)
    model.summary()

    retrained_model = retrain_model(model, x_train, y_train, x_test, y_test, save_path=None)
