import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json, model_from_config
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.backend.tensorflow.trainer")

# ✅ Import SAMModel (you must have it in Models/SAM.py)
from Models.SAM import SAMModel

# 🔁 Load model and optionally extract original compile config
def load_model_with_original_config(h5_path, use_sam=False):
    model = load_model(h5_path, custom_objects={"SAMModel": SAMModel} if use_sam else {})
    json_config = model.to_json()
    model_reset = model_from_json(json_config, custom_objects={"SAMModel": SAMModel} if use_sam else {})
    compile_config = model.get_config().get("compile_config", None)
    return model_reset, compile_config

# Dummy CIFAR-100 loader (replace with your own dataset)
def load_dummy_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    return x_train, y_train, x_test, y_test

# 🚀 Retrain model
def retrain_model(model, x_train, y_train, x_test, y_test, use_sam=False, use_original_config=False, compile_config=None, save_path=None):
    warmup_epochs = 5
    initial_lr = 0.05
    total_epochs = 50
    batchSize = 16

    def cosine_annealing_with_warmup(epoch):
        if epoch < warmup_epochs:
            return float(initial_lr * (epoch + 1) / warmup_epochs)
        else:
            cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return float(initial_lr * cosine_decay)

    if use_sam:
        model = SAMModel(model)

    if use_original_config and compile_config is not None:
        print("🔁 Using original compile configuration")
        model.compile(**model_from_config({'class_name': 'Model', 'config': compile_config}).get_config()["compile_config"])
    else:
        print("⚙️ Using custom training configuration")
        model.compile(optimizer=SGD(learning_rate=initial_lr, momentum=0.9),
                      loss=CategoricalCrossentropy(label_smoothing=0.1),
                      metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
    ]

    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, save_best_only=True, monitor='val_accuracy', mode='max'))

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

# 🧠 Main logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model with optional SAM support")
    parser.add_argument("--name", type=str, default="TakuNet_Init_2.keras", help="Model file name")
    parser.add_argument("--folder", type=str, default="NAS", help="Model folder")
    parser.add_argument("--sam", type=str, default="false", help="Use SAMModel: true or false")
    args = parser.parse_args()

    use_sam = args.sam.lower() == "true"
    use_original_config = not use_sam

    h5_path = f"/zhome/02/e/181021/Desktop/Keras_Models_Images/{args.folder}/saved_models/{args.name}"

    x_train, y_train, x_test, y_test = load_dummy_cifar100()  # 🔁 Replace with your actual data loader

    model, compile_config = load_model_with_original_config(h5_path, use_sam=use_sam)
    model.summary()

    retrain_model(model, x_train, y_train, x_test, y_test,
                  use_sam=use_sam,
                  use_original_config=use_original_config,
                  compile_config=compile_config)
