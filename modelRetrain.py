import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import deserialize as deserialize_optimizer
from tensorflow.keras.losses import deserialize as deserialize_loss
from tensorflow.keras.metrics import deserialize as deserialize_metric
import warnings

from TakuNet import AdaptiveDropout

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.backend.tensorflow.trainer")

from Models.SAM import SAMModel

def load_dummy_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    return x_train, y_train, x_test, y_test

def retrain_model(base_model, x_train, y_train, x_test, y_test,
                  use_sam=False, compile_config=None, save_path=None):
    warmup_epochs = 5
    initial_lr = 0.05
    total_epochs = 50
    batch_size = 16

    def cosine_annealing_with_warmup(epoch):
        if epoch < warmup_epochs:
            return float(initial_lr * (epoch + 1) / warmup_epochs)
        else:
            cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return float(initial_lr * cosine_decay)

    if use_sam:
        model = SAMModel(base_model)
    else:
        model = base_model

    if compile_config is not None:
        print("🔁 Using original compile configuration")
        optimizer = deserialize_optimizer(compile_config['optimizer'])
        loss = deserialize_loss(compile_config['loss'])
        metrics = [deserialize_metric(m) for m in compile_config.get('metrics', [])]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        print("⚙️ Using custom training configuration")
        model.compile(
            optimizer=SGD(learning_rate=initial_lr, momentum=0.9),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
    ]

    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, save_best_only=True,
                                         monitor='val_accuracy', mode='max'))

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=total_epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"✅ Final test accuracy: {acc:.4f}")

    if save_path:
        model.save(save_path, save_format="keras")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model with optional SAM support")
    parser.add_argument("--name", type=str, default="TakuNet_Random_0.keras", help="Model file name")
    parser.add_argument("--folder", type=str, default="Manual_Run", help="Model folder")
    parser.add_argument("--sam", type=str, default="false", help="Use SAMModel: true or false")
    args = parser.parse_args()

    use_sam = args.sam.lower() == "true"
    model_path = f"/zhome/02/e/181021/Desktop/Keras_Models_Images/{args.folder}/saved_models/{args.name}"

    print(f"📦 Loading model from: {model_path}")
    base_model = load_model(model_path, custom_objects={"SAMModel": SAMModel, "AdaptiveDropout": AdaptiveDropout})

    compile_config = base_model.get_config().get("compile_config", None)

    x_train, y_train, x_test, y_test = load_dummy_cifar100()
    base_model.summary()

    retrain_model(base_model, x_train, y_train, x_test, y_test,
                  use_sam=use_sam,
                  compile_config=compile_config,
                  save_path=model_path)
