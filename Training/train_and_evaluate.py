import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score # type: ignore
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from Counting.count_Flops import count_flops
from Counting.peak_ram import estimate_max_memory_usage
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # type: ignore

def get_optimizer(name, learning_rate):
    """Returns the optimizer instance based on the name."""
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(name.lower(), Adam(learning_rate=learning_rate)) 

class MidwayStopCallback(Callback):
    def __init__(self, total_epochs, threshold=0.35):
        super().__init__()
        self.mid_epoch = total_epochs // 2
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.mid_epoch:
            train_acc = logs.get('accuracy')
            val_acc = logs.get('val_accuracy')
            print(f"\nMidway Epoch {epoch+1}: Training Acc = {train_acc}, Validation Acc = {val_acc}")
            if train_acc < self.threshold:  # Stop training if training accuracy is too low
                print(f"\n🚨 Stopping early: Training accuracy is below {self.threshold} at epoch {epoch+1}")
                self.model.stop_training = True

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name:str, params:dict):
    optimizer = get_optimizer(params["optimizer"], params["learning_rate"])

    model.compile(optimizer=optimizer,
                  loss=params["loss"],
                  metrics=['accuracy'])

    # Callbacks
    midway_callback = MidwayStopCallback(params["num_epochs"], threshold=0.50)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    checkpoint = ModelCheckpoint(filepath=f'saved_models/{model_name}.keras', save_best_only=True)

    # Start Training
    start_time = time.time()
    history = model.fit(x_train, y_train, 
                        epochs=params["num_epochs"], 
                        batch_size=params["batch_size"], 
                        validation_data=(x_test, y_test), 
                        verbose=2,
                        callbacks=[midway_callback, early_stopping, reduce_lr, checkpoint])

    final_train_acc = history.history['accuracy'][-1]
    final_test_acc = history.history['val_accuracy'][-1]

    training_time = time.time() - start_time
    print(f"Test Accuracy: {final_test_acc}")

    # Overfitting and underfitting detection
    if final_train_acc - final_test_acc > 0.05:
        print(f"⚠️ Overfitting detected for model {model_name}! Adjusting model...")

    elif final_test_acc < 0.70:
        print(f"⚠️ Underfitting detected for model {model_name}! Increasing model complexity...")

    # Predictions & Metrics
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Compute model file size
    model_file_size = os.path.getsize(f'saved_models/{model_name}.keras')
    model_size_in_mb = model_file_size / (1024 ** 2)

    # FLOPS & Memory Usage
    flops = count_flops(model, batch_size=1) / 10**3
    max_ram_usage, param_memory, total_memory = estimate_max_memory_usage(model=model, data_dtype_multiplier=params["data_dtype_multiplier"])

    print(f"FLOPS: {flops}")
    print(f"Max RAM Usage: {max_ram_usage:.2f} KB")
    print(f"Parameter Memory: {param_memory:.2f} KB")
    print(f"Total Memory Usage: {total_memory:.2f} KB")

    return final_test_acc, final_train_acc, precision, recall, model_size_in_mb, flops, max_ram_usage, param_memory, total_memory, training_time
