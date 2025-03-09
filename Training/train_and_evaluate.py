import numpy as np
import time
import os
import tensorflow as tf  # type: ignore
from sklearn.metrics import precision_score, recall_score  # type: ignore
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
from Counting.count_Flops import count_flops
from Counting.peak_ram import estimate_max_memory_usage
from tensorflow.keras.optimizers import Adam, AdamW , SGD, RMSprop  # type: ignore

def get_optimizer(name, learning_rate, weight_decay=1e-4):
    """Returns the optimizer instance based on the name."""
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "adamw": AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        "sgd": SGD(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(name.lower(), Adam(learning_rate=learning_rate)) 


class MidwayStopCallback(Callback):
    def __init__(self, total_epochs, threshold=0.40):
        super().__init__()
        self.mid_epoch = total_epochs // 5  
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.mid_epoch:
            train_acc = logs.get('accuracy')
            val_acc = logs.get('val_accuracy')
            print(f"\nMidway Epoch {epoch+1}: Training Acc = {train_acc}, Validation Acc = {val_acc}")
            if train_acc < self.threshold:  
                print(f"\n🚨 Stopping early: Training accuracy is below {self.threshold} at epoch {epoch+1}")
                self.model.stop_training = True


def convert_to_tflite(model, model_name):
    """Converts a trained model to TFLite with full-integer quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # **Enable optimizations and quantization**
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # **Use a representative dataset to optimize quantization**
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 32, 32, 3).astype(np.float32)
            yield [data]
    converter.representative_dataset = representative_dataset

    # **Ensure full integer quantization for microcontroller compatibility**
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    # **Save TFLite model**
    tflite_model_path = f"saved_models/{model_name}.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Model converted and saved as {tflite_model_path}")
    return tflite_model_path


def convert_tflite_to_c_array(tflite_model_path, model_name):
    """Converts the TFLite model into a C array header file for Arduino integration."""
    with open(tflite_model_path, "rb") as f:
        tflite_model = f.read()

    c_array = ", ".join(f"0x{byte:02x}" for byte in tflite_model)
    model_length = len(tflite_model)

    header_content = f"""#ifndef {model_name.upper()}_H
#define {model_name.upper()}_H

// Model converted to C array for Arduino
const unsigned char {model_name}_data[{model_length}] = {{
    {c_array}
}};

unsigned int {model_name}_length = {model_length};

#endif // {model_name.upper()}_H
"""

    header_file_path = f"saved_models/{model_name}.h"
    with open(header_file_path, "w") as f:
        f.write(header_content)

    print(f"✅ C header file saved as {header_file_path}")


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name: str, params: dict):
    """Trains a model and evaluates both the Keras and TFLite versions."""

    max_ram_usage, param_memory, total_memory = estimate_max_memory_usage(model=model, data_dtype_multiplier=params["data_dtype_multiplier"])

    print(f"Max RAM Usage: {max_ram_usage:.2f} KB")
    print(f"Parameter Memory: {param_memory:.2f} KB")
    print(f"Total Memory Usage: {total_memory:.2f} KB")

    if max_ram_usage * 1024 > params["max_ram_consumption"]:
        print(f"🚨 Training aborted: Estimated RAM usage ({max_ram_usage:.2f} KB) exceeds limit ({params['max_ram_consumption']/1024} KB).")
        return None  

    print("✅ Memory check passed! Starting training...")

    optimizer = get_optimizer(params["optimizer"], params["learning_rate"])

    model.compile(optimizer=optimizer, loss=params["loss"], metrics=['accuracy'])

    # **Callbacks**
    midway_callback = MidwayStopCallback(params["num_epochs"], threshold=0.30)
    early_stopping_loss = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    early_stopping_acc = EarlyStopping(monitor='val_accuracy', patience=8, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    checkpoint = ModelCheckpoint(filepath=f'saved_models/{model_name}.keras', save_best_only=True)

    # **Start Training**
    start_time = time.time()
    history = model.fit(
        x_train, y_train, 
        epochs=params["num_epochs"], 
        batch_size=params["batch_size"], 
        validation_data=(x_test, y_test),
        verbose=2,
        callbacks=[midway_callback, early_stopping_acc, early_stopping_loss, reduce_lr, checkpoint]
    )

    final_train_acc = history.history['accuracy'][-1]
    final_test_acc = history.history['val_accuracy'][-1]
    training_time = time.time() - start_time

    print(f"Test Accuracy (Keras): {final_test_acc:.4f}")

    # **Overfitting detection**
    if final_train_acc - final_test_acc > 0.05:
        print(f"⚠️ Overfitting detected for model {model_name}!")

    elif final_test_acc < 0.70:
        print(f"⚠️ Underfitting detected for model {model_name}!")

    # **Predictions & Metrics (Keras)**
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # **Save model in multiple formats**
    keras_model_path = f'saved_models/{model_name}.keras'
    model.save(keras_model_path)
    print(f"✅ Model saved in Keras format: {keras_model_path}")

    tflite_model_path = convert_to_tflite(model, model_name)
    convert_tflite_to_c_array(tflite_model_path, model_name)

    # **Evaluate the TFLite Model**
    tflite_acc = evaluate_tflite_model(tflite_model_path, x_test, y_test)

    print(f"Test Accuracy (TFLite): {tflite_acc:.4f}")

    model_file_size = os.path.getsize(f'saved_models/{model_name}.keras')
    Keras_size_in_KB = model_file_size / 1024

    tflite_model_size_bytes = os.path.getsize(tflite_model_path)  # Get file size in bytes
    tflite_model_size_kb = tflite_model_size_bytes / 1024  # Convert to KB

    print(f"TFLite Model Size: {tflite_model_size_kb:.2f} KB")

    c_array_file_path = f"saved_models/{model_name}.h"
    c_array_file_size_bytes = os.path.getsize(c_array_file_path)  
    c_array_file_size_kb = c_array_file_size_bytes / 1024 
    print(f"C Array Header File Size: {c_array_file_size_kb:.2f} KB")

    return final_test_acc, tflite_acc, final_train_acc, precision, recall, Keras_size_in_KB, tflite_model_size_kb, c_array_file_size_kb, count_flops(model, batch_size=1) / 10**3, max_ram_usage, param_memory, total_memory, training_time


def evaluate_tflite_model(tflite_model_path, x_test, y_test):
    """Evaluates the TFLite model and returns the accuracy."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def preprocess_input(input_data):
        """Adjusts input data if the model uses uint8 quantization."""
        if input_details[0]["dtype"] == np.uint8:
            scale, zero_point = input_details[0]["quantization"]
            input_data = tf.cast(input_data / scale + zero_point, tf.uint8)
        return input_data

    y_pred = []
    for i in range(len(x_test)):
        input_data = preprocess_input(x_test[i:i+1])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]["dtype"] == np.uint8:
            scale, zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        y_pred.append(output)

    y_pred = np.array(y_pred).squeeze()  # Remove unnecessary dimensions
    y_pred_classes = np.argmax(y_pred, axis=-1)  # Ensure correct reduction

    y_true_classes = np.argmax(y_test, axis=-1)  # Ensure correct shape

    accuracy = np.mean(y_pred_classes == y_true_classes)

    return accuracy