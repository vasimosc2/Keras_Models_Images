import os
from Counting.count_Flops import count_flops
from Counting.peak_ram import estimate_max_memory_usage
from models.deeper_cnn import create_deeper_cnn
from models.simple_cnn import create_simple_cnn
from models.cnn_with_gap import create_cnn_with_gap
from models.cnn_with_batchnorm import create_cnn_with_batchnorm
from models.cnn_with_dropout import create_cnn_with_dropout
from models.resnet_like import create_resnet_like_cnn  # Import new ResNet-like model
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

#tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to train and evaluate models
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=2)

    # Save model after training
    model_path = f'saved_models/{model_name}.keras'
    model.save(model_path)
    print(f"Model {model_name} saved!")

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Compute model file size
    model_file_size = os.path.getsize(model_path)
    model_size_in_mb = model_file_size / (1024 ** 2)

    flops = count_flops(model, batch_size=1) / 10**3

    print(f"The FLOPS are : {flops}")

    max_ram_usage, param_memory, total_memory = estimate_max_memory_usage(model)
    print(f"Max RAM Usage: {max_ram_usage:.2f} KB")
    print(f"Parameter Memory: {param_memory:.2f} KB")
    print(f"Total Memory Usage: {total_memory:.2f} KB")

    return test_acc, precision, recall, model_size_in_mb, flops, max_ram_usage, param_memory, total_memory

# Ensure directories exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# List of models to train
models_to_train = {
    "Simple_CNN": create_simple_cnn(),
    "Deeper_CNN": create_deeper_cnn(),
    "CNN_With_Dropout": create_cnn_with_dropout(),
    "CNN_With_BatchNorm": create_cnn_with_batchnorm(),
    "CNN_With_GAP": create_cnn_with_gap(),
    "ResNet_Like_CNN": create_resnet_like_cnn()
}

results = []

for model_name, model in models_to_train.items():
    print(f"\nTraining {model_name}...")
    acc, precision, recall, model_size, flops , max_ram, param_mem, total_Ram_mem= train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Size_MB": model_size,
        "Flops_K": flops,
        "Max_RAM_KB": max_ram,
        "Param_Memory_KB": param_mem,
        "Total_Memory_KB": total_Ram_mem
    })

    # Clear the session to free up memory after each model
    K.clear_session()

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('results/evaluation_results2.csv', index=False)
print("Evaluation results saved to 'results/evaluation_results2.csv'")

# Load one model and compute FLOPs
# model = load_model('saved_models/Simple_CNN.keras')
# model.summary()

# flops = count_flops(model, batch_size=1)
# print(f"FLOPS: {flops / 10 ** 9:.03f} G")

# model_file_size = os.path.getsize('saved_models/Simple_CNN.keras')
# model_size_in_mb = model_file_size / (1024 ** 2)
# print(f'Model size on disk: {model_size_in_mb:.2f} MB')
