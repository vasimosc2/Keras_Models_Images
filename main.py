import os
from Counting.count_Flops import count_flops
from models.simple_cnn import create_simple_cnn
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



# Add other model functions like create_deeper_cnn, create_cnn_with_dropout, etc. here

# Function to train and evaluate models
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=2)

    # Save model after training
    model.save(f'saved_models/{model_name}.keras')
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

    return test_acc, precision, recall

# Ensure the 'saved_models' directory exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Train models
models_to_train = [
    create_simple_cnn(),
    # Add other models like create_deeper_cnn(), create_cnn_with_dropout(), etc.
]

results = []

for i, model in enumerate(models_to_train):
    model_name = f"model_{i + 1}"
    print(f"\nTraining Model {i + 1}...")
    acc, precision, recall = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall
    })

    # Clear the session to free up memory after each model
    K.clear_session()

# Save evaluation results to a CSV file
if not os.path.exists('results'):
    os.makedirs('results')

df_results = pd.DataFrame(results)
df_results.to_csv('results/evaluation_results.csv', index=False)
print("Evaluation results saved to 'results/evaluation_results.csv'")


model = load_model('saved_models/model_1.keras')
model.summary()

flops = count_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03f} G")


model_file_size = os.path.getsize('saved_models/model_1.keras')
model_size_in_mb = model_file_size / (1024 ** 2)
print(f'Model size on disk: {model_size_in_mb:.2f} MB')