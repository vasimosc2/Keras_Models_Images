import os
import psutil  # type: ignore # For measuring memory usage
from Training.train_and_evaluate import train_and_evaluate_model
from models.deeper_cnn import create_deeper_cnn
from models.simple_cnn import create_simple_cnn
from models.cnn_with_gap import create_cnn_with_gap
from models.cnn_with_batchnorm import create_cnn_with_batchnorm
from models.cnn_with_dropout import create_cnn_with_dropout
from models.resnet_like import create_resnet_like_cnn
import tensorflow as tf  # type: ignore
import pandas as pd
from tensorflow.keras import backend as K  # type: ignore
import tensorflow_model_optimization as tfmot  # type: ignore # Import TFMOT
from sklearn.metrics import precision_score, recall_score  # type: ignore # Import precision and recall score functions


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

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
    
    # Train and evaluate the original model
    acc, precision, recall, model_size, flops, max_ram, param_mem, total_ram_mem = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)

    # Store original model results
    results.append({
        "Model": model_name,
        "Type": "Original",
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Size_MB": model_size,
        "Flops_K": flops,
        "Max_RAM_KB": max_ram,
        "Param_Memory_KB": param_mem,
        "Total_Memory_KB": total_ram_mem
    })

    # Quantize the model
    quantized_model = tfmot.quantization.keras.quantize_model(model)

    # Compile the quantized model
    quantized_model.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    # Save the quantized model
    quantized_model_path = f'saved_models/quantized_{model_name}.keras'
    quantized_model.save(quantized_model_path)

    # Evaluate the quantized model
    quantized_test_loss, quantized_test_acc = quantized_model.evaluate(x_test, y_test, verbose=2)

    # Measure size of the quantized model
    quantized_model_file_size = os.path.getsize(quantized_model_path)
    quantized_model_size_in_mb = quantized_model_file_size / (1024 ** 2)

    # Get predictions from the quantized model to calculate precision and recall
    y_pred = quantized_model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_test, axis=1)

    # Calculate precision and recall for the quantized model
    quantized_precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    quantized_recall = recall_score(y_true_classes, y_pred_classes, average='weighted')

    # Measure RAM usage before and after quantization
    process = psutil.Process(os.getpid())
    quantized_max_ram = process.memory_info().rss / 1024  # Resident Set Size in KB
    quantized_param_mem = sum([K.count_params(w) for w in quantized_model.trainable_weights]) / 1024  # KB
    quantized_total_ram_mem = quantized_max_ram + quantized_param_mem  # Update as needed

    # Store quantized model results
    results.append({
        "Model": model_name,
        "Type": "Quantized",
        "Accuracy": quantized_test_acc,
        "Precision": quantized_precision,
        "Recall": quantized_recall,
        "Size_MB": quantized_model_size_in_mb,
        "Flops_K": flops,                  # Use original FLOPS for comparison
        "Max_RAM_KB": quantized_max_ram,   # Use measured RAM for quantized model
        "Param_Memory_KB": quantized_param_mem,  # Use measured parameter memory for quantized model
        "Total_Memory_KB": quantized_total_ram_mem  # Use measured total memory for quantized model
    })

    # Clear the session to free up memory after each model
    K.clear_session()

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('results/evaluation_results_with_quantization.csv', index=False)
print("Evaluation results saved to 'results/evaluation_results_with_quantization.csv'")
