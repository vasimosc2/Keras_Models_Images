import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score # type: ignore
from Counting.count_Flops import count_flops
from Counting.peak_ram import estimate_max_memory_usage

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