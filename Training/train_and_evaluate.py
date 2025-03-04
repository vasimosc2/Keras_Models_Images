import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score # type: ignore
from Counting.count_Flops import count_flops
from Counting.peak_ram import estimate_max_memory_usage

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name:str, params:dict):

    model.compile(optimizer=params["optimaizer"],
                  loss=params["loss"],
                  metrics=['accuracy'])
    
    # Start Training
    start_time = time.time()

    history = model.fit(x_train, y_train, epochs=params["number_of_epochs"], batch_size=params["batch_size"], validation_data=(x_test, y_test), verbose=2)
    
    final_train_acc = history.history['accuracy'][-1]
    final_test_acc = history.history['val_accuracy'][-1]

    training_time = time.time() - start_time
    # Save model after training
    model_path = f'saved_models/{model_name}.keras'
    model.save(model_path)
    print(f"Model {model_name} saved!")
    print(f"Test Accuracy: {final_test_acc}")

    if final_train_acc - final_test_acc > 0.05:
        print(f"⚠️ Overfitting detected for model {model_name}! Adjusting model...")
        

    # Underfitting Detected (Test Acc < 70%)
    elif final_test_acc < 0.70:
        print(f"⚠️ Underfitting detected for model {model_name}! Increasing model complexity...")
        


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

    max_ram_usage, param_memory, total_memory = estimate_max_memory_usage(model=model,dtype_size=params["dtype_size"])
    print(f"Max RAM Usage: {max_ram_usage:.2f} KB")
    print(f"Parameter Memory: {param_memory:.2f} KB")
    print(f"Total Memory Usage: {total_memory:.2f} KB")

    return final_test_acc,final_train_acc, precision, recall, model_size_in_mb, flops, max_ram_usage, param_memory, total_memory, training_time