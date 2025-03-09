# 📸 Image Recognition Project

This project is designed for efficient image recognition and consists of four key files:

## 📂 Project Files
1. **`main.py`** – The main script orchestrating the model's functionality.
2. **`config.json`** – Defines the hyperparameter search space and training configuration.
3. **`Training/train_and_evaluate.py`** – Handles model training and evaluation.
4. **`models/takunet.py`** – Contains the Takunet model architecture.

---

## 🛠 `config.json`: Model Configuration
This JSON file dictates the hyperparameters and constraints for training, ensuring optimal model performance within resource limitations.

### 🔹 **Model Architecture**
- **`stages`** – Number of Takublocks in the model.
- **`filters`** – Filters are small matrices that scan over an image to detect patterns like edges, textures, or shapes. 
                  The number of filters determines how many different features the model can learn at each layer.
                  More filters → The model learns more complex patterns but requires more computation.
                  Fewer filters → Faster training, but might miss some details.
                
- **`kernel_size`** – The kernel size defines the height × width of the small matrix (filter) that moves across the image. 
                      Common sizes include 3×3, 5×5, and 7×7.
                      Smaller kernels (e.g., 3×3) → Better at detecting fine details like edges and textures.
                      Larger kernels (e.g., 5×5, 7×7) → Capture bigger patterns but may miss fine details.

- **`activation`** – Activation function in convolutional layers.
- **`strides`** – Step size for the convolutional filter.
- **`dropout_rate`** – Probability of neuron dropout for overfitting prevention.
- **`num_units`** – Neurons in fully connected (dense) layers.
- **`dense_activation`** – Activation function for dense layers.
- **`num_output_classes`** – Number of classification output classes.

### 🔹 **Training Parameters**
- **`optimizer`** – Optimization algorithm for training.
- **`loss`** – Loss function (e.g., `categorical_crossentropy` for multi-class classification).

- **`learning_rate`** – The learning rate (LR) controls how much the model updates weights during training.
                        High LR → Model learns fast but might overshoot the optimal point.
                        Low LR → Model learns slowly but might get stuck in local minima.
                        
- **`num_epochs`** –  An epoch is one full pass through the entire dataset during training.
                      Too few epochs → The model might underfit (not learn enough patterns).
                      Too many epochs → The model might overfit (memorizing the training data instead of generalizing well to new data).

- **`batch_size`** - Batch size refers to the number of training samples processed before the model updates its weights.
                     Smaller batch sizes → More frequent updates, more generalization, but slower training.
                     Larger batch sizes → Faster training, but may lead to less generalization.

### 🔹 **Memory Constraints** *(Optimized for Arduino Nano 33 BLE Sense)*
- **`max_ram_consumption`** – Maximum RAM usage (256 KB).
- **`max_flash_consumption`** – Maximum flash memory (1 MB).
- **`data_dtype_multiplier`** – Memory scaling based on data type (`int8`, thus `1`).
- **`model_dtype_multiplier`** – Scaling factor for model precision.

---
This structured configuration ensures streamlined model training while adhering to the hardware constraints of resource-limited environments 
like the **Arduino Nano 33 BLE Sense**. 🚀

## Setup

To set up the environment, install the required dependencies:

python3.11 -m venv venv
or
python -m venv venv

```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

pip freeze