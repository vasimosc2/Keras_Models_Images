import numpy as np
import time
import os
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, Model # type: ignore
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score # type: ignore
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop # type: ignore

class TakuNetModel:
    def __init__(self, model_name:str, input_shape: Tuple[int, int, int], model_params: Dict, train_params:Dict, x_train, y_train, x_test, y_test):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model_params = model_params
        self.train_params = train_params
        self.model = self._build_model()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.results = TrainingResults()
    
    def _stem_block(self, inputs:tuple):
        """
        The input shape is: (None,32,32,3) (Given input 32,32,3)
        The output shape is: (None, 32 / (Conv_strides * DWConv_kernel), 32 / (Conv_strides * DWConv_kernel), filters,)
        """
        print(f"Stem 1 block shape {inputs.shape}\n")
        x = layers.Conv2D(filters=self.model_params["stem_block"]["filters"], 
                          kernel_size=self.model_params["stem_block"]["Conv_kernel"],
                          strides=self.model_params["stem_block"]["Conv_strides"], 
                          padding='same', 
                          use_bias=False)(inputs)
        print(f"Stem 2 block shape {x.shape}\n")
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        if self.model_params["stem_block"]["dropout"] > 0:
            x = layers.Dropout(self.model_params["stem_block"]["dropout"])(x)
        x = layers.DepthwiseConv2D(kernel_size=self.model_params["stem_block"]["DWConv_kernel"],
                                   strides=self.model_params["stem_block"]["DWConv_strides"],
                                   padding='same', 
                                   use_bias=False)(x)
        print(f"Stem 3 block shape {x.shape}\n")
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    def _taku_block(self, inputs:tuple):
        print(f"TakuBlock 1 shape {inputs.shape}\n")
        x = layers.DepthwiseConv2D(kernel_size=self.model_params["stages_block"]["taku_block"]["DWConv_kernel"], 
                                   strides=self.model_params["stages_block"]["taku_block"]["DWConv_strides"], 
                                   padding='same', 
                                   use_bias=False)(inputs)
        
        print(f"TakuBlock 2 shape {x.shape}\n")
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        if self.model_params["stages_block"]["taku_block"]["dropout"] > 0:
            x = layers.Dropout(self.model_params["stages_block"]["taku_block"]["dropout"])(x)
        return layers.Add()([x, inputs])
    
    def _downsampler_block(self, inputs:tuple, curr_stage_number):
        print(f"DownSample input shape {inputs.shape}\n")
        filters = inputs.shape[-1]
        num_groups = max(1, min(self.model_params["stages_block"]["stages_number"], filters))
        if filters % num_groups != 0:
            num_groups = 1  
        kernel_size = min(self.model_params["stages_block"]["downsampler"]["Conv_kernel"], inputs.shape[1], inputs.shape[2])
        
        x = layers.Conv2D(filters=filters, 
                          kernel_size=kernel_size, 
                          groups=num_groups, 
                          use_bias=False)(inputs)
        print(f"DownSample 2 shape {x.shape}\n")
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        if self.model_params["stages_block"]["downsampler"]["dropout"] > 0:
            x = layers.Dropout(self.model_params["stages_block"]["downsampler"]["dropout"])(x)
        pool_layer = layers.MaxPooling2D if curr_stage_number < self.model_params["stages_block"]["stages_number"] else layers.AveragePooling2D
        x = pool_layer(pool_size=self.model_params["stages_block"]["downsampler"]["pool_size"], 
                       strides=self.model_params["stages_block"]["downsampler"]["strides"], 
                       padding='same')(x)
        print(f"DownSample 3 shape {x.shape}\n")
        return layers.LayerNormalization()(x)
    
    def _stage_block(self, inputs, curr_stage_number):
        x = inputs
        for _ in range(self.model_params["stages_block"]["taku_block"]["taku_block_number"]):
            x = self._taku_block(x)
        concat = layers.Concatenate()([inputs, x])
        return self._downsampler_block(concat, curr_stage_number)
    
    def _refiner_block(self, inputs):
        print(f"Refiner Block input shape {inputs.shape}\n")
        x = layers.DepthwiseConv2D(kernel_size=self.model_params["refiner_block"]["DWConv_kernel"], 
                                   strides=["refiner_block"]["DWConv_strides"], 
                                   padding='same', 
                                   use_bias=False)(inputs)
        print(f"Refiner Block 2 shape {x.shape}\n")
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.GlobalAveragePooling2D()(x)
        print(f"Refiner Block 3 shape {x.shape}\n")
        if self.model_params["refiner_block"]["dropout"] > 0:
            x = layers.Dropout(self.model_params["refiner_block"]["dropout"])(x)
        return layers.Dense(self.model_params["refiner_block"]["num_output_classes"], activation='softmax')(x)
    
    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self._stem_block(inputs)
        for curr_stage_number in range(self.model_params["stages_block"]["stages_number"]):
            x = self._stage_block(x, curr_stage_number)
        outputs = self._refiner_block(x)
        return Model(inputs, outputs)
    
    def train(self):
        """Train the model, evaluate metrics, and store results."""
        
        # **Memory Estimation Before Training**
        max_ram_usage, param_memory, total_memory = self.memoryEstimation(data_dtype_multiplier=self.train_params["data_dtype_multiplier"])
        print(f"Max RAM Usage: {max_ram_usage:.2f} KB")
        print(f"Parameter Memory: {param_memory:.2f} KB")
        print(f"Total Memory Usage: {total_memory:.2f} KB")

        if max_ram_usage * 1024 > self.train_params["max_ram_consumption"]:
            print(f"🚨 Training aborted: Estimated RAM usage ({max_ram_usage:.2f} KB) exceeds limit.")
            print("\n The results will be None in this Model")
            return None

        print("✅ Memory check passed! Starting training...")

        # **Compile Model**
        optimizer = get_optimizer(self.train_params["optimizer"], self.train_params["learning_rate"])
        self.model.compile(optimizer=optimizer, loss=self.train_params["loss"], metrics=['accuracy'])

        # **Callbacks**
        checkpoint_path = f'saved_models/{self.model_name}_best.keras'
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
        early_stopping_loss = EarlyStopping(monitor='val_loss', patience=self.train_params["learning_rate_patience"], restore_best_weights=True)
        early_stopping_acc = EarlyStopping(monitor='val_accuracy', patience=self.train_params["early_stopping_patience"], mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.train_params["learning_rate_patience"], verbose=1)
        midway_callback = MidwayStopCallback(total_epochs=self.train_params["num_epochs"], divider=self.train_params["divider"], threshold=0.30)

        # **Train Model with Timing**
        start_time = time.time()
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.train_params["num_epochs"],
            batch_size=self.train_params["batch_size"],
            validation_data=(self.x_test, self.y_test),
            verbose=2,
            callbacks=[midway_callback, early_stopping_acc, early_stopping_loss, reduce_lr, checkpoint]
        )

        training_time = time.time() - start_time

        # **Load Best Model**
        self.model.load_weights(checkpoint_path)
        print(f"✅ Best model restored from {checkpoint_path}")

        # **Compute Accuracy Metrics**
        best_train_acc = max(history.history['accuracy'])   # training accuracy
        best_test_acc = max(history.history['val_accuracy'])  # test accuracy

        print(f"✅ Best Test Accuracy (Best Model): {best_test_acc:.4f}")

        # **Overfitting Check**
        # if final_train_acc - final_test_acc > 0.05:
        #     print(f"⚠️ Overfitting detected for model {self.model_name}!")

        # elif final_test_acc < 0.70:
        #     print(f"⚠️ Underfitting detected for model {self.model_name}!")

        # **Predictions & Metrics**
        y_test_pred = self.model.predict(self.x_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        self.results.history = history
        self.results.train_accuracy = best_train_acc
        self.results.test_accuracy = best_test_acc
        self.results.precision = precision_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.recall = recall_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.f1_score = f1_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.max_ram_usage = max_ram_usage
        self.results.param_memory = param_memory
        self.results.total_memory = total_memory
        self.results.training_time = training_time 

        # **Save Model in Multiple Formats**
        keras_model_path = f'saved_models/{self.model_name}.keras'
        self.model.save(keras_model_path)
        print(f"✅ Model saved in Keras format: {keras_model_path}")

        self.convert_to_tflite()
        self.convert_tflite_to_c_array()

        # **Evaluate the TFLite Model**
        tflite_acc = self.evaluate_tflite_model()
        print(f"Test Accuracy (TFLite): {tflite_acc:.4f}")

        # **File Size Reporting**
        keras_size_kb = os.path.getsize(keras_model_path) / 1024
        tflite_size_kb = os.path.getsize(f"TfLiteModels/{self.model_name}.tflite") / 1024
        c_array_size_kb = os.path.getsize(f"HeaderFiles/{self.model_name}.h") / 1024

        print(f"Keras Model Size: {keras_size_kb:.2f} KB")
        print(f"TFLite Model Size: {tflite_size_kb:.2f} KB")
        print(f"C Array File Size: {c_array_size_kb:.2f} KB")

        print("\n✅ Training complete. Best model and metrics stored in `self.results`.\n")
    
    def memoryEstimation(self,data_dtype_multiplier: int = 4)-> Tuple[float, float, float]:
        """
        ROM (Read-Only Memory) → Memory used to store layer parameters (weights & biases).
        RAM (Random-Access Memory) → Memory used to store activations (input & output tensors).
        """
        max_activation_memory: int = 0  # Peak RAM usage
        total_param_memory: int = 0      # ROM for storing weights

        for layer in self.model.layers:
            
            layer_params: int = layer.count_params() #  Number of parameters in the layer (weights & biases).
            layer_param_memory: int = layer_params * data_dtype_multiplier#  Converts the number of parameters into bytes.
            total_param_memory += layer_param_memory # Adds up all the layer_param_memory of each layer

            # Compute activation memory (RAM)
            if isinstance(layer.output, list):
                output_memory: int = sum(np.prod(out.shape[1:]) * data_dtype_multiplier for out in layer.output) # I wont be inside there are layer.output is  <class 'keras.src.backend.common.keras_tensor.KerasTensor'>
            else:
                output_memory: int = np.prod(layer.output.shape[1:]) * data_dtype_multiplier # If the output shape is 30 x 30 x 32 , the output memmory is  28800 * data_size

            if isinstance(layer.input, list):
                input_memory: int = sum(np.prod(inp.shape[1:]) * data_dtype_multiplier for inp in layer.input)
            else:
                input_memory: int = np.prod(layer.input.shape[1:]) * data_dtype_multiplier

            # Track peak RAM usage
            layer_ram_usage: int = input_memory + output_memory
            max_activation_memory = max(max_activation_memory, layer_ram_usage) # Here we keep the the maximum use of RAM of each layer

        # Convert bytes to KB
        max_ram_usage: float = max_activation_memory / 1024
        param_memory: float = total_param_memory / 1024
        total_memory: float = (max_activation_memory + total_param_memory) / 1024

        return max_ram_usage, param_memory, total_memory
    
    def count_flops(self, batch_size=1)-> int:
        """
        Count FLOPs of a TensorFlow 2.x model.
        
        Parameters:
            model (tf.keras.Model): The model whose FLOPs need to be counted.
            batch_size (int): The batch size for FLOP computation.

        Returns:
            int: The total number of FLOPs in the model.
        """
        # Create a concrete function from the model call
        input_shape = (batch_size,) + self.input_shape
        dummy_input = tf.ones(input_shape)

        # Convert model to a TensorFlow function graph
        concrete_function = tf.function(self.model).get_concrete_function(dummy_input)
        frozen_func = concrete_function.graph

        # Count the number of float operations
        flops = 0
        for op in frozen_func.get_operations():
            for output in op.outputs:
                shape = output.shape
                if shape.is_fully_defined():
                    flops += tf.reduce_prod(shape).numpy()

        return flops
    
    def convert_to_tflite(self)->None:
        """Converts a trained model to TFLite with full-integer quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

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
        os.makedirs('TfLiteModels', exist_ok=True)
        tflite_model_path = f"TfLiteModels/{self.model_name}.tflite"
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ Model converted and saved as {tflite_model_path}")

    def convert_tflite_to_c_array(self)->None:
        """Converts the TFLite model into a C array header file for Arduino integration."""
        with open(f"TfLiteModels/{self.model_name}.tflite", "rb") as f:
            tflite_model = f.read()

        c_array = ", ".join(f"0x{byte:02x}" for byte in tflite_model)
        model_length = len(tflite_model)

        header_content = f"""#ifndef {self.model_name.upper()}_H
        #define {self.model_name.upper()}_H

        // Model converted to C array for Arduino
        const unsigned char {self.model_name}_data[{model_length}] = {{
            {c_array}
        }};

        unsigned int {self.model_name}_length = {model_length};

        #endif // {self.model_name.upper()}_H
        """
        os.makedirs('HeaderFiles', exist_ok=True)
        header_file_path = f"HeaderFiles/{self.model_name}.h"
        with open(header_file_path, "w") as f:
            f.write(header_content)

        print(f"✅ C header file saved as {header_file_path}")
    
    
        
    def evaluate_tflite_model(self)-> float:
        """Evaluates the TFLite model and returns the accuracy."""
        interpreter = tf.lite.Interpreter(model_path=f"TfLiteModels/{self.model_name}.tflite")

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Debugging prints
        print("✅ Model loaded successfully!")
        print("📌 Input Details:", input_details)
        print("📌 Output Details:", output_details)
        print("Expected Input Shape:", input_details[0]['shape'])
        print("Actual Input Shape:", self.x_test[0].shape)

        def preprocess_input(input_data):
            """Adjusts input data if the model uses uint8 quantization."""
            if input_details[0]["dtype"] == np.uint8:
                scale, zero_point = input_details[0]["quantization"]
                input_data = np.round(input_data / scale + zero_point).astype(np.uint8)
            return input_data

        y_pred = []
        for i in range(len(self.x_test)):
            input_data = preprocess_input(self.x_test[i:i+1])

            # Ensure shape is correct
            input_data = np.reshape(input_data, input_details[0]['shape'])
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            
            if output_details[0]["dtype"] == np.uint8:
                scale, zero_point = output_details[0]["quantization"]
                output = (output.astype(np.float32) - zero_point) * scale
            
            y_pred.append(output)

        y_pred = np.array(y_pred).squeeze()
        y_pred_classes = np.argmax(y_pred, axis=-1) if output.ndim > 1 else (output > 0.5).astype(np.int32)
        y_true_classes = np.argmax(self.y_test, axis=-1)
        accuracy = np.mean(y_pred_classes == y_true_classes)
        return accuracy



    def summary(self):
        self.model.summary()
    
    def get_model(self):
        return self.model


# Helpers

def get_optimizer(name, learning_rate, weight_decay=1e-4):
    """Returns the optimizer instance based on the name."""
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "adamw": AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        "sgd": SGD(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(name.lower(), Adam(learning_rate=learning_rate))  # If the name is not found return Adam by default


class MidwayStopCallback(Callback):
    def __init__(self, total_epochs:int, divider:int, threshold:float):
        super().__init__()
        self.mid_epoch = total_epochs // divider 
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.mid_epoch:
            train_acc = logs.get('accuracy')
            val_acc = logs.get('val_accuracy')
            print(f"\nMidway Epoch {epoch+1}: Training Acc = {train_acc}, Validation Acc = {val_acc}")
            if train_acc < self.threshold:  
                print(f"\n🚨 Stopping early: Training accuracy is below {self.threshold} at epoch {epoch+1}")
                self.model.stop_training = True


class TrainingResults:
    """Class to store training and evaluation results."""
    def __init__(self):
        self.history = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.max_ram_usage = None
        self.param_memory = None
        self.total_memory = None
        self.training_time = None
        self.fitness_score = None

    def __repr__(self):
        return (f"TrainingResults(\n"
                f"  Train Accuracy: {self.train_accuracy:.4f}\n"
                f"  Test Accuracy: {self.test_accuracy:.4f}\n"
                f"  Precision: {self.precision:.4f}\n"
                f"  Recall: {self.recall:.4f}\n"
                f"  F1 Score: {self.f1_score:.4f}\n"
                f"  Max Ram Use: {self.max_ram_usage:.4f}\n"
                f"  Max Param Memory Use: {self.param_memory:.4f}\n"
                f"  Total_memory Use: {self.total_memory:.4f}\n"
                f"  Training Time: {self.training_time}\n)")