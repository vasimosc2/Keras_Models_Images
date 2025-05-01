import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from tensorflow.keras import regularizers
from utils import memoryEstimator
import math
import random

class TakuNetModel:
    def __init__(self, 
                model_name:str, 
                input_shape: Tuple[int, int, int] = (32, 32, 3), 
                model_params: Optional[Dict] = None, 
                train_params: Optional[Dict] = None,
                x_train: Optional[tf.Tensor]= None, 
                y_train: Optional[tf.Tensor]= None, 
                x_test: Optional[tf.Tensor] = None, 
                y_test: Optional[tf.Tensor] = None,
                folder:Optional[str] = None,
                given_model:Optional[tf.keras.Model] = None
                ):
        
        self.model_name:str = model_name
        self.input_shape: Tuple[int, int, int] = input_shape
        self.model_params: Optional[Dict] = model_params
        self.train_params: Optional[Dict] = train_params
        
        self.adaptive_dropout_stem: AdaptiveDropout = None
        self.adaptive_dropout_taku: List[AdaptiveDropout] = [] # This will have Length As much as the Stages
        self.adaptive_dropout_refiner: List[AdaptiveDropout] = [] # This will have a fix lenght of 2
        
        self.model:tf.keras.Model = given_model if given_model else self._build_model()
        self.x_train: Optional[tf.Tensor] = x_train
        self.y_train: Optional[tf.Tensor] = y_train
        self.x_test: Optional[tf.Tensor] = x_test
        self.y_test: Optional[tf.Tensor] = y_test
        
        self.is_trained:bool = False
        self.folderName:str = folder if folder is not None else "."
        self.epochs:int = None
        self.learningRate:Optional[float] = 0.0005 if given_model else None
        self.results: TrainingResults = TrainingResults()
        self.is_trainable: bool = self.check_trainability()

  

    
    def _stem_block(self, inputs:tuple):
        """
        The input shape is: (None,32,32,3) (Given input 32,32,3)
        The output shape is: (None, 32 / (Conv_strides * DWConv_stride), 32 / (Conv_strides * DWConv_stride), filters)
        """

        x = layers.Conv2D(filters=self.model_params["stem_block"]["filters"], 
                          kernel_size=self.model_params["stem_block"]["Conv_kernel"],
                          strides=self.model_params["stem_block"]["Conv_strides"], 
                          padding='same', 
                          use_bias=False,
                          kernel_regularizer = regularizers.l2(self.model_params["stem_block"]["l2_weight_decay"]) )(inputs)
        
        x = layers.BatchNormalization()(x)

        x = layers.ReLU(6.0)(x)

        self.adaptive_dropout_stem = AdaptiveDropout(initial_rate=0.05, name="adaptive_dropout_stem")
        x = self.adaptive_dropout_stem(x)

        # if self.model_params["stem_block"]["dropout"] > 0:

        #     self.adaptive_dropout_stem = AdaptiveDropout(initial_rate=self.model_params["stem_block"]["dropout"],
        #                                                  name=f"adaptive_dropout_stem")
        #     x = self.adaptive_dropout_stem(x)

        x = layers.DepthwiseConv2D(kernel_size=self.model_params["stem_block"]["DWConv_kernel"],
                                   strides=self.model_params["stem_block"]["DWConv_strides"],
                                   padding='same', 
                                   use_bias=False)(x)


        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    def _taku_block(self, inputs:tuple, taku_block_number:int, stage_number:int):

        x = layers.DepthwiseConv2D( kernel_size=self.model_params["stages_block"]["taku_block"]["DWConv_kernel"], 
                                    strides=self.model_params["stages_block"]["taku_block"]["DWConv_strides"], 
                                    padding='same', 
                                    use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        if self.model_params["stages_block"]["taku_block"]["dropout"] > 0:

            adaptiveDropout = AdaptiveDropout(initial_rate=self.model_params["stages_block"]["taku_block"]["dropout"],
                                              name=f"adaptive_dropout_taku_stage{stage_number}_block{taku_block_number}")
            
            self.adaptive_dropout_taku.append(adaptiveDropout)

            x = adaptiveDropout(x)

        return layers.Add()([x, inputs])
    

    
    def _downsampler_block(self, inputs:tuple, curr_stage_number:int):

        input_channels:int = inputs.shape[-1]

        """
        desired_groups, represents the input_channel + output_channel, which match
        Divided by the number of stages we have
        """
        desired_groups:int = math.floor(2 * input_channels / self.model_params["stages_block"]["stages_number"]) 

        groups:int = find_nearest_valid_groups( desired_groups=desired_groups,
                                           input_channels=input_channels)
        
        """
        This Grouped Conv2D, DOES NOT CHANGE the shape if input is (None,1,1,2048) then the output is also (None,1,1,2048), because
            filters = filters

        If the number of stages ( Taken from the Config ) does not divide accurate the filters (filters % num_groups)

        The Convlution becomes a normal convolution ( Conv2D ) as we have :
            groups = 1
        So mix up all channels in one Group

        Kernel size must be 1 to perform a PointWise Convolution

        """
        x = layers.Conv2D(  filters=input_channels, 
                            kernel_size=1, 
                            groups=groups, 
                            use_bias=False,
                            kernel_regularizer=regularizers.l2(self.model_params["stages_block"]["downsampler"]["l2_weight_decay"]))(inputs)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        pool_layer = layers.MaxPooling2D if curr_stage_number < self.model_params["stages_block"]["stages_number"] else layers.AveragePooling2D

        x = pool_layer(pool_size=self.model_params["stages_block"]["downsampler"]["pool_size"], 
                       strides=self.model_params["stages_block"]["downsampler"]["strides"], 
                       padding='same')(x)
        
        return layers.LayerNormalization()(x)
    
    
    def _stage_block(self, inputs, curr_stage_number):
        x = inputs
        for i in range(self.model_params["stages_block"]["taku_block"]["taku_block_number"]):
            x = self._taku_block(inputs=x, taku_block_number=i, stage_number=curr_stage_number)
        concat = layers.Concatenate()([inputs, x])
        return self._downsampler_block(inputs=concat, curr_stage_number=curr_stage_number)
    
    def _refiner_block(self, inputs):

        x = layers.DepthwiseConv2D( kernel_size=self.model_params["refiner_block"]["DWConv_kernel"], 
                                    strides = self.model_params["refiner_block"]["DWConv_strides"], 
                                    padding='same', 
                                    use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)

        dropout_after_dw = AdaptiveDropout(initial_rate=self.model_params["refiner_block"]["dropout"],
                                           name=f"adaptive_dropout_refiner_after_dw")
        
        self.adaptive_dropout_refiner.append(dropout_after_dw)

        x = dropout_after_dw(x)

        x = layers.GlobalAveragePooling2D()(x)

        if self.model_params["refiner_block"]["dropout"] > 0:

            dropout_after_gap = AdaptiveDropout(initial_rate=self.model_params["refiner_block"]["dropout"],
                                                            name=f"adaptive_dropout_refiner_after_gap")
            self.adaptive_dropout_refiner.append(dropout_after_gap)

            x = dropout_after_gap(x)

        return layers.Dense(self.model_params["refiner_block"]["num_output_classes"], 
                            activation='softmax',
                            kernel_regularizer=regularizers.l2(self.model_params["refiner_block"]["l2_weight_decay"]))(x)
    


    
    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self._stem_block(inputs)
        for curr_stage_number in range(self.model_params["stages_block"]["stages_number"]):
            x = self._stage_block(x, curr_stage_number)
        outputs = self._refiner_block(x)
        return Model(inputs, outputs)


    def _count_flops(self, batch_size=1)-> int:
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
    




    def _convert_to_tflite(self,x_train:Optional[tf.Tensor] = None)->None:
        """Converts a trained model to TFLite with full-integer quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # **Enable optimizations and quantization**
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # **Use a representative dataset to optimize quantization**
        """
        You give the converter a small sample of real inputs (x_train[:100]).

        TensorFlow runs the model (silently) with those inputs.

        It records the ranges (min/max) of each activation tensor.

        Then it uses those stats to compute:

        A scale (how many float values each int8 step represents)

        A zero-point (what int8 value maps to 0.0 in float)

        This mapping is then used to quantize the entire model.
        """
        def representative_dataset():
            for i in range(100):
                data:tf.Tensor = tf.cast(x_train[i:i+1], tf.float32)
                yield [data]

        converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset)

        # Force fully int8 quantized kernels
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Set input and output types to uint8
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8


        tflite_model = converter.convert()

        # **Save TFLite model**
        os.makedirs(f'{self.folderName}/TfLiteModels', exist_ok=True)
        tflite_model_path = f"{self.folderName}/TfLiteModels/{self.model_name}.tflite"

        try:
            with open( tflite_model_path, "wb") as f:
                f.write( tflite_model )
            print(f"✅ Model converted and saved as {tflite_model_path}")
        except OSError as e:
            if e.errno == 28:
                print("⚠️ Skipping header file generation: No space left on device.")
            else:
                print(f"❌ Unexpected error while writing header file: {e}")




    def _convert_tflite_to_c_array(self)->None:
        """Converts the TFLite model into a C array header file for Arduino integration."""
        tflite_path = f"{self.folderName}/TfLiteModels/{self.model_name}.tflite"
    
        try:
            with open(tflite_path, "rb") as f:
                tflite_model = f.read()
        except FileNotFoundError:
            print(f"❌ TFLite file not found at {tflite_path}")
            return

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

        os.makedirs(f'{self.folderName}/HeaderFiles', exist_ok=True)
        header_file_path = f"{self.folderName}/HeaderFiles/{self.model_name}.h"

        try:
            with open(header_file_path, "w") as f:
                f.write(header_content)
            print(f"✅ C header file saved as {header_file_path}")
        except OSError as e:
            if e.errno == 28:
                print("⚠️ Skipping header file generation: No space left on device.")
            else:
                print(f"❌ Unexpected error while writing header file: {e}")
    
    
        
    def _evaluate_tflite_model(self,
                               x_test:Optional[tf.Tensor]= None,
                               y_test:Optional[tf.Tensor]= None)-> float:
        
        """Evaluates the TFLite model and returns the accuracy."""
        tflite_path = f"{self.folderName}/TfLiteModels/{self.model_name}.tflite"

        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
        except (OSError, ValueError) as e:
            print(f"⚠️ Could not load TFLite model from {tflite_path}: {e}")
            return -1.0

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Debugging prints
        print("✅ Model loaded successfully!\n")
        print("📌 Input Details:", input_details)
        print("📌 Output Details:", output_details)
        print("Expected Input Shape:", input_details[0]['shape'])
        print("Actual Input Shape: \n", x_test[0].shape)
        print()

        def preprocess_input(input_data):
            """Adjusts input data if the model uses uint8 quantization."""
            if input_details[0]["dtype"] == np.uint8:
                scale, zero_point = input_details[0]["quantization"]
                input_data = np.round(input_data / scale + zero_point).astype(np.uint8)
            return input_data

        y_pred = []
        for i in range(len(x_test)):
            input_data = preprocess_input(x_test[i:i+1])

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
        y_true_classes = np.argmax(y_test, axis=-1)
        accuracy = np.mean(y_pred_classes == y_true_classes)
        return accuracy

    
    def check_trainability(self) -> bool:
        """Check if the model fits within the memory constraints."""
        if self.train_params is None:
            print("⚠️ Cannot check trainability: `train_params` is None.")
            return False

        (self.results.estimatedMaxRam, 
        self.results.estimatedFlash, 
        self.results.AccurateMaxRam,
        self.results.ModelRam)= memoryEstimator.memoryEstimation(model = self.model, data_dtype_multiplier = self.train_params["data_dtype_multiplier"])

        print(f"⚠️ Checking model {self.model_name}.....\n")
        print(f"Max RAM Usage: {self.results.estimatedMaxRam:.2f} KB\n")
        print(f"Parameter Memory: {self.results.estimatedFlash:.2f} KB\n")

        ram_limit = self.train_params["max_ram_consumption"] - self.train_params["additional_ram_consumption"]

        flash_limit = self.train_params["max_flash_consumption"] - self.train_params["additional_flash_consumption"]


        if  self.results.estimatedFlash * 1024 > flash_limit:
            print(f"🚨 Model not trainable: Flash usage ({ self.results.estimatedFlash:.2f} KB) exceeds limit ({flash_limit / 1024:.2f} KB). \n")
            return False
        
        if self.results.ModelRam * 1024 > ram_limit  :
            print(f"🚨 Model not trainable: Flash usage ({ self.results.ModelRam:.2f} KB) exceeds limit ({ram_limit / 1024:.2f} KB). \n")
            return False
        
        return True
    
    
    def train(self,
              x_train:Optional[tf.Tensor]= None,
              y_train:Optional[tf.Tensor]= None,
              x_test:Optional[tf.Tensor]= None,
              y_test:Optional[tf.Tensor]= None):
        
        """Train the model, evaluate metrics, and store results."""
        x_train = x_train if x_train is not None else self.x_train
        y_train = y_train if y_train is not None else self.y_train
        x_test = x_test if x_test is not None else self.x_test
        y_test = y_test if y_test is not None else self.y_test

        if self.check_trainability is False:
            return None

        print("✅ Memory check passed! Starting training... \n")

        # **Compile Model**
        if not self.is_trained:

            optimizer = get_optimizer(self.train_params["optimizer"], 
                                      self.train_params["learning_rate"] if self.learningRate is None else self.learningRate)
            
            self.model.compile( optimizer = optimizer, 
                                loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.train_params["label_smothing"]),
                                metrics = ['accuracy'])

        # **Callbacks**
        checkpoint_path = f'{self.folderName}/saved_models/{self.model_name}.keras'

        checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                     monitor='val_accuracy', 
                                     save_best_only=True, 
                                     mode='max', 
                                     verbose=0,  
                                     save_weights_only=False)
        
        early_stopping_acc = EarlyStopping(monitor='val_accuracy', 
                                           patience=self.train_params["stop_patience"], # We stop the training if for "stop_patience" we have no improvement
                                           mode='max', 
                                           restore_best_weights=True)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                      factor=self.train_params["learning_factor"], 
                                      patience=self.train_params["learning_rate_patience"], 
                                      verbose=1)
        
        midway_callback = MidwayStopCallback(total_epochs=self.train_params["num_epochs"], 
                                             divider=self.train_params["divider"], 
                                             threshold=0.30)
        
        adjust_dropout = AdjustDropoutCallback(model_instance=self,
                                               overfitting_threshold=self.train_params["overfitting"],
                                               factor=self.train_params['incrementFactor'],
                                               max_rate=self.train_params["max_dropout"],
                                               cooldown=3,
                                               total_epochs=self.train_params["num_epochs"],
                                               divider=self.train_params["divider"])

        # **Train Model with Timing**
        start_time = time.time()
        print(f"✅Start training of {self.model_name}\n")

        history = self.model.fit(
            x_train, y_train,
            epochs= self.epochs if self.epochs else self.train_params["num_epochs"],
            batch_size=self.train_params["batch_size"],
            validation_data=(x_test, y_test),
            verbose=2,
            callbacks=[midway_callback, early_stopping_acc, reduce_lr, checkpoint, adjust_dropout]
        )

        training_time = time.time() - start_time

        # **Load Best Model**
        self.model.load_weights(checkpoint_path)
        print(f"✅ Best model restored from {checkpoint_path}\n")

        # **Compute Accuracy Metrics**
        best_test_acc = max(history.history['val_accuracy'])  # test accuracy

        print(f"✅ Best Test Accuracy (Best Model): {best_test_acc:.4f}\n")

        if best_test_acc > 0.50:
            print(f"\n\🚀 Best test accuracy ({best_test_acc:.4f}) exceeded 50%. Continuing training for 100 more epochs.")

            history_extra = self.model.fit(
                x_train, y_train,
                epochs=self.results.epochs_trained + 100,
                initial_epoch=self.results.epochs_trained,
                batch_size=self.train_params["batch_size"],
                validation_data=(x_test, y_test),
                verbose=2,
                callbacks=[midway_callback, early_stopping_acc, reduce_lr, checkpoint, adjust_dropout]
            )

            self.results.epochs_trained += len(history_extra.history['loss'])

            best_test_acc = max(history_extra.history['val_accuracy'])
            self.results.test_accuracy = best_test_acc
            self.results.train_accuracy = max(history_extra.history['accuracy'])

            print(f"\n🔁 Continued Training Complete. New Best Test Accuracy: {best_test_acc:.4f}\n")
        
        self.model.load_weights(checkpoint_path)
        print(f"✅ Final Best model restored from {checkpoint_path}\n")

        # **Predictions & Metrics**
        y_test_pred = self.model.predict(x_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        self.results.history = history
        self.results.epochs_trained = len(history.history['loss'])
        self.results.train_accuracy = max(history.history['accuracy']) 
        self.results.test_accuracy = best_test_acc
        self.results.precision = precision_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.recall = recall_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.f1_score = f1_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.training_time = training_time 

        # ** Declare that this model is trained.
        self.is_trained = True

        # **Save Model in Multiple Formats**
        self._convert_to_tflite(x_train=x_train)
        self._convert_tflite_to_c_array()
        self.results.flops = self._count_flops()
        print(f"📊 Estimated FLOPs: {self.results.flops:,}")

        # **Evaluate the TFLite Model**
        tflite_acc = self._evaluate_tflite_model(x_test=x_test,
                                                 y_test=y_test)
        self.results.tflite_accuracy = tflite_acc
        print(f"Test Accuracy (TFLite): {tflite_acc:.4f}")

        # **File Size Reporting**
        keras_size_kb = os.path.getsize(checkpoint_path) / 1024
        tflite_size_kb = os.path.getsize(f"{self.folderName}/TfLiteModels/{self.model_name}.tflite") / 1024
        c_array_size_kb = os.path.getsize(f"{self.folderName}/HeaderFiles/{self.model_name}.h") / 1024

        self.results.tflite_size = tflite_size_kb
        
        print(f"Keras Model Size: {keras_size_kb:.2f} KB")
        print(f"TFLite Model Size: {tflite_size_kb:.2f} KB")
        print(f"C Array File Size: {c_array_size_kb:.2f} KB")


        print("\n✅ Training complete. Best model and metrics stored in `self.results`.\n")
    

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


def find_nearest_valid_groups(desired_groups:int, input_channels:int) -> int:
    """
    Given the desired number and the input channels
    Try to find the closet integer number ( With priority given to the smallest number )
    Which will perfectly divide the inpu_channe; ( input_channels % candidate ) 
    """
    for offset in range(0, desired_groups):
        for candidate in (desired_groups - offset, desired_groups + offset):
            if candidate > 0 and input_channels % candidate == 0:
                return candidate
    return 1

# Helper Classes




class AdaptiveDropout(tf.keras.layers.Layer):
    def __init__(self, initial_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_rate = initial_rate
        self.rate = tf.Variable(initial_value=initial_rate, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=False):
        return tf.nn.dropout(inputs, rate=self.rate) if training else inputs


class AdjustDropoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_instance:TakuNetModel, overfitting_threshold:float=0.1, factor:float=1.2, max_rate:float=0.5,
                 cooldown:int=3, total_epochs:int=50, divider:int = 5):
        super().__init__()
        self.model_instance = model_instance
        self.overfitting_threshold = overfitting_threshold
        self.factor = factor
        self.max_rate = max_rate
        self.cooldown = cooldown  # Number of epochs to wait after adjusting
        self.apply_after_epoch = total_epochs // divider  # Ignore overfitting detection before this epoch
        self.cooldown_counter = 0  # Internal counter

    def on_epoch_end(self, epoch, logs=None):
        # If still warming up, skip
        if epoch < self.apply_after_epoch :
            return

        # If still in cooldown after last adjustment, skip
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        if train_acc is None or val_acc is None:
            return

        gap = train_acc - val_acc

        if gap > self.overfitting_threshold:
            print(f"\n⚠️ Overfitting detected! Train Acc - Val Acc = {gap:.3f} > {self.overfitting_threshold}")
            self._increase_one_dropout()
            self.cooldown_counter = self.cooldown  # Reset cooldown after adjusting

    def _increase_one_dropout(self):
        dropout_layers:List[AdaptiveDropout] = []

        if self.model_instance.adaptive_dropout_stem is not None:
            dropout_layers.append(self.model_instance.adaptive_dropout_stem)

        if self.model_instance.adaptive_dropout_taku is not None:
            dropout_layers.extend([d for d in self.model_instance.adaptive_dropout_taku if d is not None])

        if self.model_instance.adaptive_dropout_refiner is not None:
            dropout_layers.extend([d for d in self.model_instance.adaptive_dropout_refiner if d is not None])

        if not dropout_layers:
            print("⚠️ No AdaptiveDropout layers found to adjust.")
            return

        chosen_layer:AdaptiveDropout = random.choice(dropout_layers)
        new_rate = min(self.factor * float(chosen_layer.rate.numpy()), self.max_rate)
        chosen_layer.rate.assign(new_rate)
        print(f"🔧 {chosen_layer.name}: dropout rate increased to {new_rate:.3f}")




class MidwayStopCallback(Callback):
    def __init__(self, total_epochs:int, divider:int, threshold:float):
        super().__init__()
        self.mid_epoch = total_epochs // divider 
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.mid_epoch:
            train_acc = logs.get('accuracy')
            val_acc = logs.get('val_accuracy')
            print(f"\nMidway Epoch {epoch}: Training Acc = {train_acc}, Validation Acc = {val_acc}")
            if val_acc < self.threshold:  
                print(f"\n🚨 Stopping early: Training accuracy is below {self.threshold} at epoch {epoch}")
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
        self.estimatedMaxRam = None
        self.AccurateMaxRam = None
        self.ModelRam = None
        self.estimatedFlash = None
        self.training_time = None
        self.fitness_score = None
        self.tflite_accuracy = None
        self.tflite_size = None
        self.epochs_trained = None
        self.flops = None

    def __repr__(self):
        return (f"TrainingResults(\n"
                f"  Train Accuracy: {self.train_accuracy:.4f}\n"
                f"  Test Accuracy: {self.test_accuracy:.4f}\n"
                f"  TFlite Test Accuracy: {self.tflite_accuracy:.4f}\n"
                f"  Precision: {self.precision:.4f}\n"
                f"  Recall: {self.recall:.4f}\n"
                f"  F1 Score: {self.f1_score:.4f}\n"
                f"  Estimated Max Ram Use: {self.estimatedMaxRam:.4f}\n"
                f"  Analyzed Max Ram Use: {self.AccurateMaxRam:.4f}\n"
                f"  Estimated Flash Memory Use: {self.estimatedFlash:.4f}\n"
                f"  TFlite Memory Use: {self.tflite_size:.4f}\n"
                f"  Training Time: {self.training_time}\n)"
                f"  FLOPs: {self.flops:,}\n)")  
