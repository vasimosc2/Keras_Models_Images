import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from utils import memoryEstimator
import math
import random
from SurrogateComparisson.Embedding import simple_architecture_embedding
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
        self.embedded: Union[np.ndarray,None] = simple_architecture_embedding(model_params) if model_params else None
        self.folderName:str = folder if folder is not None else "."
        self.epochs:int = None
        self.learningRate:Optional[float] = 0.0005 if given_model else None
        self.results: TrainingResults = TrainingResults()
        self.test:bool = False
        self.is_trainable: bool = self.check_trainability() if self.test is False else True

  

    
    def _stem_block(self, inputs:tuple):
        """
        The input shape is: (None,32,32,3) (Given input 32,32,3)
        The output shape is: (None, 32 / (Conv_strides * DWConv_stride), 32 / (Conv_strides * DWConv_stride), filters)
        """

        x = layers.Conv2D(filters=self.model_params["stem_block"]["filters"], 
                          kernel_size=self.model_params["stem_block"]["Conv_kernel"],
                          strides=self.model_params["stem_block"]["Conv_strides"], 
                          padding='same', 
                          use_bias=False)(inputs)
        
        x = layers.BatchNormalization()(x)

        x = layers.ReLU(6.0)(x)

        self.adaptive_dropout_stem = AdaptiveDropout(initial_rate=0.0, 
                                                     name="adaptive_dropout_stem")
        x = self.adaptive_dropout_stem(x)

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

        x = layers.Conv2D(filters=inputs.shape[-1],
                          kernel_size=1,
                          padding='same',
                          use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        adaptiveDropout = AdaptiveDropout(initial_rate=0.0,
                                          name=f"adaptive_dropout_taku_stage{stage_number}_block{taku_block_number}")
            
        self.adaptive_dropout_taku.append(adaptiveDropout)

        x = adaptiveDropout(x)

        return layers.Add()([x, inputs])
    
    def _se_block(self, inputs, ratio=8):
        """Squeeze-and-Excitation block."""
        filters = inputs.shape[-1]
        se = layers.GlobalAveragePooling2D()(inputs)
        se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
        se = layers.Reshape((1,1,filters))(se)
        return layers.multiply([inputs, se])
    
    def _downsampler_block(self, inputs:tuple, curr_stage_number:int):

        input_channels:int = inputs.shape[-1]

        """
        desired_groups, represents the input_channel + output_channel, 
        Input_Channel, is the output of the first DownSampler
        Output_Channel is the output of the Last TakuBlock
        BUT
        When we use the Concat layer:
        input:  (batch, height, width, C1)
        output:  (batch, height, width, C1)
        concat will be shape (batch, height, width, C1 + C2), which is what goes inside the DownSampler
        """
        desired_groups:int = math.floor(input_channels / self.model_params["stages_block"]["stages_number"]) 

        groups:int = find_nearest_valid_groups(desired_groups=desired_groups,
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
                            use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        pool_layer = layers.MaxPooling2D if curr_stage_number < self.model_params["stages_block"]["stages_number"] else layers.AveragePooling2D

        x = pool_layer(pool_size=self.model_params["stages_block"]["downsampler"]["pool_size"], 
                       strides=self.model_params["stages_block"]["downsampler"]["strides"], 
                       padding='same')(x)
        
        x = self._se_block(x, ratio=8)
        
        return layers.LayerNormalization()(x)
    
    
    def _stage_block(self, inputs, curr_stage_number):
        x = inputs
        for i in range(self.model_params["stages_block"]["taku_block"]["taku_block_number"]):
            x = self._taku_block(inputs=x, taku_block_number=i, stage_number=curr_stage_number)

        x = layers.Add()([x, inputs])
        concat = layers.Concatenate()([inputs, x])
        return self._downsampler_block(inputs=concat, curr_stage_number=curr_stage_number)
    
    def _refiner_block(self, inputs):

        x = layers.DepthwiseConv2D( kernel_size=self.model_params["refiner_block"]["DWConv_kernel"], 
                                    strides = self.model_params["refiner_block"]["DWConv_strides"], 
                                    padding='same', 
                                    use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)

        dropout_after_dw = AdaptiveDropout(initial_rate=0.0,
                                           name=f"adaptive_dropout_refiner1_after_dw")
        
        self.adaptive_dropout_refiner.append(dropout_after_dw)

        x = dropout_after_dw(x)

        x = layers.GlobalAveragePooling2D()(x)

        dropout_after_gap = AdaptiveDropout(initial_rate = 0.0,
                                            name=f"adaptive_dropout_refiner2_after_gap")
        
        self.adaptive_dropout_refiner.append(dropout_after_gap)

        x = dropout_after_gap(x)

        return layers.Dense(self.model_params["refiner_block"]["num_output_classes"], 
                            activation='softmax')(x)
    
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

            optimizer = get_optimizer(name=self.train_params["optimizer"], 
                                      learning_rate=self.train_params["learning_rate"] if self.learningRate is None else self.learningRate)
            
            self.model.compile( optimizer = optimizer, 
                                loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.train_params["label_smothing"]),
                                metrics = ['accuracy'])
        """
        Label smoothing: [0,0,1,0,0] -> [a/(C-1), a/(C-1), 1-a, a/(C-1), a/(C-1)] = [0.025, 0.025, 0.9, 0.025, 0.025] ,
                        where C is the number of Classes and a = label_smoothing
                        If C is big , it might makes sense a to be also bigger to cause some significant generalaization
        
        """
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

        midway_callback = MidwayStopCallback(total_epochs=self.train_params["num_epochs"], 
                                             divider=self.train_params["divider"], 
                                             threshold=0.30)
        
        learning_rate_callback = SmartLearningRateScheduler(manual_threshold=2e-3,
                                                            manual_factor=0.5,
                                                            manual_start_epoch=10,
                                                            smart_factor=0.5,
                                                            smart_patience=8,
                                                            smart_min_delta=4e-2,
                                                            smart_min_lr=1e-4,
                                                            smart_start_epoch=15,
                                                            verbose=True)
        
        adjust_dropout = AdjustDropoutCallback(model_instance=self,
                                               overfitting_threshold=self.train_params["overfitting"],
                                               cooldown=3,
                                               start_dropout_epoch=20)

        # **Train Model with Timing**
        start_time = time.time()
        print(f"✅Start training of {self.model_name}\n")

        history = self.model.fit(
            x_train, y_train,
            epochs = self.epochs if self.epochs else self.train_params["num_epochs"],
            batch_size=self.train_params["batch_size"],
            validation_data=(x_test, y_test),
            verbose=2,
            callbacks=[midway_callback, early_stopping_acc, checkpoint, adjust_dropout, learning_rate_callback]
        )

        training_time = time.time() - start_time

        # **Load Best Model**
        self.model.load_weights(checkpoint_path)
        print(f"✅ Best model restored from {checkpoint_path}\n")

        # **Compute Accuracy Metrics**
        best_test_acc = max(history.history['val_accuracy'])  # test accuracy

        print(f"✅ Best Test Accuracy (Best Model): {best_test_acc:.4f}\n")

        # Initialize full history and epoch counter
        full_history = history
        total_epochs_trained = len(history.history['loss'])

        # **Check if we should continue training**
        if best_test_acc > 0.50:
            print(f"\n🚀 Best test accuracy ({best_test_acc:.4f}) exceeded 50%. Continuing training for 100 more epochs.\n")
            
            already_used_epochs = self.epochs if self.epochs else self.train_params["num_epochs"]
            
            # Extra Training Phase
            history_extra = self.model.fit(
                x_train, y_train,
                epochs=already_used_epochs + 100,
                initial_epoch=already_used_epochs,
                batch_size=self.train_params["batch_size"],
                validation_data=(x_test, y_test),
                verbose=2,
                callbacks=[midway_callback, early_stopping_acc, checkpoint, adjust_dropout, learning_rate_callback]
            )
            
            # Update best accuracy after extra training
            best_test_acc = max(history_extra.history['val_accuracy'])

            # Merge histories
            for key in full_history.history.keys():
                full_history.history[key].extend(history_extra.history[key])

            total_epochs_trained += len(history_extra.history['loss'])

            print(f"\n🔁 Continued Training Complete. New Best Test Accuracy: {best_test_acc:.4f}\n")

        # **Load final Best Model**
        self.model.load_weights(checkpoint_path)
        print(f"✅ Final Best model restored from {checkpoint_path}\n")

        # **Predictions & Metrics**
        y_test_pred = self.model.predict(x_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # **Save Results**
        self.results.history = full_history
        self.results.epochs_trained = total_epochs_trained
        self.results.train_accuracy = max(full_history.history['accuracy'])
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
        self.addtion:float = 0.0

    def call(self, inputs, training=False):
        """

        Use noise to emulate the SpatialDropout2D 
        Use None to have the original Dropout

        """
        if training:
            input_shape = tf.shape(inputs)
            input_rank = inputs.shape.rank  # static rank if possible

            if input_rank == 4:
                # (batch, height, width, channels) -> Spatial Dropout
                noise_shape = (input_shape[0], 1, 1, input_shape[-1])
            elif input_rank == 2:
                # (batch, features) -> Normal Dropout
                noise_shape = (input_shape[0], input_shape[1])
            else:
                raise ValueError(f"Unsupported input rank {input_rank} for AdaptiveDropout")

            return tf.nn.dropout(inputs, rate=self.rate, noise_shape=None)
        else:
            return inputs





class AdjustDropoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_instance:TakuNetModel, overfitting_threshold:float=0.1,
                 cooldown:int=3, start_dropout_epoch:int=15):
        super().__init__()
        self.model_instance = model_instance
        self.overfitting_threshold = overfitting_threshold
        self.cooldown = cooldown
        self.start_dropout_epoch = start_dropout_epoch
        self.cooldown_counter = 0
        self.dropout_initialized = False

    def on_epoch_end(self, epoch, logs=None):
        # 🔵 Step 1: Initialize Dropout after a specific epoch
        if not self.dropout_initialized and epoch >= self.start_dropout_epoch:
            print(f"\n🚀 Initializing Dropout rates at Epoch {epoch}")
            self._initialize_dropout_rates()
            self.dropout_initialized = True

        # If still warming up for overfitting detection, skip
        if epoch < self.start_dropout_epoch:
            return

        # If still in cooldown after last adjustment, skip
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # 🔵 Step 2: Normal overfitting detection
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        if train_acc is None or val_acc is None:
            return

        gap = train_acc - val_acc

        if gap > self.overfitting_threshold:
            print(f"\n⚠️ Overfitting detected! Train Acc - Val Acc = {gap:.3f} > {self.overfitting_threshold}")
            self._increase_one_dropout()
            self.cooldown_counter = self.cooldown  # Reset cooldown after adjusting

    def _initialize_dropout_rates(self):
        dropout_layers: List[AdaptiveDropout] = []

        if isinstance(self.model_instance.adaptive_dropout_stem, AdaptiveDropout) :
            dropout_layers.append(self.model_instance.adaptive_dropout_stem)

        if self.model_instance.adaptive_dropout_taku is not None:
            dropout_layers.extend([d for d in self.model_instance.adaptive_dropout_taku if isinstance(d, AdaptiveDropout)])

        if self.model_instance.adaptive_dropout_refiner is not None:
            dropout_layers.extend([d for d in self.model_instance.adaptive_dropout_refiner if isinstance(d, AdaptiveDropout)])

        for layer in dropout_layers:
            if "stem" in layer.name:
                initial_rate = 0.02
                layer.addtion = 0.02
                layer.max_rate = 0.15
            elif "taku" in layer.name:
                initial_rate = 0.03
                layer.addtion = 0.03
                layer.max_rate = 0.4
            elif "refiner1" in layer.name:
                initial_rate = 0.05
                layer.addtion = 0.03
                layer.max_rate = 0.4
            elif "refiner2" in layer.name:
                initial_rate = 0.1
                layer.addtion = 0.03
                layer.max_rate = 0.5
            else:
                initial_rate = 0.05
                layer.addtion = 0.03
                layer.max_rate = 0.3

            layer.rate.assign(initial_rate)
            print(f"🔧 {layer.name}: initialized dropout rate to {initial_rate:.3f} (max {layer.max_rate:.3f})")



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
        old_rate = float(chosen_layer.rate.numpy())
        new_rate = max(0.05, min(old_rate + chosen_layer.addtion, chosen_layer.max_rate)) 
        chosen_layer.rate.assign(new_rate)
        print(f"🔧 {chosen_layer.name}: dropout rate increased from {old_rate:.3f} to {new_rate:.3f}")





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


class SmartLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, 
                 manual_threshold=2e-3, 
                 manual_factor=0.5, 
                 manual_start_epoch=10,
                 smart_factor=0.5,
                 smart_patience=8,
                 smart_min_delta=4e-2,
                 smart_min_lr=1e-4,
                 smart_start_epoch=15,
                 verbose=True):
        """
        Combines Manual LR scheduling and Smart ReduceLROnPlateau into one callback.
        
        manual_threshold: threshold for manual decay
        manual_factor: factor for manual decay
        manual_start_epoch: when to start manual decay
        smart_factor: factor for smart decay
        smart_patience: patience for smart decay
        smart_min_delta: min improvement for smart decay
        smart_min_lr: minimum learning rate allowed
        smart_start_epoch: when to activate smart decay
        verbose: print messages
        """
        super().__init__()
        self.manual_threshold = manual_threshold
        self.manual_factor = manual_factor
        self.manual_start_epoch = manual_start_epoch

        self.smart_factor = smart_factor
        self.smart_patience = smart_patience
        self.smart_min_delta = smart_min_delta
        self.smart_min_lr = smart_min_lr
        self.smart_start_epoch = smart_start_epoch

        self.verbose = verbose
        self.best_val_acc = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self._get_current_lr()
        current_val_acc = logs.get('val_accuracy')

        # --- Manual LR Scheduling ---
        epoch_decades = epoch // self.manual_start_epoch
        if epoch_decades >= 1 and epoch_decades <= 2:
            if current_lr >= self.manual_threshold / epoch_decades:
                new_lr = current_lr * self.manual_factor
                self._set_current_lr(new_lr)
                if self.verbose:
                    print(f"\n🔧 [Manual LR Scheduler] Epoch {epoch}: LR adjusted from {current_lr:.6f} → {new_lr:.6f}")

        # --- Smart Reduce on Plateau ---
        if epoch >= self.smart_start_epoch:
            if current_val_acc is not None:
                if current_val_acc > self.best_val_acc + self.smart_min_delta:
                    self.best_val_acc = current_val_acc
                    self.wait = 0  # Reset wait counter
                else:
                    self.wait += 1
                    if self.wait >= self.smart_patience:
                        if current_lr > self.smart_min_lr:
                            new_lr = max(current_lr * self.smart_factor, self.smart_min_lr)
                            self._set_current_lr(new_lr)
                            if self.verbose:
                                print(f"\n🔻 [SmartReduceLROnPlateau] Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                        self.wait = 0  # Reset wait counter

    def _get_current_lr(self):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            return float(tf.keras.backend.get_value(lr))
        else:
            return float(lr)

    def _set_current_lr(self, new_lr):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'assign'):
            lr.assign(new_lr)
        else:
            self.model.optimizer.learning_rate = new_lr


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
