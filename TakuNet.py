import json
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from Models.SAM import SAMModel
from utils import memoryEstimator
import math
import random
from keras.saving import register_keras_serializable

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
                epochs:Optional[int] = None,
                given_model:Optional[tf.keras.Model] = None,
                enable_dropout: bool = True,
                hardwareConstrains:Optional[bool] = True,
                performaceStoppage:Optional[bool] = False,
                early_stopping_acc:Optional[bool] = False,
                midway_callback:Optional[bool] = False,
                lr_schedule_strategy:Optional[str] = "cosine"
                ):
        
        self.model_name:str = model_name
        self.input_shape: Tuple[int, int, int] = input_shape
        self.model_params: Optional[Dict] = model_params
        self.train_params: Optional[Dict] = train_params
        self.enable_dropout: bool = enable_dropout
        
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
        self.epochs:int = epochs if epochs else self.train_params["num_epochs"]
        self.learningRate:Optional[float] = 0.0005 if given_model else None
        self.results: TrainingResults = TrainingResults()
        self.modelType:str = "normal" #"SAM"
        """
        If self.hardwareConstrains is activated (True), then we take into consideration the resources of Arduino Nano 33 BLU
        If it false, then all the models are trainable and we don't care about how much memory the consume
        """
        self.hardwareConstrains:bool = hardwareConstrains
        self.performaceStoppage:bool = performaceStoppage
        self.early_stopping_acc:bool = early_stopping_acc
        self.midway_callback:bool = midway_callback
        self.lr_schedule_strategy = lr_schedule_strategy.lower()
        self.is_trainable: bool = self.check_trainability( stopBigModels=self.hardwareConstrains )  # if HardwareConstrains is True, the this means we will stop big Models 
                                                                                                    # if it is False we will allow models to be created up to the size of the env.sh variables

  
    def _norm_relu6_block(self,x: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:
        """
        Applies BatchNormalization followed by ReLU6 activation.

        Args:
            x (tf.Tensor): Input tensor.
            name (Optional[str]): Optional name prefix for layers.

        Returns:
            tf.Tensor: Output tensor after normalization and activation.
        """
        
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0,name=f"ReLu_{name}")(x)
        return x

    
    def _stem_block(self, inputs:tuple):
        """
        The input shape is: (None,32,32,3) (Given input 32,32,3)
        The output shape is: (None, 32 / (Conv_strides * DWConv_stride), 32 / (Conv_strides * DWConv_stride), filters)
        """

        x = layers.Conv2D(filters=self.model_params["stem_block"]["filters"], 
                          kernel_size=self.model_params["stem_block"]["Conv_kernel"],
                          strides=self.model_params["stem_block"]["Conv_strides"], 
                          padding='same',
                          name="InitialConv",
                          use_bias=False)(inputs)
        
        x = self._norm_relu6_block(x=x, name="stem1")

        self.adaptive_dropout_stem = AdaptiveDropout(initial_rate=0.0,
                                                     enabled= self.enable_dropout,
                                                     name="adaptive_dropout_stem")
        
        x = self.adaptive_dropout_stem(x)
        
        # This part is just like 1 Extra Taku_Block and I dont think it needed

        x = layers.DepthwiseConv2D(kernel_size=self.model_params["stem_block"]["DWConv_kernel"],
                                   strides=self.model_params["stem_block"]["DWConv_strides"],
                                   padding='same', 
                                   use_bias=False)(x)


        x = self._norm_relu6_block(x, name="stem2")
        return x
    
    def _taku_block(self, inputs:tuple, taku_block_number:int, stage_number:int):

        x = layers.DepthwiseConv2D( kernel_size=self.model_params["stages_block"]["taku_block"]["DWConv_kernel"], 
                                    strides=self.model_params["stages_block"]["taku_block"]["DWConv_strides"], 
                                    padding='same',
                                    name=f"DepthWiseConV_TakuStage{stage_number}_DepthWise_Block{taku_block_number}",
                                    use_bias=False)(inputs)

        x = self._norm_relu6_block(x, name=f"Norm_TakuStage{stage_number}_DepthWise_Block{taku_block_number}")
        """
        This is PointWise Conv, it is used in BiblioGraphy after the DepthWiseConv2D,
        But in this case we "collect" all the DeptWise into one PointWise in the DownSampler
        Better Performance but More Flash Consumption, 0 RAM consumption

        """

        x = layers.Conv2D(filters=inputs.shape[-1],
                          kernel_size=1,
                          padding='same',
                          use_bias=False)(x)

        x = self._norm_relu6_block(x=x, name=f"Norm_TakuStage{stage_number}_PointWise_Block{taku_block_number}")


        adaptiveDropout = AdaptiveDropout(initial_rate=0.0,
                                          enabled= self.enable_dropout,
                                          name=f"adaptive_dropout_taku_stage{stage_number}_block{taku_block_number}")
            
        self.adaptive_dropout_taku.append(adaptiveDropout)

        x = adaptiveDropout(x)

        return layers.Add(name=f"TakuBlock_SkipConnection_stage{stage_number}_block{taku_block_number}")([x, inputs]) # This is the SKIP-Connection
    

    
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
                            name=f"GroupedPointWiseConv{curr_stage_number}",
                            use_bias=False)(inputs)

        x = self._norm_relu6_block(x,name=f"GroupedPointWiseConv{curr_stage_number}")

        pool_layer = layers.MaxPooling2D if curr_stage_number < self.model_params["stages_block"]["stages_number"] else layers.AveragePooling2D

        """
        MaxPooling2D:   It selects the maximum value from small local
                        regions (typically 2 × 2 patches), effectively halving both the height and width of
                        the feature map when using a stride of 2. This results in a 75% reduction in the
                        number of spatial elements (e.g., (32 × 32) → (16 × 16) )

        AveragePooling2D:
                        It does exactly the same, but it does not return the maximum value, rather it returns the average number
                        It diminishes the influence of outliers and noisy activations while retaining a more balanced view of the learned features
        """
        x = pool_layer(pool_size=self.model_params["stages_block"]["downsampler"]["pool_size"], 
                       strides=self.model_params["stages_block"]["downsampler"]["strides"], 
                       padding='same')(x)
        
        #x = self._se_block(x, ratio=8)
        
        return layers.LayerNormalization()(x)
    

    def _se_block(self, inputs, ratio=8):
        """Squeeze-and-Excitation block."""
        """
        Good in Theory -> Does not seem to produce Better Results
        
        """
        filters = inputs.shape[-1]
        """
        If the input here is (None,32,32,40)
        """
        se = layers.GlobalAveragePooling2D()(inputs) # For each of filters, it computes the average -> Output 40

        se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se) # 

        se = layers.Dense(filters, activation='hard_sigmoid', use_bias=False)(se)

        se = StaticReshape((1, 1, filters))(se)
        return layers.multiply([inputs, se])
    
    
    def _stage_block(self, inputs, curr_stage_number):
        x = inputs
        for i in range(self.model_params["stages_block"]["taku_block"]["taku_block_number"]):
            x = self._taku_block(inputs=x, taku_block_number=i, stage_number=curr_stage_number)

        #x = layers.Add()([x, inputs]) I dont think I need a Skip-Connection here between the Input and the Last TakuBlock
        concat = layers.Concatenate(name=f"concat_stage{curr_stage_number}")([inputs, x])

        return self._downsampler_block(inputs=concat, curr_stage_number=curr_stage_number)
    
    def _refiner_block(self, inputs):

        x = layers.DepthwiseConv2D( kernel_size=self.model_params["refiner_block"]["DWConv_kernel"], 
                                    strides = self.model_params["refiner_block"]["DWConv_strides"], 
                                    padding='same',
                                    name=f"Rediner_DepthWiseConv",
                                    use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)

        dropout_after_dw = AdaptiveDropout(initial_rate=0.0,
                                           enabled= self.enable_dropout,
                                           name=f"adaptive_dropout_refiner1_after_dw")
        
        self.adaptive_dropout_refiner.append(dropout_after_dw)
        
        x = dropout_after_dw(x)

        # ✅ Add a Pointwise Convolution (1x1) to combine channel information
        x = layers.Conv2D(filters=x.shape[-1], 
                          kernel_size=1, 
                          padding='same',
                          name=f"Refiner_PointWiseConv",
                          use_bias=False)(x)
        x = self._norm_relu6_block(x,name="Refiner_PointWiseConv")


        x = layers.GlobalAveragePooling2D()(x)

        dropout_after_gap = AdaptiveDropout(initial_rate = 0.0,
                                            enabled= self.enable_dropout,
                                            name=f"adaptive_dropout_refiner2_after_gap")
        
        self.adaptive_dropout_refiner.append(dropout_after_gap)

        x = dropout_after_gap(x)

        return layers.Dense(self.model_params["refiner_block"]["num_output_classes"],
                            name=f"Classification",
                            activation='softmax')(x)
    
    def _freeze_dropout_for_inference(self, rate_override: float = None):
        """
        Replace all AdaptiveDropout layers with fixed Dropout(rate) layers for TFLite export.
        Optionally override the final dropout rate.
        """
        def convert_layer(layer):
            if isinstance(layer, AdaptiveDropout):
                final_rate = float(rate_override) if rate_override is not None else float(layer.rate.numpy())
                return tf.keras.layers.Dropout(rate=final_rate, name=layer.name + "_frozen")
            return layer

        # Create a new model with dropout layers replaced
        frozen_model = tf.keras.models.clone_model(self.model, clone_function=convert_layer)
        frozen_model.set_weights(self.model.get_weights())  # Preserve learned weights
        self.model = frozen_model

    
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
        model_to_convert = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)

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

    
    def check_trainability(self, stopBigModels:bool = True ) -> bool:
        """
        
        If the Hardware Contrains is turn into False, we train bigger models and we dont convert them into .tflite to add them into the Arduino
        But for GENERAL purposes and in order to allow the training to be complete we want to Dont allow huge models

        """
        print(f"⚠️ Checking model {self.model_name}.....\n")
        """Check if the model fits within the memory constraints."""
        if self.train_params is None:
            print("⚠️ Cannot check trainability: `train_params` is None.")
            return False

        (self.results.estimatedFlash,self.results.ModelRam)= memoryEstimator.memoryEstimation(model = self.model, data_dtype_multiplier = self.train_params["data_dtype_multiplier"])

        
        print(f"Max RAM Usage: {self.results.ModelRam:.2f} KB\n")
        print(f"Parameter Memory: {self.results.estimatedFlash:.2f} KB\n")

        ram_limit = (
            self.train_params["max_ram_consumption"] - self.train_params["additional_ram_consumption"]
            if stopBigModels
            else float(os.getenv("TAKUNET_RAM_LIMIT_MB", "0")) * 1024 * 1024
        )

        flash_limit = (
                self.train_params["max_flash_consumption"] - self.train_params["additional_flash_consumption"]
                if stopBigModels
                else float(os.getenv("TAKUNET_FLASH_LIMIT_MB", "0")) * 1024 * 1024
            )


        if  self.results.estimatedFlash * 1024 > flash_limit:
            print(f"🚨 Model not trainable: Flash usage ({ self.results.estimatedFlash:.2f} KB) exceeds limit ({flash_limit / 1024:.2f} KB). \n")
            return False
        
        if self.results.ModelRam * 1024 > ram_limit  :
            print(f"🚨 Model not trainable: Flash usage ({ self.results.ModelRam:.2f} KB) exceeds limit ({ram_limit / 1024:.2f} KB). \n")
            return False
        print(f"✅ Model: {self.model_name} is Trainable as it fits inside Arduino\n")
        return True
    
    


    def _fitness(self) -> float:

        MAX_RAM:int = (self.train_params["max_ram_consumption"] - self.train_params["additional_ram_consumption"])/1024
        MAX_FLASH:int = (self.train_params["max_flash_consumption"] - self.train_params["additional_flash_consumption"])/1024

        acc:int = self.results.test_accuracy or 0.0
        ram:int = self.results.ModelRam or MAX_RAM
        flash:int = self.results.estimatedFlash or MAX_FLASH

        #if(ram > MAX_RAM or flash > MAX_FLASH):
            # For fair resoning (if the check_Training filter is deActivated)
            # We want to see if the GA, will keep creating models that are untrainable
            # So we penetalize models which are outside of the boundaries heavily
            #return -1000

        # Normalized scores (higher is better)
        norm_ram_score:float = float(max(0.0, 1.0 - ram / MAX_RAM))
        norm_flash_score:float = float(max(0.0, 1.0 - flash / MAX_FLASH))

        # Weight factors (adjust to preference)
        w_acc = 1
        w_ram = 0.001
        w_flash = 0.001

        return w_acc * acc #+ w_ram * norm_ram_score + w_flash * norm_flash_score




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

        if self.is_trainable is False:
            return None
        
        if self.hardwareConstrains is False:
            print("⚠️ We don't Care about the Memory, and the training will be executed normally ... \n")
            """
            Here we dont care about the memory, but we still estimate to see, how big the model is
            """
            self.results.estimatedFlash,self.results.ModelRam= memoryEstimator.memoryEstimation(model = self.model, data_dtype_multiplier = self.train_params["data_dtype_multiplier"])
        else:
            print("✅ Memory check passed! Starting training ... \n")

        # **Compile Model**
        if not self.is_trained:

             # === Hyperparameters ===
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.train_params["label_smothing"])
            batchSize:int = max(8, int(self.train_params["batch_size"] / 2))
            initial_lr:float = 0.05 if self.model_params["optimizer"].lower() == "sgd" else 0.01
            print(f"The initial Learning rate is {initial_lr}\n")

            steps_per_epoch = len(x_train) // batchSize # This the integer part of the diviation
            total_steps = self.epochs * steps_per_epoch
            warmup_epochs = 5
            print(f"The steps per epoch are {steps_per_epoch}\n")

            def cosine_annealing_with_warmup(epoch)->float:
                if epoch < warmup_epochs:
                    return float(initial_lr * (epoch + 1) / warmup_epochs)
                else:
                    cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (self.epochs - warmup_epochs)))
                    return float(initial_lr * cosine_decay)



            optimizer = get_optimizer(name=self.model_params["optimizer"], 
                                      learning_rate=initial_lr)
            
            print(f"The selected optimaizer if {optimizer}\n")

            if self.modelType.lower() == "sam":
                sam_model = SAMModel(self.model)
                sam_model.compile(optimizer=optimizer, 
                                loss=loss, 
                                metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
                self.model = sam_model
            else:
                self.model.compile( optimizer = optimizer, 
                                    loss = loss,
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
                                     verbose=1,  
                                     save_weights_only=False)
        
        

        midway_callback = MidwayStopCallback(stopEpoch = 3, 
                                             threshold = 0.37 )
        
        adjust_dropout = AdjustDropoutCallback(model_instance=self,
                                               overfitting_threshold=self.train_params["overfitting"],
                                               cooldown=3)
        
        # Early_Stopping_acc +  performanceCallback = PerformanceOptimization

        early_stopping_acc = EarlyStopping(monitor='val_accuracy', 
                                           patience= 5,
                                           mode='max', 
                                           restore_best_weights=True)
        
        performanceCallback = PerformanceStopping(patience = 0.3 * self.epochs,
                                                  min_improvement = 0.07)

        swa_callback = SWACallback(self.model, swa_start=10)

        if self.lr_schedule_strategy == "cosine":
            print("📉 Using Cosine Decay LR schedule")
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)

        elif self.lr_schedule_strategy == "linear":
            print("📉 Using Linear Decay LR schedule")
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(LinearDecay(initial_lr=initial_lr, total_steps=total_steps, total_epochs=self.epochs), verbose=1)


        elif self.lr_schedule_strategy == "step":
            
            def step_decay(epoch):
                drop = 0.5
                epochs_drop = 10
                return initial_lr * (drop ** (epoch // epochs_drop))
            
            print("📉 Using Step Decay LR schedule")
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(step_decay)
        else:
            raise ValueError(f"Unknown learning rate schedule strategy: {self.lr_schedule_strategy}")




        # **Train Model with Timing**
        start_time = time.time()
        print(f"✅Start training of {self.model_name}\n")
        print(f"✅Accurate RAM Memory: {self.results.ModelRam} KB\n")
        print(f"✅Flash Memory Ram: {self.results.estimatedFlash} KB\n")

        callbacks:list = [checkpoint,lr_schedule,swa_callback]

        if self.performaceStoppage:
            """
            If for some specific epochs we do not see a big validation accuracy increase we stop the training
            """
            print("🔁 Training with **Performance Callback** enabled.")
            time.sleep(3)
            callbacks.append(performanceCallback)
        else:
            print("⚙️  Training without Performance Callback.")

        if self.early_stopping_acc:
            """
            If for fewer epochs we do not see absolutely any accuracy improvement, we stop the training
            """
            print("🛑 Training with **Early Stopping Accuracy Callback** enabled.")
            time.sleep(3)
            callbacks.append(early_stopping_acc)
        else:
            print("⚙️  Training without Early Stopping Accuracy Callback.")

        if self.midway_callback:
            """
            If at a specific epoch, the model has not reached a threshold of accuracy, we stop its training
            """
            print("⏱️ Training with **Midway Accuracy Callback** enabled.")
            time.sleep(3)
            callbacks.append(midway_callback)
        else:
            print("⚙️  Training without Midway Accuracy Callback.")

        if self.enable_dropout:
            print("🎯 Training with **Adaptive Dropout** enabled.")
            callbacks.append(adjust_dropout)
        else:
            print("⚙️  Training without Adaptive Dropout.")

        history = self.model.fit(
            x_train, y_train,
            epochs = self.epochs,
            batch_size = batchSize,
            validation_data=(x_test, y_test),
            verbose=2,
            callbacks=callbacks
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

        goal_val_accuract:float = 0.90


        if best_test_acc > goal_val_accuract:
            extra_epochs:int = 50
            print(f"\n🚀 Best test accuracy ({best_test_acc:.4f} = {best_test_acc * 100}) exceeded {goal_val_accuract * 100}%. Continuing training for {extra_epochs} more epochs.\n")
            
            already_used_epochs = self.epochs if self.epochs else self.train_params["num_epochs"]

            print(f"🔧 Setting learning rate very low for polishing phase...")
            final_lr = self.model.optimizer.learning_rate.numpy()

            polishing_lr = max(final_lr * 0.3, 1e-3)

            self.model.optimizer.learning_rate.assign(polishing_lr)
            print(f"🔧 Fine-tuning with learning rate: {polishing_lr:.6f}")

            callbacks2 = [ checkpoint, swa_callback ]
            if self.enable_dropout:
                print("✅ The TakuNet model is trained with Dropout\n")
                callbacks2.append(adjust_dropout)
            else:
                print("Training happens without Dropout\n")
            
            history_extra = self.model.fit(
                x_train, y_train,
                epochs=already_used_epochs + extra_epochs,
                initial_epoch=already_used_epochs,
                batch_size=int(batchSize / 2),
                validation_data=(x_test, y_test),
                verbose=2,
                callbacks=callbacks2
            )

            for key in full_history.history.keys():
                if key in history_extra.history:
                    full_history.history[key].extend(history_extra.history[key])
                elif key == "learning_rate":
                    fixed_lr = polishing_lr
                    num_extra_epochs = len(history_extra.history["loss"])
                    full_history.history[key].extend([fixed_lr] * num_extra_epochs)
                    print(f"ℹ️ Filled 'learning_rate' with fixed value {fixed_lr:.6f} for {num_extra_epochs} extra epochs.")
                else:
                    print(f"⚠️ Skipping key '{key}' — not found in extra training history.")


            # Update best accuracy after extra training

            total_epochs_trained = len(full_history.history['loss'])
            best_test_acc = max(full_history.history['val_accuracy'])
            print(f"\n🔁 Continued Training Complete. New Best Test Accuracy: {best_test_acc:.4f}\n")

        # **Load final Best Model**
        self.model.load_weights(checkpoint_path) # I want here to bring the best model back in order to convert it
        print(f"✅ Final Best model restored from {checkpoint_path}\n")

        print("\n🔄 Applying Moving Average (SWA) weights...\n")
        swa_callback.apply_swa_weights()
        _, swa_val_accuracy = self.evaluate(x_test, y_test)
        print("✅ SWA weights applied!\n")

        # **Predictions & Metrics**
        y_test_pred = self.model.predict(x_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # **Save Results**
        self.results.history = full_history
        self.results.epochs_trained = total_epochs_trained
        self.results.train_accuracy = max(full_history.history['accuracy'])
        self.results.test_accuracy = best_test_acc
        self.results.SWA_test_accuracy = swa_val_accuracy
        self.results.precision = precision_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.recall = recall_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.f1_score = f1_score(y_true_classes, y_test_pred_classes, average='macro')
        self.results.training_time = training_time 
        self.results.fitness_score = self._fitness()
        
        # ** Declare that this model is trained.
        self.is_trained = True
        self._freeze_dropout_for_inference()
        # **Save Model in Multiple Formats**
        if self.hardwareConstrains == True:
            tfilte_time_start = time.time()
            self._convert_to_tflite(x_train=x_train)
            self._convert_tflite_to_c_array()
            self.results.flops = self._count_flops()
            print(f"📊 Estimated FLOPs: {self.results.flops:,}")

            # **Evaluate the TFLite Model**
            try:
                self.results.tflite_accuracy = self._evaluate_tflite_model(x_test=x_test, y_test=y_test)
            except Exception as e:
                print(f"❌ TFLite evaluation failed: {e}")
                self.results.tflite_accuracy = 0.0
                
            self.results.tfliteConversionTime = time.time() - tfilte_time_start
            print(f"Test Accuracy (TFLite): {self.results.tflite_accuracy:.4f}")

            # **File Size Reporting**
            keras_size_kb = os.path.getsize(checkpoint_path) / 1024
            tflite_size_kb = os.path.getsize(f"{self.folderName}/TfLiteModels/{self.model_name}.tflite") / 1024
            c_array_size_kb = os.path.getsize(f"{self.folderName}/HeaderFiles/{self.model_name}.h") / 1024
            if "Retraining" not in self.folderName:
                save_config_to_file(self.model_params, f"{self.folderName}/saved_configs/model_params/{self.model_name}_model_params.json")
                save_config_to_file(self.train_params, f"{self.folderName}/saved_configs/train_params/{self.model_name}_train_params.json")
            else:
                print("⚠️ Skipping config saving (Retraining mode detected). Config already saved\n")

            self.results.tflite_size = tflite_size_kb
            
            print(f"Keras Model Size: {keras_size_kb:.2f} KB")
            print(f"TFLite Model Size: {tflite_size_kb:.2f} KB")
            print(f"C Array File Size: {c_array_size_kb:.2f} KB")
            print(f"RAM: {self.results.ModelRam:.2f} KB")
        else:
            print("⚠️ We don't check for Hardware constrains so it is not Safe to convert to TfLite\n")

    def summary(self):
        self.model.summary()

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=2)
    
    def get_model(self):
        return self.model




# Helpers
def save_config_to_file(config: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

def get_optimizer(name, learning_rate):
    """Returns the optimizer instance based on the name."""
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "adamw": AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        "sgd": SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4),
        "rmsprop": RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(name.lower(), Adam(learning_rate=learning_rate))


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



@register_keras_serializable()
class AdaptiveDropout(tf.keras.layers.Layer):
    def __init__(self, initial_rate=0.1, enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.initial_rate = initial_rate
        self.rate = tf.Variable(initial_value=initial_rate, trainable=False, dtype=tf.float32)
        self.addtion:float = 0.0
        self.enabled = enabled
        

    def call(self, inputs, training=True):
        """

        Use noise to emulate the SpatialDropout2D 
        Use None to have the original Dropout

        """
        if not self.enabled:
            return inputs
        
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
    def __init__(self, model_instance: TakuNetModel, overfitting_threshold: float = 0.1, cooldown: int = 3):
        super().__init__()
        self.model_instance = model_instance
        self.overfitting_threshold = overfitting_threshold
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.dropout_initialized = False  # No manual start_epoch anymore

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        if train_acc is None or val_acc is None:
            return

        gap = train_acc - val_acc

        if not self.dropout_initialized and gap > self.overfitting_threshold:
            # 🚀 First overfitting detected — initialize dropouts now
            print(f"\n🚀 Initializing Dropout rates at Epoch {epoch} due to overfitting!")
            self._initialize_dropout_rates()
            self.dropout_initialized = True
            self.cooldown_counter = self.cooldown  # start cooldown
            return

        if self.dropout_initialized:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return

            if gap > self.overfitting_threshold:
                print(f"\n⚠️ Overfitting detected! Train Acc - Val Acc = {gap:.3f} > {self.overfitting_threshold}")
                self._increase_one_dropout()
                self.cooldown_counter = self.cooldown


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
                layer.addtion = 0.05
                layer.max_rate = 0.15
            elif "taku" in layer.name:
                initial_rate = 0.03
                layer.addtion = 0.05
                layer.max_rate = 0.4
            elif "refiner1" in layer.name:
                initial_rate = 0.05
                layer.addtion = 0.05
                layer.max_rate = 0.4
            elif "refiner2" in layer.name:
                initial_rate = 0.1
                layer.addtion = 0.05
                layer.max_rate = 0.5
            else:
                initial_rate = 0.05
                layer.addtion = 0.05
                layer.max_rate = 0.3

            layer.rate.assign(initial_rate)
            print(f"🔧 {layer.name}: initialized dropout rate to {initial_rate:.3f} (max {layer.max_rate:.3f})")



    def _increase_one_dropout(self):
        dropout_layers:List[AdaptiveDropout] = []
        weights = []
        if self.model_instance.adaptive_dropout_stem is not None:
            dropout_layers.append(self.model_instance.adaptive_dropout_stem)
            weights.append(1)

        if self.model_instance.adaptive_dropout_taku is not None:
            layers = [d for d in self.model_instance.adaptive_dropout_taku if d is not None]
            dropout_layers.extend(layers)
            weights.extend([1] * len(layers))

        if self.model_instance.adaptive_dropout_refiner is not None:
            layers = [d for d in self.model_instance.adaptive_dropout_refiner if d is not None]
            dropout_layers.extend(layers)
            weights.extend([4] * len(layers))

        if not dropout_layers:
            print("⚠️ No AdaptiveDropout layers found to adjust.")
            return

        chosen_layer:AdaptiveDropout = random.choices(dropout_layers, weights=weights, k=1)[0]
        old_rate = float(chosen_layer.rate.numpy())
        new_rate = max(0.05, min(old_rate + chosen_layer.addtion, chosen_layer.max_rate)) 
        chosen_layer.rate.assign(new_rate)
        print(f"🔧 {chosen_layer.name}: dropout rate increased from {old_rate:.3f} to {new_rate:.3f}")


class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.total_steps = total_steps
    def __call__(self, epoch):
        return self.initial_lr * (1.0 - epoch / self.total_epochs)


class MidwayStopCallback(Callback):
    def __init__(self, stopEpoch:int, threshold:float):
        super().__init__()
        self.mid_epoch = stopEpoch
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.mid_epoch:
            train_acc = logs.get('accuracy')
            val_acc = logs.get('val_accuracy')
            print(f"\nMidway Epoch {epoch}: Training Acc = {train_acc}, Validation Acc = {val_acc}")
            if val_acc < self.threshold:  
                print(f"\n🚨 Stopping early: Training accuracy is below {self.threshold} at epoch {epoch}")
                self.model.stop_training = True

class PerformanceStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_improvement=0.02):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_val_acc = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')
        if current_val_acc is None:
            return

        if current_val_acc > self.best_val_acc * (1 + self.min_improvement):
            self.best_val_acc = current_val_acc
            self.wait = 0  # Reset wait
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n🚨 Early stopping: No val_acc improvement >{self.min_improvement*100:.1f}% in {self.patience} epochs.")
                self.model.stop_training = True



class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, model, swa_start=10):
        super().__init__()
        self.swa_start = swa_start
        self._model_ref = model  # ✅ fix: avoid clashing with built-in .model
        self.weights_accumulator = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start:
            self.weights_accumulator.append(self._model_ref.get_weights())

    def apply_swa_weights(self):
        if not self.weights_accumulator:
            print("⚠️ No SWA weights to average.")
            return

        avg_weights = []
        for weights in zip(*self.weights_accumulator):
            avg_weights.append(np.mean(weights, axis=0))

        self._model_ref.set_weights(avg_weights)
        print("✅ Manual SWA weights applied.")


class StaticReshape(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, (-1,) + self.target_shape)


class TrainingResults:
    """Class to store training and evaluation results."""
    def __init__(self):
        self.history = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.SWA_test_accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.ModelRam = 0.0
        self.estimatedFlash = 0.0
        self.training_time = None
        self.fitness_score = None
        self.tflite_accuracy = None
        self.tflite_size = None
        self.tfliteConversionTime = None
        self.epochs_trained = None
        self.flops = None

    def __repr__(self):
        return (f"TrainingResults(\n"
                f"  Train Accuracy: {self.train_accuracy:.4f}\n"
                f"  Test Accuracy: {self.test_accuracy:.4f}\n"
                f"  Swa Test Accuracy: {self.SWA_test_accuracy:.4f}\n"
                f"  TFlite Test Accuracy: {self.tflite_accuracy:.4f}\n"
                f"  Precision: {self.precision:.4f}\n"
                f"  Recall: {self.recall:.4f}\n"
                f"  F1 Score: {self.f1_score:.4f}\n"
                f"  Analyzed Max Ram Use: {self.ModelRam:.4f}\n"
                f"  Estimated Flash Memory Use: {self.estimatedFlash:.4f}\n"
                f"  TFlite Memory Use: {self.tflite_size:.4f}\n"
                f"  Training Time: {self.training_time}\n"
                f"  FLOPs: {self.flops:,}\n)")  
