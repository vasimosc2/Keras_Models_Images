import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        if any(tf.config.experimental.get_memory_growth(gpu) for gpu in gpus):
            print("⚠️ GPU is already initialized! `set_memory_growth()` will fail.")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️ No GPU found, running on CPU.")



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import memoryEstimator

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

input_shape = (32, 32, 3)
num_classes = 100

# --- ResNet50 Training ---
print("\n\n🔧 Training ResNet50")
resnet_input = Input(shape=input_shape)
base_model = ResNet50(include_top=False, weights=None, input_tensor=resnet_input)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(num_classes, activation='softmax')(x)
resnet_model = Model(inputs=resnet_input, outputs=x)

# Estimate memory usage for ResNet
batch_size = 64
max_ram_usage, flash_memory, total_memory = memoryEstimator.memoryEstimation(model=resnet_model, 
                                                                             data_dtype_multiplier=4)

print(f"🧠 Estimated RAM memory usage for ResNet50: {max_ram_usage:.2f} MB")
print(f"🧠 Estimated FLASH memory usage for ResNet50: {flash_memory:.2f} MB")

resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                     metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)
]

resnet_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=batch_size,
    callbacks=callbacks
)

resnet_eval = resnet_model.evaluate(x_test, y_test, verbose=2)
print(f"✅ ResNet50 Test Accuracy: {resnet_eval[1] * 100:.2f}%")
