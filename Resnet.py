import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import memoryEstimator

# GPU Setup
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

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Use tf.data pipeline to resize efficiently
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .map(lambda x, y: (tf.image.resize(x, [224, 224]), y), num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(64).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .map(lambda x, y: (tf.image.resize(x, [224, 224]), y), num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(64).prefetch(tf.data.AUTOTUNE)

input_shape = (224, 224, 3)
num_classes = 100

# --- EfficientNetB0 Training ---
print("\n\n🔧 Training EfficientNetB0")
eff_input = Input(shape=input_shape)
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=eff_input, pooling='avg')
x = Dense(num_classes, activation='softmax')(base_model.output)
eff_model = Model(inputs=eff_input, outputs=x)

# Estimate memory usage for EfficientNetB0
batch_size = 64
max_ram_usage, flash_memory, total_memory = memoryEstimator.memoryEstimation(model=eff_model, 
                                                                             data_dtype_multiplier=4)

print(f"🧠 Estimated RAM memory usage for EfficientNetB0: {max_ram_usage:.2f} MB")
print(f"🧠 Estimated FLASH memory usage for EfficientNetB0: {flash_memory:.2f} MB")

eff_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)
]

eff_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=callbacks
)

eff_eval = eff_model.evaluate(test_ds, verbose=2)
print(f"✅ EfficientNetB0 Test Accuracy: {eff_eval[1] * 100:.2f}%")