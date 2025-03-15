import tensorflow as tf # type: ignore


def load_cifar100(output_classes:int):
    """Loads CIFAR-100 dataset and normalizes it."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

    y_train = tf.keras.utils.to_categorical(y_train, output_classes)
    y_test = tf.keras.utils.to_categorical(y_test, output_classes)

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    return x_train, y_train, x_test, y_test

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

def create_augmented_dataset(x, y):
    """Creates an augmented dataset efficiently using tf.data."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    aug_dataset = dataset.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf.data.AUTOTUNE)
    aug_dataset = aug_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

    x_aug_list, y_aug_list = [], []
    for img_batch, label_batch in aug_dataset:
        x_aug_list.append(img_batch)
        y_aug_list.append(label_batch)

    x_aug = tf.concat(x_aug_list, axis=0)
    y_aug = tf.concat(y_aug_list, axis=0)

    return tf.concat([x, x_aug], axis=0), tf.concat([y, y_aug], axis=0)

def get_dataset(output_classes:int,use_augmented_data=False):
    """Returns the CIFAR-100 dataset, either augmented or original."""
    x_train, y_train, x_test, y_test = load_cifar100(output_classes=output_classes)
    if use_augmented_data:
        print("🔄 Augmenting dataset...")
        print(f"Initial Training size : {x_train.shape[0]} images")
        x_train, y_train = create_augmented_dataset(x_train, y_train)
        print(f"✅ Dataset Size After augmentation: {x_train.shape[0]} images")
    else:
        print("✅ Using original dataset.")

    return x_train, y_train, x_test, y_test
