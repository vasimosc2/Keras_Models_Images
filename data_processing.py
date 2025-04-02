from typing import Tuple
import tensorflow as tf

def load_cifar100(output_classes: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, output_classes)
    y_test = tf.keras.utils.to_categorical(y_test, output_classes)

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    return x_train, y_train, x_test, y_test

def get_augmentation_pipeline(aug_type: str) -> tf.keras.Sequential:
    if aug_type == "standard":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ])
    elif aug_type == "color":
        return tf.keras.Sequential([
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ])
    elif aug_type == "geometric":
        return tf.keras.Sequential([
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
    elif aug_type == "none":
        return tf.keras.Sequential([])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def mixup(x: tf.Tensor, y: tf.Tensor, alpha: float = 0.4) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(x)[0]
    idx = tf.random.shuffle(tf.range(batch_size))
    shuffled_x = tf.gather(x, idx)
    shuffled_y = tf.gather(y, idx)

    lambda_val = tf.random.uniform([], minval=0, maxval=alpha)
    x_mix = lambda_val * x + (1 - lambda_val) * shuffled_x
    y_mix = lambda_val * y + (1 - lambda_val) * shuffled_y

    return x_mix, y_mix


def apply_pipeline(x: tf.Tensor, y: tf.Tensor, augmentation: tf.keras.Sequential) -> Tuple[tf.Tensor, tf.Tensor]:
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    aug_dataset = dataset.map(lambda img, label: (augmentation(img), label), num_parallel_calls=tf.data.AUTOTUNE)
    aug_dataset = aug_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

    x_aug_list, y_aug_list = [], []
    for img_batch, label_batch in aug_dataset:
        x_aug_list.append(img_batch)
        y_aug_list.append(label_batch)

    x_aug = tf.concat(x_aug_list, axis=0)
    y_aug = tf.concat(y_aug_list, axis=0)
    return x_aug, y_aug

def create_augmented_dataset(
    x: tf.Tensor,
    y: tf.Tensor,
    apply_standard: bool = True,
    apply_color: bool = False,
    apply_geometric: bool = False,
    apply_mixup: bool = False
) -> Tuple[tf.Tensor, tf.Tensor]:

    aug_x_list = [x]
    aug_y_list = [y]

    if apply_standard:
        print("✅ Applying Standard Augmentation")
        aug = get_augmentation_pipeline("standard")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)

    if apply_color:
        print("✅ Applying Color Augmentation")
        aug = get_augmentation_pipeline("color")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)

    if apply_geometric:
        print("✅ Applying Geometric Augmentation")
        aug = get_augmentation_pipeline("geometric")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)

    if apply_mixup:
        print("✅ Applying MixUp Augmentation")
        x_mix, y_mix = mixup(x, y)
        aug_x_list.append(x_mix)
        aug_y_list.append(y_mix)

    return tf.concat(aug_x_list, axis=0), tf.concat(aug_y_list, axis=0)



def get_dataset(
    output_classes: int,
    use_augmented_data: bool = False,
    apply_standard: bool = False,
    apply_color: bool = False,
    apply_geometric: bool = False,
    apply_mixup: bool = False
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    x_train, y_train, x_test, y_test = load_cifar100(output_classes)

    if use_augmented_data:
        x_train, y_train = create_augmented_dataset(
            x_train, y_train,
            apply_standard=apply_standard,
            apply_color=apply_color,
            apply_geometric=apply_geometric,
            apply_mixup=apply_mixup
        )
        print(f"✅ Final Augmented Training Set Size: {x_train.shape[0]}")
    else:
        print("✅ Using original dataset without augmentation.")

    return x_train, y_train, x_test, y_test
