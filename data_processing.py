from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from utils import getClassLabels

def load_cifar100(output_classes: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.device('/CPU:0'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

        # First cast to float32, then divide
        x_train = tf.cast(x_train, tf.float32) / 255.0
        x_test = tf.cast(x_test, tf.float32) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, output_classes)
    y_test = tf.keras.utils.to_categorical(y_test, output_classes)

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

def mixup(x: tf.Tensor, y: tf.Tensor, alpha: float = 0.4, batch_size: int = 1024):
    num_samples = tf.shape(x)[0]
    x_mix_list, y_mix_list = [], []
    idx_list = []

    for i in range(0, num_samples, batch_size):
        end = tf.minimum(i + batch_size, num_samples)
        x_batch = tf.cast(x[i:end], tf.float32)
        y_batch = tf.cast(y[i:end], tf.float32)

        idx = tf.random.shuffle(tf.range(tf.shape(x_batch)[0]))
        shuffled_x = tf.gather(x_batch, idx)
        shuffled_y = tf.gather(y_batch, idx)

        lambda_val = tf.constant(alpha, dtype=tf.float32)
        print(lambda_val)
        x_mix = lambda_val * x_batch + (1 - lambda_val) * shuffled_x
        y_mix = lambda_val * y_batch + (1 - lambda_val) * shuffled_y

        x_mix_list.append(x_mix)
        y_mix_list.append(y_mix)
        idx_list.append((tf.range(i, end), tf.gather(tf.range(i, end), idx)))

    return tf.concat(x_mix_list, axis=0), tf.concat(y_mix_list, axis=0), idx_list

def cutmix_batch(x, y, alpha=1.0):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    shuffled_x = tf.gather(x, indices)
    shuffled_y = tf.gather(y, indices)

    lam = tf.random.gamma(shape=[], alpha=alpha, beta=alpha)
    
    img_h = tf.shape(x)[1]
    img_w = tf.shape(x)[2]

    r_x = tf.cast(tf.random.uniform([], 0, img_w), tf.float32)
    r_y = tf.cast(tf.random.uniform([], 0, img_h), tf.float32)
    r_w = tf.cast(img_w * tf.math.sqrt(1. - lam), tf.float32)
    r_h = tf.cast(img_h * tf.math.sqrt(1. - lam), tf.float32)

    x1 = tf.clip_by_value(r_x - r_w // 2, 0, img_w)
    y1 = tf.clip_by_value(r_y - r_h // 2, 0, img_h)
    x2 = tf.clip_by_value(r_x + r_w // 2, 0, img_w)
    y2 = tf.clip_by_value(r_y + r_h // 2, 0, img_h)

    # Replace region
    patched_x = x
    patched_x = tf.tensor_scatter_nd_update(
        patched_x,
        indices=tf.reshape(tf.range(batch_size), (-1, 1)),
        updates=tf.tensor_scatter_nd_update(
            x,
            indices=tf.reshape(tf.range(batch_size), (-1, 1)),
            updates=tf.tensor_scatter_nd_update(x, [[0]], [shuffled_x[0]])  # dummy to keep shapes
        )
    )

    # But it's better to use slicing directly:
    x_cutmix = tf.identity(x)
    x_cutmix[:, y1:y2, x1:x2, :].assign(shuffled_x[:, y1:y2, x1:x2, :])

    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    y_cutmix = lam_adjusted * y + (1 - lam_adjusted) * shuffled_y

    return x_cutmix, y_cutmix


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

def save_mixup_samples(x: tf.Tensor, y: tf.Tensor, x_mix: tf.Tensor, y_mix: tf.Tensor, idx_list, root_folder: str):
    """Save 5 MixUp samples under Samples/mixup/ with class names loaded from CIFAR meta file."""
    fine_labels = getClassLabels.load_fine_labels_from_json()
    folder_path = os.path.join(root_folder, "mixup")
    os.makedirs(folder_path, exist_ok=True)

    full_idx_A = tf.concat([pair[0] for pair in idx_list], axis=0).numpy()
    full_idx_B = tf.concat([pair[1] for pair in idx_list], axis=0).numpy()

    for i in range(5):
        idx_a = full_idx_A[i]
        idx_b = full_idx_B[i]

        img_a = x[idx_a].numpy()
        img_b = x[idx_b].numpy()
        img_mix = x_mix[i].numpy()

        label_mix = y_mix[i].numpy()
        label_indices = np.argsort(label_mix)[-2:]
        weights = label_mix[label_indices]

        name_a = fine_labels[np.argmax(y[idx_a])]
        name_b = fine_labels[np.argmax(y[idx_b])]


        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_a)
        axs[0].set_title(f"Image A:\n{name_a}")
        axs[0].axis("off")

        axs[1].imshow(img_b)
        axs[1].set_title(f"Image B:\n{name_b}")
        axs[1].axis("off")

        axs[2].imshow(img_mix)
        axs[2].set_title(f"MixUp\n{weights[0]:.2f}*{fine_labels[label_indices[0]]}, {weights[1]:.2f}*{fine_labels[label_indices[1]]}")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(folder_path, f"mixup_sample_{i}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ Saved mixup samples with class names to: {folder_path}")

def save_cutmix_samples(x: tf.Tensor, y: tf.Tensor, x_cut: tf.Tensor, y_cut: tf.Tensor, shuffled_indices, root_folder: str):
    """Save 5 CutMix samples with class names."""
    fine_labels = getClassLabels.load_fine_labels_from_json()
    folder_path = os.path.join(root_folder, "cutmix")
    os.makedirs(folder_path, exist_ok=True)

    for i in range(5):
        idx_a = i
        idx_b = shuffled_indices[i].numpy()

        img_a = x[idx_a].numpy()
        img_b = x[idx_b].numpy()
        img_cut = x_cut[i].numpy()

        label_mix = y_cut[i].numpy()
        label_indices = np.argsort(label_mix)[-2:]
        weights = label_mix[label_indices]

        name_a = fine_labels[np.argmax(y[idx_a])]
        name_b = fine_labels[np.argmax(y[idx_b])]

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_a)
        axs[0].set_title(f"Image A:\n{name_a}")
        axs[0].axis("off")

        axs[1].imshow(img_b)
        axs[1].set_title(f"Image B:\n{name_b}")
        axs[1].axis("off")

        axs[2].imshow(img_cut)
        axs[2].set_title(f"CutMix\n{weights[0]:.2f}*{fine_labels[label_indices[0]]}, {weights[1]:.2f}*{fine_labels[label_indices[1]]}")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(folder_path, f"cutmix_sample_{i}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ Saved CutMix samples with class names to: {folder_path}")


def save_augmented_samples(x: tf.Tensor, y: tf.Tensor, root_folder: str, aug_type: str):
    """Save 5 sample images before and after augmentation to Samples/<aug_type>/"""
    folder_path = os.path.join(root_folder, aug_type)
    os.makedirs(folder_path, exist_ok=True)

    augmentation = get_augmentation_pipeline(aug_type)

    for i in range(5):
        img = x[i]
        label = tf.argmax(y[i]).numpy()

        aug_img = augmentation(tf.expand_dims(img, 0))[0]
        aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
        aug_img = tf.where(tf.math.is_nan(aug_img), 0.0, aug_img).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img)
        axs[0].set_title(f"Original (Class {label})")
        axs[0].axis("off")

        axs[1].imshow(aug_img)
        axs[1].set_title(f"{aug_type.capitalize()} Augmented")
        axs[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(folder_path, f"{aug_type}_sample_{i}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ Saved {aug_type} samples to: {folder_path}")


def create_augmented_dataset(
    x: tf.Tensor,
    y: tf.Tensor,
    apply_standard: bool = True,
    apply_color: bool = False,
    apply_geometric: bool = False,
    apply_mixup: bool = False,
    apply_cutmix: bool = False 
) -> Tuple[tf.Tensor, tf.Tensor]:

    aug_x_list = [x]
    aug_y_list = [y]

    if apply_mixup:
        print("Applying MixUp Augmentation ....\n")
        x_mix, y_mix, idx_list = mixup(x, y)
        aug_x_list.append(x_mix)
        aug_y_list.append(y_mix)
        save_mixup_samples(x, y, x_mix, y_mix, idx_list, root_folder="Samples")
        print("✅\n")
    
    if apply_cutmix:
        print("Applying CutMix Augmentation ....\n")
        x_cut, y_cut = cutmix_batch(x, y)
        aug_x_list.append(x_cut)
        aug_y_list.append(y_cut)
        shuffled_indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:5]  # just for sample display
        save_cutmix_samples(x, y, x_cut, y_cut, shuffled_indices, root_folder="Samples")
        print("✅\n")

    if apply_standard:
        print("Applying Standard Augmentation ....\n")
        aug = get_augmentation_pipeline("standard")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)
        save_augmented_samples(x, y, root_folder="Samples", aug_type="standard")
        print("✅\n")

    if apply_color:
        print("Applying Color Augmentation ....\n")
        aug = get_augmentation_pipeline("color")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)
        save_augmented_samples(x, y, root_folder="Samples", aug_type="color")
        print("✅\n")

    if apply_geometric:
        print("Applying Geometric Augmentation ....\n")
        aug = get_augmentation_pipeline("geometric")
        aug_x, aug_y = apply_pipeline(x, y, aug)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)
        save_augmented_samples(x, y, root_folder="Samples", aug_type="geometric")
        print("✅\n")

    with tf.device('/CPU:0'):
        x_train = tf.concat(aug_x_list, axis=0)
        y_train = tf.concat(aug_y_list, axis=0)
        
    return x_train, y_train

def get_dataset(
    output_classes: int,
    augementation_technique:Dict = {"apply_standard":False,
                                    "apply_color":False,
                                    "apply_geometric":False,
                                    "apply_mixup": False,
                                    "apply_cutmix": False
                                    }

) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    x_train, y_train, x_test, y_test = load_cifar100(output_classes)

    if augementation_technique.get("apply_standard") or augementation_technique.get("apply_color") or augementation_technique.get("apply_geometric") or augementation_technique.get("apply_mixup") or augementation_technique.get(" apply_cutmix ") :
        x_train, y_train = create_augmented_dataset(
            x_train, y_train,
            apply_standard=augementation_technique.get("apply_standard"),
            apply_color=augementation_technique.get("apply_color"),
            apply_geometric=augementation_technique.get("apply_geometric"),
            apply_mixup=augementation_technique.get("apply_mixup"),
            apply_cutmix=augementation_technique.get("apply_cutmix")
        )
        print(f"✅ Final Augmented Training Set Size: {x_train.shape[0]}")
    else:
        print("✅ Using original dataset without augmentation.")

    return x_train, y_train, x_test, y_test
