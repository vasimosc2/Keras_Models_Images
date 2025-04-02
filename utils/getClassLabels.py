import os
import pickle
import json
from tensorflow.keras.utils import get_file

def save_fine_labels_to_json(save_path="cifar100_fine_labels.json"):
    meta_path = get_file(
        "cifar-100-python",
        origin="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        untar=True
    )
    meta_file = os.path.join(meta_path, "cifar-100-python", "meta")

    with open(meta_file, 'rb') as f:
        labels = pickle.load(f, encoding='latin1')
        fine_labels = labels['fine_label_names']

    with open(save_path, "w") as f:
        json.dump(fine_labels, f, indent=2)
    print(f"✅ Saved fine labels to {save_path}")

def load_fine_labels_from_json(path="cifar100_fine_labels.json"):
    with open(path, "r") as f:
        return json.load(f)