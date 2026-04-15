import sys
from pathlib import Path

def count_keras_files(folder_name):
    base_path = Path("/zhome/02/e/181021/Desktop/Keras_Models_Images/NAS")
    target_folder = base_path / folder_name / "saved_models"

    if not target_folder.exists():
        print(f"Folder does not exist: {target_folder}")
        return

    if not target_folder.is_dir():
        print(f"Not a directory: {target_folder}")
        return

    keras_files = list(target_folder.glob("*.keras"))
    print(f"Folder: {target_folder}")
    print(f"Number of .keras files: {len(keras_files)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_keras.py <folder_name>")
        print("Example: python count_keras.py Nov-22")
    else:
        count_keras_files(sys.argv[1])