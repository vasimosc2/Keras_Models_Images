import os

def rename_and_replace(folder_path, input_filename, new_model_name):
    # Build the full path to the input file
    input_path = os.path.join(folder_path, input_filename)

    # Generate the new filename by replacing occurrences in the original filename
    new_filename = input_filename.replace("CROSSOVER", new_model_name.upper()).replace("Crossover", new_model_name)
    output_path = os.path.join(folder_path, new_filename)

    # Read the original file content
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace inside the file content
    content_modified = content.replace("CROSSOVER", new_model_name.upper()).replace("Crossover", new_model_name)

    # Write the modified content to the new file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content_modified)

    print(f"✅ File saved as: {output_path}")

    os.remove(input_path)
    print(f"🗑️ Original file deleted: {input_path}")

# Example usage
if __name__ == "__main__":
    folder = "/mnt/c/Users/mosho/OneDrive/Arduino/First_Attempt"
    original_file = "TakuNet_Crossover_15.h"
    new_model = "Random"  # This will replace CROSSOVER → RANDOM and Crossover → Random
    rename_and_replace(folder, original_file, new_model)
