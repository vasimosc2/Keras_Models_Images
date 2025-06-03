import argparse
import os

def rename_and_replace(folder_path, input_filename, new_model_name,givenName):
    # Build the full path to the input file
    input_path = os.path.join(folder_path, input_filename)

    # Generate the new filename by replacing occurrences in the original filename
    new_filename = input_filename.replace(givenName.upper(), new_model_name.upper()).replace(givenName, new_model_name)
    output_path = os.path.join(folder_path, new_filename)

    # Read the original file content
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace inside the file content
    content_modified = content.replace(givenName.upper(), new_model_name.upper()).replace(givenName, new_model_name)

    # Write the modified content to the new file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content_modified)

    print(f"✅ File saved as: {output_path}")

    os.remove(input_path)
    print(f"🗑️ Original file deleted: {input_path}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change the name of the TakuNet model to upload it into Arduino")
    parser.add_argument("--givenName", default="Crossover", choices=["Init", "Crossover", "Mutant"] ,help="Name of the file")
    parser.add_argument("--number", default="22", help="Memory type to filter")
    args = parser.parse_args()

    folder = "/mnt/c/Users/mosho/OneDrive/Arduino/First_Attempt"
    original_file = f"TakuNet_{args.givenName}_{args.number}.h"

    new_model = "Random"  # This will replace CROSSOVER → RANDOM and Crossover → Random
    rename_and_replace(folder, original_file, new_model, args.givenName)
