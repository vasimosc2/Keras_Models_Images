import argparse
import os
import csv

def filter_symbols_by_memory(input_file: str, memory_type: str, output_csv: str = "filtered_symbols.csv"):
    # Flash and RAM symbol type tags
    flash_types = {'T', 't', 'R', 'r', 'w', 'W'}
    ram_types = {'D', 'd', 'B', 'b'}

    # Set target symbol types based on memory type
    if memory_type == 'flash':
        target_types = flash_types
    elif memory_type == 'ram':
        target_types = ram_types
    else:
        raise ValueError("Invalid memory type. Use 'flash' or 'ram'.")

    # Build full path to the input file
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, input_file)

    filtered = []

    # Read and filter symbol lines
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[2] in target_types:
                address = parts[0]
                size_str = parts[1]
                try:
                    size_int = int(size_str, 16)  # Always interpret as hex
                except ValueError:
                    continue  # Skip malformed lines
                symbol_type = parts[2]
                name = ' '.join(parts[3:])
                filtered.append({
                    'address': address,
                    'size_str': size_str,
                    'size_int': size_int,
                    'type': symbol_type,
                    'name': name
                })

    # Save filtered results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['address', 'size_str','size_int', 'type', 'name'])
        writer.writeheader()
        for row in filtered:
            writer.writerow({
                'address': row['address'],
                'size_str': row['size_str'],
                'size_int': row['size_int'],
                'type': row['type'],
                'name': row['name']
            })

    # Sort symbols by actual size and extract top 5
    top5 = sorted(filtered, key=lambda x: x['size_int'], reverse=True)[:5]

    print(f"\nTop 5 {memory_type} memory-consuming objects:")
    for entry in top5:
        print(f"{entry['name']} - {entry['size_str']} (hex)")

    total_top5 = sum(entry['size_int'] for entry in top5)
    total_memory = sum(entry['size_int'] for entry in filtered)

    print(f"\nTotal memory usage: {total_memory} bytes")
    print(f"Top 5 symbols consume {total_top5} bytes ({(total_top5 / total_memory * 100):.2f}% of total)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Flash or RAM symbols from ELF symbol table.")
    parser.add_argument("--input_file", default="symbols.txt", help="Path to the symbols.txt file")
    parser.add_argument("--memory_type", default="ram", choices=["flash", "ram"], help="Memory type to filter")

    args = parser.parse_args()
    output_csv = f"filtered_symbols_{args.memory_type}.csv"

    filter_symbols_by_memory(args.input_file, args.memory_type, output_csv)
