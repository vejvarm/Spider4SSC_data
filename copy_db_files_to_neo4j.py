import shutil
from pathlib import Path

def collect_ttl_files(source_dir, output_dir):
    # Convert strings to Path objects
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Use rglob to find all .ttl files recursively
    count = 0
    for file in source_path.rglob('*.ttl'):
        # Define the destination path
        destination = output_path / file.name
        
        # Copy the file
        shutil.copy2(file, destination)
        print(f"Copied: {file.name}")
        count += 1

    print(f"\nFinished! Moved {count} files to {output_dir}")

# --- Configuration ---
SOURCE = 'data/Spider4SSC/test_database'
OUTPUT = '/home/freya/neo4j/import/KGs_test'

if __name__ == "__main__":
    collect_ttl_files(SOURCE, OUTPUT)