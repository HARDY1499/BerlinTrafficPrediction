import tarfile
import os
import glob

# --- Configuration ---
# This should match the DOWNLOAD_DIR from the other script
ARCHIVE_DIR = "berlin_traffic_data/2024/Fahrstreifendetektoren_tgz"

# Where to put the extracted CSV files
EXTRACT_DIR = os.path.join("berlin_traffic_data", "2024", "CSV_data")
# --- End of Configuration ---

def extract_all_archives(archive_path, extract_path):
    """
    Finds all .tgz files in archive_path and extracts them
    to extract_path.
    """
    print(f"--- Archive Extractor ---")
    print(f"Looking for .tgz files in: {archive_path}")
    print(f"Extracting CSVs to: {extract_path}")

    # 1. Create the extraction directory
    os.makedirs(extract_path, exist_ok=True)

    # 2. Find all .tgz files in the archive directory
    # The glob.glob function finds files matching a pattern
    archive_files = glob.glob(os.path.join(archive_path, "*.tgz"))
    
    if not archive_files:
        print(f"ERROR: No .tgz files found in {archive_path}")
        return

    print(f"Found {len(archive_files)} archives to extract.")

    # 3. Loop over each archive and extract it
    for archive_file in archive_files:
        filename = os.path.basename(archive_file)
        print(f"\nExtracting: {filename}")
        
        try:
            # Open the .tgz file for reading with gzip compression
            with tarfile.open(archive_file, "r:gz") as tar:
                # Get a list of all members (files) in the archive
                members = tar.getmembers()
                print(f"  ... Contains {len(members)} files (e.g., {members[0].name})")
                
                # Extract all members to the target directory
                tar.extractall(path=extract_path)
                print(f"  ... SUCCESS: Extracted all files to {extract_path}")
        
        except tarfile.ReadError as e:
            print(f"  ... ERROR: Failed to read archive {filename}. It might be corrupt. {e}")
        except Exception as e:
            print(f"  ... ERROR: An unexpected error occurred: {e}")

    print("\n--- Extraction complete! ---")
    print(f"All CSV files should be in: {os.path.abspath(extract_path)}")

if __name__ == "__main__":
    extract_all_archives(ARCHIVE_DIR, EXTRACT_DIR)