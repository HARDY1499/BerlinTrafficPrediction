import requests
import os

# --- Configuration ---
# Set the year you want to download
TARGET_YEAR = "2024"  # Using 2023 as it's a full year

# This is the path inside the Azure container
DATA_PATH = "neue_qualitaetssicherung/Fahrstreifendetektoren"

# This is the new base URL for the Azure Blob Storage
BASE_URL = f"https://mdhopendata.blob.core.windows.net/verkehrsdetektion/{TARGET_YEAR}/{DATA_PATH}/"

# Local directory where .tgz files will be saved
# I've added '_tgz' to the name to make it clear these are the archives
DOWNLOAD_DIR = os.path.join("berlin_traffic_data", TARGET_YEAR, "Fahrstreifendetektoren_tgz")
# --- End of Configuration ---

def download_monthly_archives(base_url, download_path, year):
    """
    Downloads monthly .tgz archives for the target year by
    constructing the URL for each month.
    """
    print(f"--- Berlin Traffic Data Downloader (v3) ---")
    print(f"Targeting: {base_url}")
    print(f"Saving to: {download_path}")
    print(f"Target Year: {year}")

    # 1. Create the local directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # 2. Loop through all 12 months
    for month in range(1, 13):
        # Format month as 2 digits (e.g., 1 -> "01", 12 -> "12")
        month_str = f"{month:02d}"
        
        # Construct the filename and the full URL
        filename = f"detektoren_{year}_{month_str}.tgz"
        file_url = f"{base_url}{filename}"
        
        local_file_path = os.path.join(download_path, filename)
        
        print(f"\nAttempting to download: {filename}")

        # Check if file already exists to avoid re-downloading
        if os.path.exists(local_file_path):
            file_size = os.path.getsize(local_file_path)
            print(f"  ... SKIPPING: File already exists at {local_file_path} ({file_size} bytes)")
            continue

        # 3. Download the file
        try:
            with requests.get(file_url, stream=True) as r:
                # Check for HTTP errors (like 404 Not Found)
                r.raise_for_status()
                
                # Open the local file in 'write binary' mode
                with open(local_file_path, 'wb') as f:
                    # Write file in chunks to avoid using too much memory
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            
            file_size = os.path.getsize(local_file_path)
            print(f"  ... SUCCESS: Saved to {local_file_path} ({file_size} bytes)")
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ... FAILED: File not found (404). This month may not be available yet.")
            else:
                print(f"  ... FAILED: HTTP Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"  ... FAILED: A network error occurred: {e}")
        except IOError as e:
            print(f"  ... FAILED: Could not write file to disk: {e}")

    print("\n--- Download complete! ---")
    print(f"All archive files are in: {os.path.abspath(download_path)}")

# This makes the script runnable from the command line
if __name__ == "__main__":
    download_monthly_archives(BASE_URL, DOWNLOAD_DIR, TARGET_YEAR)