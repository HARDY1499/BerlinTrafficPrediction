import pandas as pd
import glob
import os

# --- Configuration ---
# This MUST match the EXTRACT_DIR from your previous script
EXTRACT_DIR = os.path.join("berlin_traffic_data", "2023", "CSV_data")

# We'll try to find any .csv file in that directory
# Note: The data source notes a semicolon (;) as the separator.
CSV_SEPARATOR = ';'
# --- End of Configuration ---

def explore_first_csv(extract_path):
    """
    Finds the first .csv file in the extract_path, loads it into
    pandas, and prints a comprehensive summary.
    """
    print(f"--- CSV Data Explorer ---")
    print(f"Looking for CSV files in: {extract_path}")

    # 1. Find the first CSV file in the directory
    # glob.glob finds all files matching a pattern
    # We'll use recursive=True just in case the .tgz
    # extracted files into a subdirectory.
    try:
        csv_files = glob.glob(os.path.join(extract_path, "**", "*.csv"), recursive=True)
        if not csv_files:
            print(f"ERROR: No .csv files were found in {extract_path} or its subdirectories.")
            print("Please check the EXTRACT_DIR variable in the script.")
            return

        first_file = csv_files[0]
        print(f"\nFound {len(csv_files)} CSV files. Loading the first one for analysis:")
        print(f"  {first_file}")

    except Exception as e:
        print(f"ERROR: An error occurred while searching for files: {e}")
        return

    # 2. Load the CSV file into a pandas DataFrame
    try:
        # We use low_memory=False as these can be large files,
        # which helps pandas load them without guessing data types
        # multiple times.
        df = pd.read_csv(first_file, sep=CSV_SEPARATOR, low_memory=False)
        
    except FileNotFoundError:
        print(f"ERROR: File not found at {first_file}. Something went wrong.")
        return
    except pd.errors.ParserError as e:
        print(f"ERROR: Pandas couldn't parse the file. Is the separator '{CSV_SEPARATOR}' correct?")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # 3. Print the summary
    print("\n--- 1. DataFrame Head (First 5 Rows) ---")
    print(df.head())

    print("\n--- 2. DataFrame Info (Columns & Data Types) ---")
    # .info() is the most useful command here. It tells us
    # column names, the count of non-null values, and the data type.
    df.info()

    print("\n--- 3. Statistical Summary (For Numeric Columns) ---")
    # .describe() gives you min, max, mean, std, and quartiles
    # for all numeric columns. This is great for sanity checks
    # (e.g., is "speed" really between 0 and 200?).
    print(df.describe())

    print("\n--- 4. Identifying Key Columns (Educated Guess) ---")
    
    # Let's try to guess the key columns
    columns = df.columns.str.lower()
    
    # Guess for Sensor ID (Node ID)
    id_col = None
    if 'messstelle' in columns:
        id_col = 'messstelle' # German for "measuring point"
    elif 'detektor' in columns:
        id_col = 'detektor'
    elif 'id' in columns:
        id_col = 'id'
        
    if id_col:
        print(f"  > Sensor ID Column (Guess): '{id_col}'")
        num_sensors = df[id_col].nunique()
        print(f"    This file contains data for {num_sensors} unique sensors.")
    else:
        print("  > Sensor ID Column (Guess): Not found (Look for 'messstelle' or 'detektor')")

    # Guess for Timestamp
    time_col = None
    if 'messzeit' in columns:
        time_col = 'messzeit' # German for "measuring time"
    elif 'timestamp' in columns:
        time_col = 'timestamp'
        
    if time_col:
        print(f"  > Timestamp Column (Guess): '{time_col}'")
        # Convert to datetime to get min/max
        df[time_col] = pd.to_datetime(df[time_col])
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        print(f"    Data ranges from: {min_time} to {max_time}")
        
        # Try to find the time interval
        if len(df[time_col]) > 1:
            interval = (df[time_col].iloc[1] - df[time_col].iloc[0]).total_seconds() / 60
            print(f"    Appears to be {interval}-minute intervals (if data is sorted)")
    else:
        print("  > Timestamp Column (Guess): Not found (Look for 'messzeit' or 'timestamp')")

    # Guess for Features
    print("  > Feature Columns (Guess): Look for 'geschwindigkeit' (speed), 'anzahl_kfz' (num_vehicles), 'anzahl_pkw', 'anzahl_lkw'")


if __name__ == "__main__":
    explore_first_csv(EXTRACT_DIR)


### How to Use:

# 1.  Save the code above as `explore_traffic_csv.py`.
# 2.  Make sure you have `pandas` installed: `pip install pandas`
# 3.  Run the script from your terminal:
#     ```bash
#     python explore_traffic_csv.py