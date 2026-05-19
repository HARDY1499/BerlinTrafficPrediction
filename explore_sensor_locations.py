import geopandas as gpd
import os

# --- Configuration ---
# This is the link you downloaded from the Berlin Open Data portal.
# I'm assuming a common filename, PLEASE UPDATE THIS
# if your local file is named differently.
GEOJSON_FILE = "Standorte_Verkehrsdetektion_Berlin.geojson" 
# (The link was '02371a53-0c21-4b2a-a53b-0164e29241f1.download'
#  you may need to rename it to .geojson or check its contents)

# --- End of Configuration ---

def explore_locations_geojson(filepath):
    """
    Loads the GeoJSON file with sensor locations and
    prints a summary.
    """
    print(f"--- Sensor Location Explorer ---")
    
    # 1. Check if the file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at '{filepath}'")
        print("Please download the GeoJSON file for sensor locations,")
        print("place it in this directory, and update the")
        print("GEOJSON_FILE variable in this script to match the filename.")
        return

    print(f"Loading locations file: {filepath}")

    # 2. Load the GeoJSON file using geopandas
    try:
        # GeoPandas can read GeoJSON files directly
        gdf = gpd.read_file(filepath)
        
    except Exception as e:
        print(f"ERROR: Could not load the GeoJSON file.")
        print("Make sure you have 'geopandas' installed: pip install geopandas")
        print(f"Details: {e}")
        return

    # 3. Print the summary
    print("\n--- 1. GeoDataFrame Head (First 5 Rows) ---")
    print(gdf.head())

    print("\n--- 2. GeoDataFrame Info (Columns & Data Types) ---")
    # .info() will show us all the columns
    gdf.info()

    print("\n--- 3. GeoDataFrame Columns List ---")
    print(gdf.columns.to_list())
    
    print("\n--- 4. Coordinate Reference System (CRS) ---")
    # This tells us what map projection the GPS data is in.
    print(gdf.crs)

    print("\n--- 5. Identifying Key Columns (Educated Guess) ---")
    columns = gdf.columns.str.lower()
    
    # Guess for Sensor ID (Node ID)
    id_col = None
    if 'messstelle_id' in columns:
        id_col = 'messstelle_id'
    elif 'detektor_id' in columns:
        id_col = 'detektor_id'
    elif 'external_id' in columns:
        id_col = 'external_id'
    elif 'name' in columns:
        id_col = 'name'
        
    if id_col:
        print(f"  > Sensor ID Column (Guess): '{id_col}'")
        print(f"    Example ID from this file: {gdf.iloc[0][id_col]}")
        print(f"    We need this to match our CSV filenames (e.g., 'TEU00425_Det2')")
    else:
        print("  > Sensor ID Column (Guess): Not found. We will need to look manually.")

    # The 'geometry' column is special in GeoPandas
    if 'geometry' in columns:
        print(f"  > Geometry Column: 'geometry' (This contains the GPS coordinates)")
        print(f"    Example Coordinate: {gdf.iloc[0]['geometry']}")
    else:
        print("  > Geometry Column: Not found. This is a problem.")


if __name__ == "__main__":
    explore_locations_geojson(GEOJSON_FILE)
# ```

# ### Your Next Steps:

# 1.  **Find the file:** Locate the sensor location file you downloaded earlier (from the link `.../02371a53-0c21-4b2a-a53b-0164e29241f1/download`).
# 2.  **Rename it:** Rename that file to `Standorte_Verkehrsdetektion_Berlin.geojson` and place it in the same folder as this new script.
# 3.  **Install `geopandas`:** This is a new, essential library for this.
#     ```bash
#     pip install geopandas
#     ```
# 4.  **Run the script:**
#     ```bash
#     python explore_sensor_locations.py