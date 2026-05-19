import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
from scipy.spatial.distance import pdist, squareform
import torch

# --- Configuration ---
# 1. Path to the sensor locations (GeoJSON)
GEOJSON_FILE = "Standorte_Verkehrsdetektion_Berlin.geojson"

# 2. Path to the folder containing your CSV files (the time-series data)
# We need this to know WHICH sensors to put in the graph.
# (Make sure this matches where you extracted your data)
CSV_DATA_DIR = os.path.join("berlin_traffic_data", "2023", "CSV_data")

# 3. Connection Threshold
# Sensors within this many meters will be connected.
DISTANCE_THRESHOLD_METERS = 2000 

# 4. Output file name
OUTPUT_FILE = "berlin_traffic_graph.pt"
# --- End of Configuration ---

def build_graph():
    print("--- Building Traffic Graph Topology ---")

    # --- Step 1: Identify Valid Sensors ---
    # We only want nodes in our graph if we actually have data for them.
    print(f"Scanning data folder: {CSV_DATA_DIR}")
    csv_files = glob.glob(os.path.join(CSV_DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("ERROR: No CSV files found! Check your CSV_DATA_DIR path.")
        return

    # Extract ID from filename: "TEU00425_Det2.csv" -> "TEU00425_Det2"
    available_sensor_ids = set()
    for f in csv_files:
        filename = os.path.basename(f)
        sensor_id = filename.replace(".csv", "")
        available_sensor_ids.add(sensor_id)
    
    print(f"Found {len(available_sensor_ids)} sensors with time-series data.")

    # --- Step 2: Load and Filter Locations ---
    print(f"Loading locations from {GEOJSON_FILE}...")
    if not os.path.exists(GEOJSON_FILE):
        print(f"ERROR: GeoJSON file not found at {GEOJSON_FILE}")
        return

    gdf = gpd.read_file(GEOJSON_FILE)
    
    # Filter: Keep only sensors that are in our CSV list
    gdf = gdf[gdf['teuID'].isin(available_sensor_ids)].copy()
    
    # CRITICAL STEP: Sort by ID
    # This ensures Row 0 of our matrix always corresponds to the same sensor.
    gdf = gdf.sort_values('teuID').reset_index(drop=True)
    
    print(f"Matched {len(gdf)} sensors in the location file.")

    # --- Step 3: Calculate Distances ---
    # Convert to a metric CRS (Coordinate Reference System) to measure meters.
    # EPSG:25833 is the standard UTM zone for Berlin.
    gdf = gdf.to_crs(epsg=25833)

    # Extract X, Y coordinates
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    
    # Calculate Euclidean distance between all pairs
    # pdist returns a condensed array, squareform makes it a matrix
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    
    print(f"Distance Matrix calculated. Shape: {dist_matrix.shape}")
    print(f"Closest pair: {dist_matrix[dist_matrix > 0].min():.1f}m")
    print(f"Furthest pair: {dist_matrix.max():.1f}m")

    # --- Step 4: Create Adjacency Matrix (Edges) ---
    # Create a matrix of 0s and 1s
    adj_matrix = np.zeros_like(dist_matrix)
    
    # Set to 1 where distance is less than threshold
    adj_matrix[dist_matrix < DISTANCE_THRESHOLD_METERS] = 1
    
    # Set diagonal to 1 (Self-loops: a sensor is connected to itself)
    # This is required for most GNN architectures.
    np.fill_diagonal(adj_matrix, 1)
    
    # Calculate sparsity
    num_edges = np.sum(adj_matrix)
    density = num_edges / (len(gdf) ** 2)
    print(f"Graph Density: {density:.2%} (Sensors interact locally)")

    # --- Step 5: Save to Disk ---
    # We save a dictionary containing:
    # 1. The Matrix (for the model)
    # 2. The IDs (to order the data later)
    # 3. The Coords (for plotting later)
    graph_data = {
        "adj_matrix": torch.tensor(adj_matrix, dtype=torch.float32),
        "sensor_ids": gdf['teuID'].tolist(),
        "coords": coords
    }
    
    torch.save(graph_data, OUTPUT_FILE)
    print(f"\nSUCCESS: Graph topology saved to '{OUTPUT_FILE}'")
    print("You can now proceed to data processing.")

if __name__ == "__main__":
    build_graph()