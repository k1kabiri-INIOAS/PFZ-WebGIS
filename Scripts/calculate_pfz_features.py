# File Name: calculate_pfz_features.py
# Directory: F:\PFZ\Scripts\calculate_pfz_features.py

import os
import numpy as np
import xarray as xr
import rioxarray
from scipy.ndimage import generic_filter

def calculate_gradient(arr):
    # Calculate local spatial variability (standard deviation or gradient magnitude)
    # to identify thermal fronts where SST changes rapidly.
    return np.nanstd(arr)

def main():
    # Define input and output paths based on the project structure
    input_nc = r"F:\PFZ\Data\Processed\clipped_sst.nc"
    output_dir = r"F:\PFZ\Data\Processed"
    output_nc = os.path.join(output_dir, "pfz_features.nc")
    
    print("Loading clipped SST dataset...")
    ds = xr.open_dataset(input_nc)
    
    # Identify the variable name for SST dynamically (usually 'analysed_sst' or 'sst')
    sst_var = [var for var in ds.data_vars if 'sst' in var.lower()][0]
    sst_data = ds[sst_var]
    
    print(f"Processing thermal gradients for variable: {sst_var}...")
    
    # Calculate spatial gradient (thermal fronts) using a moving window (e.g., 3x3)
    # High gradient values indicate potential frontal zones (PFZ hotspots)
    # Using scipy generic filter or xarray rolling operations for spatial filtering
    
    # For demonstration of vector/matrix operations using numpy/scipy:
    # We compute rolling standard deviation along lat/lon to locate sharp temperature changes
    rolling_std = sst_data.rolling(latitude=3, longitude=3, center=True).construct(
        latitude="lat_window", longitude="lon_window"
    ).std(dim=["lat_window", "lon_window"])
    
    # Add thermal front indicator back to the dataset
    ds["thermal_fronts"] = rolling_std
    
    # Save the processed features to a new NetCDF file
    ds.to_netcdf(output_nc)
    print(f"PFZ feature extraction complete. Saved to {output_nc}")

if __name__ == "__main__":
    main()