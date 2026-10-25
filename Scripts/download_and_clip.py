# File Name: download_and_clip.py
# Directory: F:\PFZ\Scripts\download_and_clip.py

import os
import requests
import geopandas as gpd
import xarray as xr
import rioxarray
from erddapy import ERDDAP

def main():
    # Define paths based on the project structure
    shapefile_path = r"F:\PFZ\Shape_Files\SA.shp"
    output_dir = r"F:\PFZ\Data\Processed"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the unified IHO shapefile
    print("Loading the region shapefile...")
    gdf = gpd.read_file(shapefile_path)
    
    # Get the bounding box coordinates for spatial filtering during download
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Bounding Box -> Lon: [{minx}, {maxx}], Lat: [{miny}, {maxy}]")
    
    # 2. Connect to ERDDAP server to fetch Sea Surface Temperature (SST)
    print("Connecting to ERDDAP server...")
    e = ERDDAP(
        server="https://coastwatch.pfeg.noaa.gov/erddap/",
        protocol="griddap",
    )
    
    # Using the updated NOAA OISST v2.1 daily dataset ID
    e.dataset_id = "ncdcOisst21Agg"
    e.axis_names = {"longitude": "longitude", "latitude": "latitude", "time": "time"}
    
    # Update constraints safely using .update() to match erddapy internal checks
    e.constraints.update({
        "time>=": "2026-08-01T00:00:00Z",
        "time<=": "2026-08-01T00:00:00Z",
        "latitude>=": str(miny),
        "latitude<=": str(maxy),
        "longitude>=": str(minx),
        "longitude<=": str(maxx),
    })
    
    # Generate download URL and download via requests
    print("Downloading satellite data...")
    nc_file_path = os.path.join(output_dir, "raw_sst.nc")
    url = e.get_download_url(response="nc")
    
    # Fetch content and write to NetCDF file
    response = requests.get(url)
    response.raise_for_status()
    with open(nc_file_path, "wb") as f:
        f.write(response.content)
    print(f"Data downloaded successfully to {nc_file_path}")
    
    # 3. Open dataset with xarray and set spatial dimensions for rioxarray
    ds = xr.open_dataset(nc_file_path)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    
    # 4. Clip the raster data using the shapefile geometry
    print("Clipping data using the IHO shapefile mask...")
    clipped_ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)
    
    # Save the processed/clipped dataset
    output_nc = os.path.join(output_dir, "clipped_sst.nc")
    clipped_ds.to_netcdf(output_nc)
    print(f"Clipped dataset saved to {output_nc}")

if __name__ == "__main__":
    main()