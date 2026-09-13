# File Name: processor.py
# Description: Module for multi-parameter PFZ modeling with Edge Detection & Vector Contouring (INCOIS Style).
# Test

import os
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import warnings

def process_pfz_pipeline(shapefile_path, sst_nc_path, chl_nc_path, output_dir, sst_weight=0.5, chl_weight=0.5):
    """
    Process datasets, compute PFZ Index, and perform Edge Detection to extract front contours.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gdf = gpd.read_file(shapefile_path)
    
    # 1. Process SST & Thermal Gradients
    ds_sst = xr.open_dataset(sst_nc_path).rio.write_crs("EPSG:4326", inplace=True)
    clipped_sst = ds_sst.rio.clip(gdf.geometry, gdf.crs, drop=False)
    sst_var = [var for var in clipped_sst.data_vars if 'sst' in var.lower()][0]
    sst_data = clipped_sst[sst_var]
    if 'time' in sst_data.dims:
        sst_data = sst_data.isel(time=-1)
        
    rolling_std = sst_data.rolling(latitude=3, longitude=3, center=True).construct(
        latitude="lat_window", longitude="lon_window"
    ).std(dim=["lat_window", "lon_window"])
    
    # 2. Process Chlorophyll-a
    ds_chl = xr.open_dataset(chl_nc_path).rio.write_crs("EPSG:4326", inplace=True)
    clipped_chl = ds_chl.rio.clip(gdf.geometry, gdf.crs, drop=False)
    chl_var = [var for var in clipped_chl.data_vars if 'chl' in var.lower() or 'chlorophyll' in var.lower()][0]
    chl_data = clipped_chl[chl_var]
    if 'time' in chl_data.dims:
        chl_data = chl_data.isel(time=0)
        
    # Normalization helper
    def normalize(da):
        min_val = float(da.min(skipna=True))
        max_val = float(da.max(skipna=True))
        if max_val - min_val == 0:
            return da * 0.0
        return (da - min_val) / (max_val - min_val + 1e-6)
        
    norm_fronts = normalize(rolling_std)
    norm_chl = normalize(chl_data)
    
    # 3. Multi-parameter weighted model
    total_weight = sst_weight + chl_weight
    if total_weight == 0:
        sst_weight, chl_weight = 0.5, 0.5
        total_weight = 1.0
        
    w_sst = sst_weight / total_weight
    w_chl = chl_weight / total_weight
    
    pfz_index = (w_sst * norm_fronts) + (w_chl * norm_chl)
    
    # =========================================================================
    # 4. EDGE DETECTION & CONTOURING (INCOIS STYLE)
    # =========================================================================
    lon_arr = pfz_index.longitude.values
    lat_arr = pfz_index.latitude.values
    pfz_arr = pfz_index.values
    
    fig, ax = plt.subplots()
    # Extract edges only for high probability zones (Top 35%)
    levels = [0.65, 0.75, 0.85] 
    cs = ax.contour(lon_arr, lat_arr, pfz_arr, levels=levels)
    
    lines = []
    for collection in cs.collections:
        for path in collection.get_paths():
            v = path.vertices
            if len(v) >= 2:
                lines.append(LineString(v))
    plt.close(fig)
    
    fronts_geojson_path = os.path.join(output_dir, "pfz_fronts.geojson")
    if lines:
        fronts_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fronts_gdf.to_file(fronts_geojson_path, driver="GeoJSON")
    else:
        # Create empty if no fronts detected
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(fronts_geojson_path, driver="GeoJSON")
    # =========================================================================
    
    # 5. Export Rasters
    out_ds = xr.Dataset(
        {
            "thermal_fronts": rolling_std,
            "chlorophyll": chl_data,
            "pfz_index": pfz_index
        },
        coords={"latitude": sst_data.latitude, "longitude": sst_data.longitude}
    )
    out_ds = out_ds.rio.write_crs("EPSG:4326", inplace=True)
    
    output_nc = os.path.join(output_dir, "final_pfz_output.nc")
    output_tif = os.path.join(output_dir, "final_pfz_output.tif")
    
    out_ds.to_netcdf(output_nc)
    out_ds["pfz_index"].rio.to_raster(output_tif)
    
    # Return 3 variables now (including the new vector GeoJSON path)
    return output_nc, output_tif, fronts_geojson_path