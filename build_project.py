# File Name: build_project.py
# Description: Automatic builder script to setup and expand the PFZ Streamlit project structure.

import os

def create_project_structure():
    base_dir = r"F:\PFZ"
    
    # Define required directory paths
    folders = [
        os.path.join(base_dir, "modules"),
        os.path.join(base_dir, "Data", "Processed"),
        os.path.join(base_dir, "Shape_Files")
    ]
    
    # Create directories if they do not exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[Directory Created] {folder}")
        
    # Define project files and their source code (with English comments)
    files = {
        os.path.join(base_dir, "requirements.txt"): """streamlit
xarray
rioxarray
geopandas
erddapy
requests
numpy
matplotlib
""",
        
        os.path.join(base_dir, "modules", "__init__.py"): '''# Package initialization for modules
''',
        
        os.path.join(base_dir, "modules", "fetcher.py"): '''# File Name: fetcher.py
# Description: Module for fetching near real-time oceanographic data from ERDDAP server.

import os
import requests
from datetime import datetime, timedelta
from erddapy import ERDDAP

def fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir):
    """
    Fetch near real-time SST data dynamically based on bounding box coordinates 
    and recent date ranges.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate dynamic near real-time date range (past 8 days up to current time)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=8)
    
    time_start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    time_end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")
    
    print(f"Fetching Near Real-Time data from {time_start_str} to {time_end_str}...")
    
    # Connect to ERDDAP server for NOAA OISST v2.1 dataset
    e_sst = ERDDAP(server="https://coastwatch.pfeg.noaa.gov/erddap/", protocol="griddap")
    e_sst.dataset_id = "ncdcOisst21Agg"
    e_sst.axis_names = {"longitude": "longitude", "latitude": "latitude", "time": "time"}
    e_sst.constraints.update({
        "time>=": time_start_str,
        "time<=": time_end_str,
        "latitude>=": str(miny),
        "latitude<=": str(maxy),
        "longitude>=": str(minx),
        "longitude<=": str(maxx),
    })
    
    sst_nc_path = os.path.join(output_dir, "raw_sst.nc")
    try:
        url_sst = e_sst.get_download_url(response="nc")
        res_sst = requests.get(url_sst)
        res_sst.raise_for_status()
        with open(sst_nc_path, "wb") as f:
            f.write(res_sst.content)
        print("SST data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading live SST: {e}")
        return None, None
        
    return sst_nc_path, time_end_str
''',

        os.path.join(base_dir, "modules", "processor.py"): '''# File Name: processor.py
# Description: Module for spatial clipping, thermal front detection, and PFZ index modeling.

import os
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd

def process_pfz_pipeline(shapefile_path, sst_nc_path, output_dir):
    """
    Process SST dataset, apply spatial masking, compute thermal fronts, 
    and export standardized NetCDF and GeoTIFF outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load geographic shapefile mask
    gdf = gpd.read_file(shapefile_path)
    
    # Open raw SST dataset
    ds = xr.open_dataset(sst_nc_path)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    
    # Clip dataset using shapefile geometry
    clipped_ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)
    
    # Identify SST variable name dynamically
    sst_var = [var for var in clipped_ds.data_vars if 'sst' in var.lower()][0]
    sst_data = clipped_ds[sst_var]
    
    if 'time' in sst_data.dims:
        sst_data = sst_data.isel(time=-1)  # Select the latest time step
        
    # Calculate spatial thermal fronts using rolling standard deviation
    rolling_std = sst_data.rolling(latitude=3, longitude=3, center=True).construct(
        latitude="lat_window", longitude="lon_window"
    ).std(dim=["lat_window", "lon_window"])
    
    # Min-max normalization helper function
    def normalize(da):
        min_val = float(da.min())
        max_val = float(da.max())
        if max_val - min_val == 0:
            return da * 0.0
        return (da - min_val) / (max_val - min_val + 1e-6)
        
    norm_fronts = normalize(rolling_std)
    
    # Build final processed dataset
    out_ds = xr.Dataset(
        {
            "thermal_fronts": rolling_std,
            "pfz_index": norm_fronts
        },
        coords={"latitude": sst_data.latitude, "longitude": sst_data.longitude}
    )
    out_ds = out_ds.rio.write_crs("EPSG:4326", inplace=True)
    
    output_nc = os.path.join(output_dir, "final_pfz_output.nc")
    output_tif = os.path.join(output_dir, "final_pfz_output.tif")
    
    out_ds.to_netcdf(output_nc)
    out_ds["pfz_index"].rio.to_raster(output_tif)
    
    return output_nc, output_tif
''',

        os.path.join(base_dir, "app.py"): '''# File Name: app.py
# Description: Streamlit GUI main application file for PFZ management system.

import os
import streamlit as st
import geopandas as gpd
from modules.fetcher import fetch_near_realtime_data
from modules.processor import process_pfz_pipeline

st.set_page_config(page_title="PFZ Management System", layout="wide")

st.title("🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ)")
st.markdown("این سامانه با استفاده از داده‌های ماهواره‌ای به‌روز (Near Real-Time) و پردازش‌های مکانی، مناطق مستعد صید را شناسایی می‌کند.")

# Sidebar configuration inputs
st.sidebar.header("تنظیمات پردازش")
shapefile_path = st.sidebar.text_input("مسیر شیپ‌فایل منطقه:", r"F:\\PFZ\\Shape_Files\\SA.shp")
output_dir = st.sidebar.text_input("مسیر ذخیره خروجی‌ها:", r"F:\\PFZ\\Data\\Processed")

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if not os.path.exists(shapefile_path):
        st.error("فایل شیپ‌فایل یافت نشد لطفا مسیر را بررسی کنید.")
    else:
        with st.spinner("در حال اتصال به سرور و دریافت داده‌های به‌روز..."):
            gdf = gpd.read_file(shapefile_path)
            minx, miny, maxx, maxy = gdf.total_bounds
            
            # Fetch near real-time data dynamically
            sst_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
            
            if sst_nc_path:
                st.success(f"داده‌های تاریخ {latest_date} با موفقیت دریافت شد.")
                
                with st.spinner("در حال محاسبه گرادیان‌های حرارتی و تولید لایه نهایی PFZ..."):
                    nc_out, tif_out = process_pfz_pipeline(shapefile_path, sst_nc_path, output_dir)
                    
                st.success("پردازش و تحلیل چندمتواره با موفقیت به پایان رسید!")
                st.info(f"فایل رستر نهایی در مسیر زیر ذخیره شد:\\n`{tif_out}`")
            else:
                st.error("خطا در دریافت داده‌های ماهواره‌ای از سرور.")
'''
    }
    
    # Write files with proper encoding
    for file_path, content in files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"[File Created] {file_path}")
        
    print("\n[Success] All project files and directories have been successfully generated!")

if __name__ == "__main__":
    create_project_structure()