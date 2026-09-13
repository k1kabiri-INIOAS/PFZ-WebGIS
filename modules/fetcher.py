# File Name: fetcher.py
# Description: Module for fetching near real-time SST and Chlorophyll-a data from ERDDAP with robust fallback.

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from erddapy import ERDDAP

def fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir):
    """
    Fetch near real-time SST and Chlorophyll-a data dynamically with automated fallback support.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    end_date = datetime.utcnow() - timedelta(days=3)
    start_date = end_date - timedelta(days=8)
    
    time_start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    time_end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")
    
    print(f"Fetching Near Real-Time data from {time_start_str} to {time_end_str}...")
    
    sst_nc_path = os.path.join(output_dir, "raw_sst.nc")
    chl_nc_path = os.path.join(output_dir, "raw_chl.nc")
    
    # 1. Fetch SST Data
    try:
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
        
        url_sst = e_sst.get_download_url(response="nc")
        res_sst = requests.get(url_sst, timeout=30)
        res_sst.raise_for_status()
        with open(sst_nc_path, "wb") as f:
            f.write(res_sst.content)
        print("SST data downloaded successfully.")
    except Exception as e:
        print(f"[Warning] Live SST download failed: {e}. Using spatial fallback.")
        latitudes = np.linspace(miny, maxy, 100)
        longitudes = np.linspace(minx, maxx, 100)
        lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
        sst_vals = 28.0 - (lat_2d - miny) * 0.4 + np.sin(lon_2d * 0.1) * 1.2
        fallback_ds = xr.Dataset(
            {"analysed_sst": (["latitude", "longitude"], sst_vals)}, 
            coords={"latitude": latitudes, "longitude": longitudes}
        )
        fallback_ds.to_netcdf(sst_nc_path, engine="h5netcdf")

    # 2. Fetch Chlorophyll-a Data (with fallback)
    try:
        e_chl = ERDDAP(server="https://coastwatch.pfeg.noaa.gov/erddap/", protocol="griddap")
        e_chl.dataset_id = "erdVHNchla8day"
        e_chl.axis_names = {"longitude": "longitude", "latitude": "latitude", "time": "time", "altitude": "altitude"}
        e_chl.constraints.update({
            "time>=": time_start_str,
            "time<=": time_end_str,
            "altitude>=": "0.0",
            "altitude<=": "0.0",
            "latitude>=": str(miny),
            "latitude<=": str(maxy),
            "longitude>=": str(minx),
            "longitude<=": str(maxx),
        })
        url_chl = e_chl.get_download_url(response="nc")
        res_chl = requests.get(url_chl, timeout=30)
        res_chl.raise_for_status()
        with open(chl_nc_path, "wb") as f:
            f.write(res_chl.content)
        print("Chlorophyll-a data downloaded successfully.")
    except Exception as e:
        print(f"[Warning] Live Chlorophyll download failed: {e}. Using spatial fallback.")
        latitudes = np.linspace(miny, maxy, 100)
        longitudes = np.linspace(minx, maxx, 100)
        lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
        chl_vals = 0.5 + 0.3 * np.sin(np.radians(lat_2d)) * np.cos(np.radians(lon_2d))
        fallback_chl = xr.Dataset(
            {"chla": (["latitude", "longitude"], chl_vals)}, 
            coords={"latitude": latitudes, "longitude": longitudes}
        )
        fallback_chl.to_netcdf(chl_nc_path, engine="h5netcdf")
                   
    return sst_nc_path, chl_nc_path, time_end_str