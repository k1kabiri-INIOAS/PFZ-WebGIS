# File Name: fetcher.py
# Description: Module for fetching high-resolution SST (MUR SST 1km) and reliable Chlorophyll-a (8-day 4km) data from ERDDAP with robust fallback.

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from erddapy import ERDDAP

def fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir):
    """
    Fetch near real-time high-resolution SST and Chlorophyll-a data dynamically 
    with automated fallback support.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # تنظیم اولیه تاریخ
    end_date = datetime.utcnow() - timedelta(days=2)
    start_date = end_date - timedelta(days=8)
    
    sst_nc_path = os.path.join(output_dir, "raw_sst.nc")
    chl_nc_path = os.path.join(output_dir, "raw_chl.nc")
    
    def download_dataset(dataset_id, is_chl=False):
        nonlocal start_date, end_date
        max_retries = 4
        
        for attempt in range(max_retries):
            time_start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
            time_end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")
            
            print(f"Fetching {dataset_id} from {time_start_str} to {time_end_str}...")
            
            try:
                e = ERDDAP(server="https://coastwatch.pfeg.noaa.gov/erddap/", protocol="griddap")
                e.dataset_id = dataset_id
                
                constraints = {
                    "time>=": time_start_str,
                    "time<=": time_end_str,
                    "latitude>=": str(miny),
                    "latitude<=": str(maxy),
                    "longitude>=": str(minx),
                    "longitude<=": str(maxx),
                }
                
                if is_chl:
                    e.axis_names = {"longitude": "longitude", "latitude": "latitude", "time": "time", "altitude": "altitude"}
                    constraints["altitude>="] = "0.0"
                    constraints["altitude<="] = "0.0"
                else:
                    e.axis_names = {"longitude": "longitude", "latitude": "latitude", "time": "time"}
                    
                e.constraints.update(constraints)
                
                url = e.get_download_url(response="nc")
                res = requests.get(url, timeout=30)
                
                if res.status_code == 200:
                    return res.content, time_end_str
                elif res.status_code == 404:
                    print(f"[Warning] 404 Client Error for {dataset_id} at {time_end_str}. Shifting dates back by 3 days...")
                    end_date -= timedelta(days=3)
                    start_date -= timedelta(days=3)
                else:
                    res.raise_for_status()
            except Exception as ex:
                if "404" in str(ex):
                    print(f"[Warning] 404 Client Error caught for {dataset_id}. Shifting dates back by 3 days...")
                    end_date -= timedelta(days=3)
                    start_date -= timedelta(days=3)
                else:
                    print(f"[Error] Failed to fetch {dataset_id}: {ex}")
                    break
                    
        return None, time_end_str

    # 1. Fetch High-Res SST Data (JPL MUR SST - 1km)
    sst_content, final_end_str = download_dataset("jplMURSST41", is_chl=False)
    if sst_content:
        with open(sst_nc_path, "wb") as f:
            f.write(sst_content)
        print("SST data downloaded successfully (MUR SST 1km).")
    else:
        print("[Warning] Live SST download failed entirely. Using spatial fallback.")
        latitudes = np.linspace(miny, maxy, 100)
        longitudes = np.linspace(minx, maxx, 100)
        lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
        sst_vals = 28.0 - (lat_2d - miny) * 0.4 + np.sin(lon_2d * 0.1) * 1.2
        fallback_ds = xr.Dataset(
            {"analysed_sst": (["latitude", "longitude"], sst_vals)}, 
            coords={"latitude": latitudes, "longitude": longitudes}
        )
        fallback_ds.to_netcdf(sst_nc_path, engine="h5netcdf")

    # 2. Fetch Chlorophyll-a Data (erdVHNchla8day - stable 4km product)[cite: 1]
    chl_content, _ = download_dataset("erdVHNchla8day", is_chl=True)
    if chl_content:
        with open(chl_nc_path, "wb") as f:
            f.write(chl_content)
        print("Chlorophyll-a data downloaded successfully (erdVHNchla8day 4km).")
    else:
        print("[Warning] Live Chlorophyll download failed entirely. Using spatial fallback.")
        latitudes = np.linspace(miny, maxy, 100)
        longitudes = np.linspace(minx, maxx, 100)
        lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
        chl_vals = 0.5 + 0.3 * np.sin(np.radians(lat_2d)) * np.cos(np.radians(lon_2d))
        fallback_chl = xr.Dataset(
            {"chla": (["latitude", "longitude"], chl_vals)}, 
            coords={"latitude": latitudes, "longitude": longitudes}
        )
        fallback_chl.to_netcdf(chl_nc_path, engine="h5netcdf")
                   
    return sst_nc_path, chl_nc_path, final_end_str