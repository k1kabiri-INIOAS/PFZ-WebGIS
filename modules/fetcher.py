# File Name: fetcher.py
# Description: Near-realtime SST, Chlorophyll-a, and optional geostrophic currents / SSH from ERDDAP.

import os
import requests
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, timedelta
from erddapy import ERDDAP


def fetch_near_realtime_data(
    minx,
    miny,
    maxx,
    maxy,
    output_dir,
    fetch_current=False,
    fetch_ssh=False,
):
    """
    Fetch near real-time ocean data from ERDDAP.

    Returns dict:
      {
        "sst_nc_path": str,
        "chl_nc_path": str,
        "current_nc_path": str | None,
        "ssh_nc_path": str | None,
        "latest_date": str,
        "fetched": {"sst": bool, "chl": bool, "current": bool, "ssh": bool},
      }
    """
    os.makedirs(output_dir, exist_ok=True)

    end_date = datetime.utcnow() - timedelta(days=2)
    start_date = end_date - timedelta(days=8)

    sst_nc_path = os.path.join(output_dir, "raw_sst.nc")
    chl_nc_path = os.path.join(output_dir, "raw_chl.nc")
    current_nc_path = os.path.join(output_dir, "raw_current.nc")
    ssh_nc_path = os.path.join(output_dir, "raw_ssh.nc")
    latest_date_str = end_date.strftime("%Y-%m-%d")
    fetched = {"sst": False, "chl": False, "current": False, "ssh": False}

    # --- 1. SST (MUR 1km) — همیشه ---
    sst_content, sst_date = _erddap_download(
        server="https://coastwatch.pfeg.noaa.gov/erddap/",
        dataset_id="jplMURSST41",
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        start_date=start_date,
        end_date=end_date,
        is_chl=False,
        step_days=3,
    )
    if sst_content:
        with open(sst_nc_path, "wb") as f:
            f.write(sst_content)
        print("SST data downloaded successfully (MUR SST 1km).")
        latest_date_str = sst_date
        fetched["sst"] = True
    else:
        print("[Warning] Live SST download failed. Generating spatial fallback...")
        _write_fallback_sst(sst_nc_path, minx, miny, maxx, maxy)
        fetched["sst"] = False

    # --- 2. Chlorophyll-a — همیشه ---
    chl_content, _ = _erddap_download(
        server="https://coastwatch.pfeg.noaa.gov/erddap/",
        dataset_id="erdVHNchla8day",
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        start_date=start_date,
        end_date=end_date,
        is_chl=True,
        step_days=8,
    )
    if chl_content:
        with open(chl_nc_path, "wb") as f:
            f.write(chl_content)
        print("Chlorophyll-a data downloaded successfully (erdVHNchla8day).")
        fetched["chl"] = True
    else:
        print("[Warning] Live Chlorophyll download failed. Generating spatial fallback...")
        _write_fallback_chl(chl_nc_path, minx, miny, maxx, maxy)
        fetched["chl"] = False

    # --- 3. جریان ژئوستروفیک (اختیاری) ---
    if fetch_current:
        cur_content, _ = _erddap_download(
            server="https://oceanwatch.pfeg.noaa.gov/erddap/",
            dataset_id="noaacwBLENDEDNRTcurrentsDaily",
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            start_date=start_date,
            end_date=end_date,
            is_chl=False,
            step_days=3,
            timeout=90,
        )
        if not cur_content:
            cur_content, _ = _erddap_download(
                server="https://oceanwatch.pfeg.noaa.gov/erddap/",
                dataset_id="noaa_aoml_1db3_9713_eaa7",
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
                start_date=start_date,
                end_date=end_date,
                is_chl=False,
                step_days=3,
                timeout=90,
            )
        if cur_content:
            with open(current_nc_path, "wb") as f:
                f.write(cur_content)
            print("Geostrophic current data downloaded successfully.")
            fetched["current"] = True
        else:
            print("[Warning] Live current download failed. Continuing without currents.")
            current_nc_path = None
            fetched["current"] = False
    else:
        current_nc_path = None

    # --- 4. SSH anomaly (اختیاری) ---
    if fetch_ssh:
        ssh_content, _ = _erddap_download(
            server="https://oceanwatch.pfeg.noaa.gov/erddap/",
            dataset_id="nesdisSSH1day",
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            start_date=start_date,
            end_date=end_date,
            is_chl=False,
            step_days=3,
            timeout=90,
        )
        if ssh_content:
            with open(ssh_nc_path, "wb") as f:
                f.write(ssh_content)
            print("SSH anomaly data downloaded successfully.")
            fetched["ssh"] = True
        else:
            print("[Warning] Live SSH download failed. Continuing without SSH.")
            ssh_nc_path = None
            fetched["ssh"] = False
    else:
        ssh_nc_path = None

    return {
        "sst_nc_path": sst_nc_path,
        "chl_nc_path": chl_nc_path,
        "current_nc_path": current_nc_path if fetch_current and fetched["current"] else None,
        "ssh_nc_path": ssh_nc_path if fetch_ssh and fetched["ssh"] else None,
        "latest_date": latest_date_str,
        "fetched": fetched,
    }


def _erddap_download(
    server,
    dataset_id,
    minx,
    miny,
    maxx,
    maxy,
    start_date,
    end_date,
    is_chl=False,
    step_days=3,
    timeout=60,
):
    cur_start, cur_end = start_date, end_date
    max_retries = 5
    for _ in range(max_retries):
        time_start_str = cur_start.strftime("%Y-%m-%dT00:00:00Z")
        time_end_str = cur_end.strftime("%Y-%m-%dT00:00:00Z")
        print(f"Fetching {dataset_id} from {time_start_str} to {time_end_str}...")
        try:
            e = ERDDAP(server=server, protocol="griddap")
            e.dataset_id = dataset_id
            if is_chl:
                e.axis_names = {
                    "longitude": "longitude",
                    "latitude": "latitude",
                    "time": "time",
                    "altitude": "altitude",
                }
                constraints = {
                    "time>=": time_start_str,
                    "time<=": time_end_str,
                    "latitude>=": str(miny),
                    "latitude<=": str(maxy),
                    "longitude>=": str(minx),
                    "longitude<=": str(maxx),
                    "altitude>=": "0.0",
                    "altitude<=": "0.0",
                }
            else:
                e.axis_names = {
                    "longitude": "longitude",
                    "latitude": "latitude",
                    "time": "time",
                }
                constraints = {
                    "time>=": time_start_str,
                    "time<=": time_end_str,
                    "latitude>=": str(miny),
                    "latitude<=": str(maxy),
                    "longitude>=": str(minx),
                    "longitude<=": str(maxx),
                }
            e.constraints.update(constraints)
            url = e.get_download_url(response="nc")
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                return res.content, cur_end.strftime("%Y-%m-%d")
            if res.status_code == 404:
                print(f"[Warning] 404 for {dataset_id}. Shifting back by {step_days} days...")
                cur_end -= timedelta(days=step_days)
                cur_start -= timedelta(days=step_days)
            else:
                res.raise_for_status()
        except Exception as ex:
            print(f"[Warning] Retrying {dataset_id} due to: {ex}")
            cur_end -= timedelta(days=step_days)
            cur_start -= timedelta(days=step_days)
    return None, end_date.strftime("%Y-%m-%d")


def _write_fallback_sst(path, minx, miny, maxx, maxy):
    latitudes = np.linspace(miny, maxy, 100)
    longitudes = np.linspace(minx, maxx, 100)
    lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
    sst_vals = 28.0 - (lat_2d - miny) * 0.4 + np.sin(lon_2d * 0.1) * 1.2
    fallback_ds = xr.Dataset(
        {"analysed_sst": (["latitude", "longitude"], sst_vals)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    fallback_ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    fallback_ds.rio.write_crs("EPSG:4326", inplace=True)
    fallback_ds.to_netcdf(path, engine="h5netcdf")


def _write_fallback_chl(path, minx, miny, maxx, maxy):
    latitudes = np.linspace(miny, maxy, 100)
    longitudes = np.linspace(minx, maxx, 100)
    lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
    chl_vals = 0.5 + 0.3 * np.sin(np.radians(lat_2d)) * np.cos(np.radians(lon_2d))
    fallback_chl = xr.Dataset(
        {"chla": (["latitude", "longitude"], chl_vals)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    fallback_chl.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    fallback_chl.rio.write_crs("EPSG:4326", inplace=True)
    fallback_chl.to_netcdf(path, engine="h5netcdf")
