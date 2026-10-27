# File Name: fetcher.py
# Description: Near-realtime SST, Chlorophyll-a, optional currents & SSH from ERDDAP.

import os
import requests
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, timedelta, timezone
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
    Returns dict with paths and fetch status.
    Currents/SSH are optional; failure does not block SST/Chl.
    """
    os.makedirs(output_dir, exist_ok=True)

    end_date = datetime.now(timezone.utc) - timedelta(days=2)
    start_date = end_date - timedelta(days=8)

    sst_nc_path = os.path.join(output_dir, "raw_sst.nc")
    chl_nc_path = os.path.join(output_dir, "raw_chl.nc")
    current_nc_path = os.path.join(output_dir, "raw_current.nc")
    ssh_nc_path = os.path.join(output_dir, "raw_ssh.nc")
    latest_date_str = end_date.strftime("%Y-%m-%d")
    fetched = {"sst": False, "chl": False, "current": False, "ssh": False}

    # --- 1. SST (MUR 1km) ---
    sst_content, sst_date = _erddap_download(
        server="https://coastwatch.pfeg.noaa.gov/erddap/",
        dataset_id="jplMURSST41",
        minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        start_date=start_date, end_date=end_date,
        is_chl=False, step_days=3,
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

    # --- 2. Chlorophyll-a ---
    # erdVHNchla8day اغلب برای این محدوده 404 می‌دهد؛ اول erdMH1chla1day
    chl_content = _download_chlorophyll(minx, miny, maxx, maxy)
    if chl_content:
        with open(chl_nc_path, "wb") as f:
            f.write(chl_content)
        print("Chlorophyll-a data downloaded successfully.")
        fetched["chl"] = True
    else:
        # fallback قدیمی با altitude
        chl_content, _ = _erddap_download(
            server="https://coastwatch.pfeg.noaa.gov/erddap/",
            dataset_id="erdVHNchla8day",
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            start_date=start_date, end_date=end_date,
            is_chl=True, step_days=8,
        )
        if chl_content:
            with open(chl_nc_path, "wb") as f:
                f.write(chl_content)
            print("Chlorophyll-a downloaded via erdVHNchla8day fallback.")
            fetched["chl"] = True
        else:
            print("[Warning] Live Chlorophyll download failed. Generating spatial fallback...")
            _write_fallback_chl(chl_nc_path, minx, miny, maxx, maxy)

    # --- 3+4. Currents / SSH ---
    # بهترین منبع ترکیبی: nesdisSSH1day روی coastwatch.pfeg
    # شامل sla + ugos + vgos → هم جریان هم SSH از یک فایل
    if fetch_current or fetch_ssh:
        combo = _download_nesdis_ssh_currents(minx, miny, maxx, maxy, output_dir)
        if combo:
            if fetch_ssh and combo.get("ssh_path"):
                ssh_nc_path = combo["ssh_path"]
                fetched["ssh"] = True
                print("SSH (sla) downloaded via nesdisSSH1day.")
            if fetch_current and combo.get("current_path"):
                current_nc_path = combo["current_path"]
                fetched["current"] = True
                print("Currents (ugos/vgos) downloaded via nesdisSSH1day.")

        # Fallback جریان: coastwatch.noaa.gov blended currents
        if fetch_current and not fetched["current"]:
            cur = _download_blended_currents(minx, miny, maxx, maxy, current_nc_path)
            if cur:
                fetched["current"] = True
                print("Currents downloaded via noaacwBLENDEDNRTcurrentsDaily.")
            else:
                print("[Warning] Live current download failed. Continuing without currents.")
                current_nc_path = None

        if fetch_ssh and not fetched["ssh"]:
            print("[Warning] Live SSH download failed. Continuing without SSH.")
            ssh_nc_path = None
    else:
        current_nc_path = None
        ssh_nc_path = None

    return {
        "sst_nc_path": sst_nc_path,
        "chl_nc_path": chl_nc_path,
        "current_nc_path": current_nc_path if fetch_current and fetched["current"] else None,
        "ssh_nc_path": ssh_nc_path if fetch_ssh and fetched["ssh"] else None,
        "latest_date": latest_date_str,
        "fetched": fetched,
    }


def _download_nesdis_ssh_currents(minx, miny, maxx, maxy, output_dir):
    """
    nesdisSSH1day: sla, ugos, vgos — روی coastwatch.pfeg.noaa.gov
    از (last) و چند روز عقب‌تر استفاده می‌کند.
    """
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day.nc"
    # کمی حاشیه برای شبکهٔ درشت
    minx2, maxx2 = minx - 0.3, maxx + 0.3
    miny2, maxy2 = miny - 0.3, maxy + 0.3

    attempts = ["(last)"]
    now = datetime.now(timezone.utc)
    for days_ago in (3, 5, 8, 12, 18):
        t = (now - timedelta(days=days_ago)).strftime("%Y-%m-%dT00:00:00Z")
        attempts.append(f"[({t})]")

    content = None
    for texpr in attempts:
        # ERDDAP time syntax: [(last)] or [(2026-10-20T00:00:00Z)]
        if texpr == "(last)":
            tslice = "[(last)]"
        else:
            tslice = texpr
        url = (
            f"{base}?"
            f"sla{tslice}[({miny2}):({maxy2})][({minx2}):({maxx2})],"
            f"ugos{tslice}[({miny2}):({maxy2})][({minx2}):({maxx2})],"
            f"vgos{tslice}[({miny2}):({maxy2})][({minx2}):({maxx2})]"
        )
        print(f"Trying nesdisSSH1day {tslice} ...")
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and len(r.content) > 1000:
                content = r.content
                break
            print(f"  status={r.status_code} size={len(r.content)}")
        except Exception as e:
            print(f"  error: {e}")

    if not content:
        return None

    combo_path = os.path.join(output_dir, "raw_ssh_current_combo.nc")
    with open(combo_path, "wb") as f:
        f.write(content)

    # دو فایل جدا برای سازگاری با processor (اختیاری؛ همان فایل هم کافی است)
    ssh_path = os.path.join(output_dir, "raw_ssh.nc")
    cur_path = os.path.join(output_dir, "raw_current.nc")
    with open(ssh_path, "wb") as f:
        f.write(content)
    with open(cur_path, "wb") as f:
        f.write(content)

    return {"ssh_path": ssh_path, "current_path": cur_path, "combo_path": combo_path}


def _download_blended_currents(minx, miny, maxx, maxy, out_path):
    """Fallback: noaacwBLENDEDNRTcurrentsDaily روی coastwatch.noaa.gov"""
    base = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily.nc"
    minx2, maxx2 = minx - 0.3, maxx + 0.3
    miny2, maxy2 = miny - 0.3, maxy + 0.3
    now = datetime.now(timezone.utc)
    attempts = ["(last)"]
    for days_ago in (3, 5, 8, 12):
        t = (now - timedelta(days=days_ago)).strftime("%Y-%m-%dT00:00:00Z")
        attempts.append(f"[({t})]")

    for texpr in attempts:
        tslice = "[(last)]" if texpr == "(last)" else texpr
        url = (
            f"{base}?"
            f"u_current{tslice}[({miny2}):({maxy2})][({minx2}):({maxx2})],"
            f"v_current{tslice}[({miny2}):({maxy2})][({minx2}):({maxx2})]"
        )
        print(f"Trying blended currents {tslice} ...")
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
            print(f"  status={r.status_code} size={len(r.content)}")
        except Exception as e:
            print(f"  error: {e}")
    return False



def _download_chlorophyll(minx, miny, maxx, maxy):
    """
    دانلود کلروفیل از منابع پایدارتر.
    اولویت: erdMH1chla1day (بدون altitude) با (last).
    """
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    minx2, maxx2 = minx - 0.2, maxx + 0.2
    miny2, maxy2 = miny - 0.2, maxy + 0.2
    now = datetime.now(timezone.utc)

    # (dataset, variable, has_altitude)
    candidates = [
        ("erdMH1chla1day", "chlorophyll", False),
        ("erdMH1chla8day", "chlorophyll", False),
        ("erdVH3chla8day", "chla", True),
        ("erdVHNchla8day", "chla", True),
        ("nesdisVHNchlaWeekly", "chlor_a", False),
    ]

    time_slices = ["[(last)]"]
    for days_ago in (2, 5, 8, 12, 16, 24):
        t = (now - timedelta(days=days_ago)).strftime("%Y-%m-%dT00:00:00Z")
        time_slices.append(f"[({t})]")

    for ds, var, has_alt in candidates:
        for tslice in time_slices:
            if has_alt:
                url = (
                    f"{base}/{ds}.nc?{var}{tslice}"
                    f"[(0.0):(0.0)][({miny2}):({maxy2})][({minx2}):({maxx2})]"
                )
            else:
                url = (
                    f"{base}/{ds}.nc?{var}{tslice}"
                    f"[({miny2}):({maxy2})][({minx2}):({maxx2})]"
                )
            print(f"Trying Chl {ds} {var} {tslice} ...")
            try:
                r = requests.get(url, timeout=90)
                if r.status_code == 200 and len(r.content) > 2000:
                    print(f"  Chl OK {ds} size={len(r.content)}")
                    return r.content
                print(f"  status={r.status_code} size={len(r.content)}")
            except Exception as e:
                print(f"  error: {e}")
    return None


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
