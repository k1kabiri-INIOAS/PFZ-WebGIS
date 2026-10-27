# File Path: modules/processor.py
# Description: Multi-factor PFZ pipeline — SST+Chl core, optional Current/SSH.

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from shapely.geometry import LineString
import rioxarray


def _standardize_ds(ds):
    rename_dict = {}
    for dim in list(ds.dims) + list(ds.coords):
        dl = str(dim).lower()
        if dl in ("longitude", "x") and dim != "lon":
            rename_dict[dim] = "lon"
        if dl in ("latitude", "y") and dim != "lat":
            rename_dict[dim] = "lat"
    return ds.rename(rename_dict) if rename_dict else ds


def _to_2d(da):
    da = da.squeeze(drop=True)
    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})
    return da


def _normalize_on_mask(arr, mask):
    out = np.zeros_like(arr, dtype=float)
    vals = arr[mask]
    if len(vals) == 0:
        return out
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return out
    out[mask] = (arr[mask] - v_min) / (v_max - v_min)
    return out


def _align_to_target(da, target_da):
    """Interp onto target grid; returns numpy array same shape as target or None."""
    try:
        da = _to_2d(da)
        target_da = _to_2d(target_da)
        # ensure lon/lat names
        rename = {}
        for d in da.dims:
            dl = d.lower()
            if dl in ("longitude", "x") and d != "lon":
                rename[d] = "lon"
            if dl in ("latitude", "y") and d != "lat":
                rename[d] = "lat"
        if rename:
            da = da.rename(rename)
        t_rename = {}
        for d in target_da.dims:
            dl = d.lower()
            if dl in ("longitude", "x") and d != "lon":
                t_rename[d] = "lon"
            if dl in ("latitude", "y") and d != "lat":
                t_rename[d] = "lat"
        if t_rename:
            target_da = target_da.rename(t_rename)
        aligned = da.interp_like(target_da, method="nearest")
        vals = np.asarray(_to_2d(aligned).values, dtype=float)
        if vals.shape != np.asarray(target_da.values).shape:
            # last resort: reindex
            aligned = da.reindex_like(target_da, method="nearest")
            vals = np.asarray(_to_2d(aligned).values, dtype=float)
        if vals.shape != np.asarray(target_da.values).shape:
            return None
        return vals
    except Exception as e:
        print(f"[align] {e}")
        return None


def _load_current_speed(current_nc_path, target_da):
    if not current_nc_path or not os.path.exists(current_nc_path):
        return None
    try:
        with xr.open_dataset(current_nc_path) as ds:
            ds = _standardize_ds(ds.load())
            names = {v.lower(): v for v in ds.data_vars}
            u_var = next((names[k] for k in ("ugos", "u_current", "currents_u", "uo", "u") if k in names), None)
            v_var = next((names[k] for k in ("vgos", "v_current", "currents_v", "vo", "v") if k in names), None)
            if u_var is None or v_var is None:
                vars_ = [v for v in ds.data_vars if v.lower() not in ("crs", "time")]
                if len(vars_) >= 2:
                    # prefer ugos/vgos-like
                    u_var = next((v for v in vars_ if "u" in v.lower()), vars_[0])
                    v_var = next((v for v in vars_ if "v" in v.lower() and v != u_var), vars_[1])
                else:
                    return None
            u = _to_2d(ds[u_var])
            v = _to_2d(ds[v_var])
            speed = np.sqrt(
                np.nan_to_num(np.asarray(u.values, dtype=float), nan=0.0) ** 2
                + np.nan_to_num(np.asarray(v.values, dtype=float), nan=0.0) ** 2
            )
            speed_da = xr.DataArray(speed, coords=u.coords, dims=u.dims, name="current_speed")
            return _align_to_target(speed_da, target_da)
    except Exception as e:
        print(f"[current] {e}")
        return None


def _load_ssh_gradient(ssh_nc_path, target_da):
    if not ssh_nc_path or not os.path.exists(ssh_nc_path):
        return None
    try:
        with xr.open_dataset(ssh_nc_path) as ds:
            ds = _standardize_ds(ds.load())
            ssh_var = next(
                (
                    v
                    for v in ds.data_vars
                    if any(k in v.lower() for k in ("sla", "ssh", "adt", "height", "anomaly"))
                ),
                None,
            )
            if ssh_var is None:
                # skip current vars
                for v in ds.data_vars:
                    if v.lower() not in ("ugos", "vgos", "u_current", "v_current", "crs"):
                        ssh_var = v
                        break
            if not ssh_var:
                return None
            ssh = _to_2d(ds[ssh_var])
            vals = _align_to_target(ssh, target_da)
            if vals is None:
                return None
            finite = np.isfinite(vals)
            if not finite.any():
                return None
            fill = float(np.nanmean(vals[finite]))
            filled = np.where(finite, vals, fill)
            dy, dx = np.gradient(filled)
            return np.sqrt(dx ** 2 + dy ** 2)
    except Exception as e:
        print(f"[ssh] {e}")
        return None


def process_pfz_pipeline(
    shapefile_path=None,
    sst_nc_path=None,
    chl_nc_path=None,
    output_dir="outputs",
    sst_weight=0.5,
    chl_weight=0.5,
    current_nc_path=None,
    current_weight=0.0,
    ssh_nc_path=None,
    ssh_weight=0.0,
    use_sst=True,
    use_chl=True,
    use_current=False,
    use_ssh=False,
    **kwargs,
):
    """Core = SST + Chl. Current/SSH optional and never block core output."""
    if kwargs:
        shapefile_path = kwargs.get("shapefile_path", shapefile_path)
        sst_nc_path = kwargs.get("sst_nc_path", sst_nc_path)
        chl_nc_path = kwargs.get("chl_nc_path", chl_nc_path)
        output_dir = kwargs.get("output_dir", output_dir)
        sst_weight = kwargs.get("sst_weight", sst_weight)
        chl_weight = kwargs.get("chl_weight", chl_weight)
        current_nc_path = kwargs.get("current_nc_path", current_nc_path)
        current_weight = kwargs.get("current_weight", current_weight)
        ssh_nc_path = kwargs.get("ssh_nc_path", ssh_nc_path)
        ssh_weight = kwargs.get("ssh_weight", ssh_weight)
        use_sst = kwargs.get("use_sst", use_sst)
        use_chl = kwargs.get("use_chl", use_chl)
        use_current = kwargs.get("use_current", use_current)
        use_ssh = kwargs.get("use_ssh", use_ssh)

    os.makedirs(output_dir, exist_ok=True)
    nc_out = os.path.join(output_dir, "pfz_output.nc")
    tif_out = os.path.join(output_dir, "pfz_output.tif")
    fronts_geojson = os.path.join(output_dir, "pfz_fronts.geojson")

    try:
        if not sst_nc_path or not os.path.exists(sst_nc_path):
            raise ValueError("فایل SST موجود نیست.")
        if not chl_nc_path or not os.path.exists(chl_nc_path):
            raise ValueError("فایل Chlorophyll موجود نیست.")

        with xr.open_dataset(sst_nc_path) as ds_sst_raw:
            ds_sst = _standardize_ds(ds_sst_raw.load())
        with xr.open_dataset(chl_nc_path) as ds_chl_raw:
            ds_chl = _standardize_ds(ds_chl_raw.load())

        sst_var = next(
            (v for v in ds_sst.data_vars if "sst" in v.lower() or "temp" in v.lower()),
            list(ds_sst.data_vars.keys())[0],
        )
        chl_var = next(
            (
                v
                for v in ds_chl.data_vars
                if any(k in v.lower() for k in ("chl", "chlor"))
            ),
            list(ds_chl.data_vars.keys())[0],
        )

        sst_da = _to_2d(ds_sst[sst_var])
        # align chl to sst
        try:
            ds_chl = ds_chl.interp_like(ds_sst, method="nearest")
            chl_da = _to_2d(ds_chl[chl_var])
        except Exception:
            chl_da = _to_2d(ds_chl[chl_var])
            chl_vals_try = _align_to_target(chl_da, sst_da)
            if chl_vals_try is not None:
                chl_da = xr.DataArray(chl_vals_try, coords=sst_da.coords, dims=sst_da.dims)
            else:
                raise ValueError("عدم تطبیق شبکه Chl با SST")

        if shapefile_path and os.path.exists(shapefile_path):
            try:
                gdf_roi = gpd.read_file(shapefile_path)
                if gdf_roi.crs is not None and str(gdf_roi.crs) != "EPSG:4326":
                    gdf_roi = gdf_roi.to_crs("EPSG:4326")
                sst_da = sst_da.rio.write_crs("EPSG:4326").rio.clip(
                    gdf_roi.geometry, drop=False
                )
                chl_da = chl_da.rio.write_crs("EPSG:4326").rio.clip(
                    gdf_roi.geometry, drop=False
                )
            except Exception as clip_err:
                print(f"ROI Clipping Warning: {clip_err}")

        sst_vals = np.asarray(sst_da.values, dtype=float).copy()
        chl_vals = np.asarray(chl_da.values, dtype=float).copy()
        if chl_vals.shape != sst_vals.shape:
            # force align
            aligned = _align_to_target(chl_da, sst_da)
            if aligned is None:
                raise ValueError(f"شکل Chl {chl_vals.shape} با SST {sst_vals.shape} سازگار نیست")
            chl_vals = aligned

        raw_ocean_mask = np.isfinite(sst_vals) & np.isfinite(chl_vals)
        if not raw_ocean_mask.any():
            # relax: SST only mask
            raw_ocean_mask = np.isfinite(sst_vals)
        if not raw_ocean_mask.any():
            raise ValueError("هیچ پیکسل دریایی معتبری یافت نشد.")

        ocean_mask = binary_erosion(raw_ocean_mask, structure=np.ones((3, 3)), iterations=2)
        if not ocean_mask.any():
            ocean_mask = raw_ocean_mask

        factors = {}
        raw_weights = {}

        # --- SST gradient (core) ---
        if use_sst and float(sst_weight) > 0:
            fill = float(np.nanmean(sst_vals[raw_ocean_mask]))
            sst_filled = np.where(raw_ocean_mask, sst_vals, fill)
            dy, dx = np.gradient(sst_filled)
            sst_grad = np.sqrt(dx ** 2 + dy ** 2)
            factors["sst"] = _normalize_on_mask(sst_grad, ocean_mask)
            raw_weights["sst"] = float(sst_weight)

        # --- Chlorophyll (core) ---
        if use_chl and float(chl_weight) > 0:
            chl_log = np.log1p(np.maximum(np.nan_to_num(chl_vals, nan=0.0), 0))
            factors["chl"] = _normalize_on_mask(chl_log, ocean_mask)
            raw_weights["chl"] = float(chl_weight)

        # --- Current (optional, never blocks) ---
        if use_current and float(current_weight) > 0 and current_nc_path:
            try:
                speed_vals = _load_current_speed(current_nc_path, sst_da)
                if speed_vals is not None and speed_vals.shape == sst_vals.shape:
                    factors["current"] = _normalize_on_mask(speed_vals, ocean_mask)
                    raw_weights["current"] = float(current_weight)
                    print("Current factor included.")
                else:
                    print("[Warning] Current skipped (shape/load).")
            except Exception as e:
                print(f"[Warning] Current skipped: {e}")

        # --- SSH (optional, never blocks) ---
        if use_ssh and float(ssh_weight) > 0 and ssh_nc_path:
            try:
                ssh_vals = _load_ssh_gradient(ssh_nc_path, sst_da)
                if ssh_vals is not None and ssh_vals.shape == sst_vals.shape:
                    factors["ssh"] = _normalize_on_mask(ssh_vals, ocean_mask)
                    raw_weights["ssh"] = float(ssh_weight)
                    print("SSH factor included.")
                else:
                    print("[Warning] SSH skipped (shape/load).")
            except Exception as e:
                print(f"[Warning] SSH skipped: {e}")

        if not factors:
            # force at least SST
            fill = float(np.nanmean(sst_vals[raw_ocean_mask]))
            sst_filled = np.where(raw_ocean_mask, sst_vals, fill)
            dy, dx = np.gradient(sst_filled)
            factors["sst"] = _normalize_on_mask(np.sqrt(dx ** 2 + dy ** 2), ocean_mask)
            raw_weights["sst"] = 1.0

        w_sum = sum(raw_weights.values()) or 1.0
        norm_w = {k: raw_weights[k] / w_sum for k in factors}

        pfz_index_arr = np.full_like(sst_vals, np.nan, dtype=float)
        combined = np.zeros(sst_vals.shape, dtype=float)
        for k, arr in factors.items():
            combined += norm_w[k] * arr
        pfz_index_arr[ocean_mask] = combined[ocean_mask]

        print(f"PFZ factors: {list(factors.keys())} weights={norm_w}")

        # coords
        if "lon" in sst_da.coords:
            lon_arr = np.asarray(sst_da["lon"].values)
        elif "longitude" in sst_da.coords:
            lon_arr = np.asarray(sst_da["longitude"].values)
        else:
            lon_arr = np.asarray(ds_sst["lon"].values if "lon" in ds_sst else ds_sst["longitude"].values)

        if "lat" in sst_da.coords:
            lat_arr = np.asarray(sst_da["lat"].values)
        elif "latitude" in sst_da.coords:
            lat_arr = np.asarray(sst_da["latitude"].values)
        else:
            lat_arr = np.asarray(ds_sst["lat"].values if "lat" in ds_sst else ds_sst["latitude"].values)

        # ensure 1D
        lon_arr = np.atleast_1d(lon_arr).ravel()
        lat_arr = np.atleast_1d(lat_arr).ravel()
        if pfz_index_arr.shape != (len(lat_arr), len(lon_arr)):
            # clip arrays to match
            lat_arr = lat_arr[: pfz_index_arr.shape[0]]
            lon_arr = lon_arr[: pfz_index_arr.shape[1]]

        ds_out = xr.Dataset(
            {
                "pfz": (["lat", "lon"], pfz_index_arr),
                "pfz_index": (["lat", "lon"], pfz_index_arr),
            },
            coords={"lon": lon_arr, "lat": lat_arr},
        )
        ds_out.to_netcdf(nc_out)

        try:
            pfz_da = ds_out["pfz_index"]
            pfz_da = pfz_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            pfz_da = pfz_da.rio.write_crs("EPSG:4326", inplace=False)
            pfz_da.rio.to_raster(tif_out)
        except Exception as te:
            print(f"TIFF warning: {te}")

        # crude fronts file (optional; app regenerates properly)
        try:
            contour_data = np.nan_to_num(pfz_index_arr, nan=0.0)
            fig, ax = plt.subplots()
            cs = ax.contour(lon_arr, lat_arr, contour_data, levels=[0.50])
            lines = [
                LineString(poly)
                for segs in cs.allsegs
                for poly in segs
                if len(poly) > 2
            ]
            plt.close(fig)
            if lines:
                gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_file(
                    fronts_geojson, driver="GeoJSON"
                )
        except Exception as fe:
            print(f"Front sketch warning: {fe}")

        return nc_out, tif_out, fronts_geojson

    except Exception as e:
        print(f"Error in PFZ pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
