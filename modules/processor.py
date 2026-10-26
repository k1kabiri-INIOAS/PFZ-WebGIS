# File Path: modules/processor.py
# Description: Multi-factor PFZ pipeline with independent auto-normalized weights.

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from shapely.geometry import LineString
import rioxarray


def _standardize_ds(ds):
    rename_dict = {}
    for dim in ["longitude", "x"]:
        if dim in ds.dims:
            rename_dict[dim] = "lon"
    for dim in ["latitude", "y"]:
        if dim in ds.dims:
            rename_dict[dim] = "lat"
    return ds.rename(rename_dict) if rename_dict else ds


def _to_2d(da):
    da = da.squeeze()
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


def _load_current_speed(current_nc_path, target_da):
    if not current_nc_path or not os.path.exists(current_nc_path):
        return None
    with xr.open_dataset(current_nc_path) as ds:
        ds = _standardize_ds(ds.load())
        names = {v.lower(): v for v in ds.data_vars}
        u_var = None
        v_var = None
        for key in ("ugos", "u_current", "currents_u", "uo", "u"):
            if key in names:
                u_var = names[key]
                break
        for key in ("vgos", "v_current", "currents_v", "vo", "v"):
            if key in names:
                v_var = names[key]
                break
        if u_var is None:
            u_var = next(
                (v for v in ds.data_vars if v.lower().startswith("u") and "g" in v.lower()),
                None,
            )
        if v_var is None:
            v_var = next(
                (v for v in ds.data_vars if v.lower().startswith("v") and "g" in v.lower()),
                None,
            )
        if u_var is None or v_var is None:
            vars_ = list(ds.data_vars.keys())
            if len(vars_) >= 2:
                u_var, v_var = vars_[0], vars_[1]
            else:
                return None
        u = _to_2d(ds[u_var])
        v = _to_2d(ds[v_var])
        speed = np.sqrt(
            np.asarray(u.values, dtype=float) ** 2
            + np.asarray(v.values, dtype=float) ** 2
        )
        speed_da = xr.DataArray(speed, coords=u.coords, dims=u.dims, name="current_speed")
        return _to_2d(speed_da.interp_like(target_da, method="nearest"))


def _load_ssh_gradient(ssh_nc_path, target_da):
    if not ssh_nc_path or not os.path.exists(ssh_nc_path):
        return None
    with xr.open_dataset(ssh_nc_path) as ds:
        ds = _standardize_ds(ds.load())
        ssh_var = next(
            (
                v
                for v in ds.data_vars
                if any(k in v.lower() for k in ("sla", "ssh", "adt", "height", "anomaly"))
            ),
            list(ds.data_vars.keys())[0] if ds.data_vars else None,
        )
        if not ssh_var:
            return None
        ssh = _to_2d(ds[ssh_var].interp_like(target_da, method="nearest"))
        vals = np.asarray(ssh.values, dtype=float)
        finite = np.isfinite(vals)
        fill = np.nanmean(vals[finite]) if finite.any() else 0.0
        filled = np.where(finite, vals, fill)
        dy, dx = np.gradient(filled)
        grad = np.sqrt(dx ** 2 + dy ** 2)
        return xr.DataArray(grad, coords=ssh.coords, dims=ssh.dims, name="ssh_grad")


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
    """Multi-factor PFZ pipeline with auto-normalized independent weights."""
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

        with xr.open_dataset(sst_nc_path) as ds_sst_raw, xr.open_dataset(chl_nc_path) as ds_chl_raw:
            ds_sst = _standardize_ds(ds_sst_raw.load())
            ds_chl = _standardize_ds(ds_chl_raw.load())

        sst_var = next(
            (v for v in ds_sst.data_vars if "sst" in v.lower() or "temp" in v.lower()),
            list(ds_sst.data_vars.keys())[0],
        )
        chl_var = next(
            (v for v in ds_chl.data_vars if "chl" in v.lower()),
            list(ds_chl.data_vars.keys())[0],
        )

        ds_chl = ds_chl.interp_like(ds_sst, method="nearest")
        sst_da = _to_2d(ds_sst[sst_var])
        chl_da = _to_2d(ds_chl[chl_var])

        if shapefile_path and os.path.exists(shapefile_path):
            try:
                gdf_roi = gpd.read_file(shapefile_path)
                if gdf_roi.crs is not None and str(gdf_roi.crs) != "EPSG:4326":
                    gdf_roi = gdf_roi.to_crs("EPSG:4326")
                sst_da = sst_da.rio.write_crs("EPSG:4326").rio.clip(
                    gdf_roi.geometry, gdf_roi.crs, drop=False
                )
                chl_da = chl_da.rio.write_crs("EPSG:4326").rio.clip(
                    gdf_roi.geometry, gdf_roi.crs, drop=False
                )
            except Exception as clip_err:
                print(f"ROI Clipping Warning: {clip_err}")

        sst_vals = np.asarray(sst_da.values, dtype=float).copy()
        chl_vals = np.asarray(chl_da.values, dtype=float).copy()

        raw_ocean_mask = np.isfinite(sst_vals) & np.isfinite(chl_vals)
        if not raw_ocean_mask.any():
            raise ValueError("هیچ پیکسل دریایی معتبری یافت نشد.")

        ocean_mask = binary_erosion(raw_ocean_mask, structure=np.ones((3, 3)), iterations=2)
        if not ocean_mask.any():
            ocean_mask = raw_ocean_mask

        factors = {}
        raw_weights = {}

        if use_sst and float(sst_weight) > 0:
            sst_filled = np.where(
                raw_ocean_mask, sst_vals, np.nanmean(sst_vals[raw_ocean_mask])
            )
            dy, dx = np.gradient(sst_filled)
            sst_grad = np.sqrt(dx ** 2 + dy ** 2)
            factors["sst"] = _normalize_on_mask(sst_grad, ocean_mask)
            raw_weights["sst"] = float(sst_weight)

        if use_chl and float(chl_weight) > 0:
            chl_log = np.log1p(np.maximum(chl_vals, 0))
            factors["chl"] = _normalize_on_mask(chl_log, ocean_mask)
            raw_weights["chl"] = float(chl_weight)

        if use_current and float(current_weight) > 0 and current_nc_path:
            try:
                speed_da = _load_current_speed(current_nc_path, sst_da)
                if speed_da is not None:
                    speed_vals = np.asarray(speed_da.values, dtype=float)
                    if speed_vals.shape == sst_vals.shape:
                        factors["current"] = _normalize_on_mask(speed_vals, ocean_mask)
                        raw_weights["current"] = float(current_weight)
                        print("Current factor included in PFZ index.")
                    else:
                        print("[Warning] Current grid shape mismatch; skipping.")
            except Exception as cur_err:
                print(f"[Warning] Current factor failed: {cur_err}")

        if use_ssh and float(ssh_weight) > 0 and ssh_nc_path:
            try:
                ssh_grad_da = _load_ssh_gradient(ssh_nc_path, sst_da)
                if ssh_grad_da is not None:
                    ssh_vals = np.asarray(ssh_grad_da.values, dtype=float)
                    if ssh_vals.shape == sst_vals.shape:
                        factors["ssh"] = _normalize_on_mask(ssh_vals, ocean_mask)
                        raw_weights["ssh"] = float(ssh_weight)
                        print("SSH gradient factor included in PFZ index.")
                    else:
                        print("[Warning] SSH grid shape mismatch; skipping.")
            except Exception as ssh_err:
                print(f"[Warning] SSH factor failed: {ssh_err}")

        if not factors:
            raise ValueError("هیچ عامل فعالی برای محاسبه PFZ باقی نماند.")

        w_sum = sum(raw_weights.values())
        if w_sum <= 0:
            norm_w = {k: 1.0 / len(factors) for k in factors}
        else:
            norm_w = {k: raw_weights[k] / w_sum for k in factors}

        pfz_index_arr = np.full_like(sst_vals, np.nan, dtype=float)
        combined = np.zeros(sst_vals.shape, dtype=float)
        for k, arr in factors.items():
            combined += norm_w[k] * arr
        pfz_index_arr[ocean_mask] = combined[ocean_mask]

        print(f"PFZ factors used: {list(factors.keys())}")
        print(f"Normalized weights: {norm_w}")

        lon_arr = ds_sst.lon.values if "lon" in ds_sst.coords else ds_sst.longitude.values
        lat_arr = ds_sst.lat.values if "lat" in ds_sst.coords else ds_sst.latitude.values

        contour_data = np.nan_to_num(pfz_index_arr, nan=0.0)
        fig, ax = plt.subplots()
        cs = ax.contour(lon_arr, lat_arr, contour_data, levels=[0.50])
        lines = []
        for segs in cs.allsegs:
            for poly in segs:
                if len(poly) > 2:
                    lines.append(LineString(poly))
        plt.close(fig)

        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        gdf.to_file(fronts_geojson, driver="GeoJSON")

        ds_out = xr.Dataset(
            {
                "pfz": (["lat", "lon"], pfz_index_arr),
                "pfz_index": (["lat", "lon"], pfz_index_arr),
            },
            coords={"lon": lon_arr, "lat": lat_arr},
        )
        ds_out.to_netcdf(nc_out)

        pfz_da = ds_out["pfz_index"]
        pfz_da = pfz_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        pfz_da = pfz_da.rio.write_crs("EPSG:4326", inplace=False)
        pfz_da.rio.to_raster(tif_out)

        return nc_out, tif_out, fronts_geojson

    except Exception as e:
        print(f"Error in PFZ pipeline: {e}")
        return nc_out, tif_out, fronts_geojson
