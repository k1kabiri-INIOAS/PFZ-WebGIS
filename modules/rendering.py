# modules/rendering.py — رندر heatmap و برش داده
import os
import base64
import numpy as np
import xarray as xr
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.logging_helpers import record_error


def apply_visual_offset_to_bounds(bounds, lon_offset=0.0, lat_offset=0.0):
    if not bounds:
        return bounds
    sw, ne = bounds
    return [
        [sw[0] + lat_offset, sw[1] + lon_offset],
        [ne[0] + lat_offset, ne[1] + lon_offset],
    ]


def image_to_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def render_pixel_perfect_heatmap(da, label, reg_name, cmap_name, out_dir):
    """ساخت تصویر PNG شفاف از DataArray برای ImageOverlay."""
    try:
        if da is None:
            return None, None
        lat_name = next(
            (d for d in da.dims if d.lower() in ["lat", "latitude", "y"]), None
        )
        lon_name = next(
            (d for d in da.dims if d.lower() in ["lon", "longitude", "x"]), None
        )
        if not lat_name or not lon_name:
            return None, None

        da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
        data = da.values.copy()
        lats = da[lat_name].values
        lons = da[lon_name].values

        grid_miny, grid_maxy = float(np.nanmin(lats)), float(np.nanmax(lats))
        grid_minx, grid_maxx = float(np.nanmin(lons)), float(np.nanmax(lons))

        valid = ~np.isnan(data)
        if not valid.any():
            return None, None

        vmin, vmax = float(np.nanmin(data[valid])), float(np.nanmax(data[valid]))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        norm = (data - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        try:
            cmap = plt.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap(cmap_name)
        rgba = cmap(np.nan_to_num(norm, nan=0.0))
        rgba[..., 3] = np.where(valid, 0.85, 0.0)
        rgba_img = np.flipud(rgba)

        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{label}_{reg_name.replace(' ', '_')}.png")
        Image.fromarray((rgba_img * 255).astype(np.uint8)).save(file_path)
        return file_path, [[grid_miny, grid_minx], [grid_maxy, grid_maxx]]
    except Exception as e:
        record_error(f"خطا در رندر {label}", e)
        return None, None


def load_and_crop_dataset(nc_path, shp_path):
    if not nc_path or not os.path.exists(nc_path):
        return None
    try:
        with xr.open_dataset(nc_path) as ds:
            target_var = None
            preferred_vars = ["analysed_sst", "sst", "chlor_a", "chl", "pfz_index"]
            for v in preferred_vars:
                if v in ds.data_vars:
                    target_var = v
                    break
            if not target_var:
                target_var = list(ds.data_vars.keys())[0] if ds.data_vars else None
            if not target_var:
                return None
            da = ds[target_var].load()

        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        lat_name = next(
            (d for d in da.dims if d.lower() in ["lat", "latitude", "y"]), None
        )
        lon_name = next(
            (d for d in da.dims if d.lower() in ["lon", "longitude", "x"]), None
        )
        if not lat_name or not lon_name:
            return None
        return (
            da.sortby(lat_name)
            .sortby(lon_name)
            .sel(
                {
                    lat_name: slice(miny - 0.05, maxy + 0.05),
                    lon_name: slice(minx - 0.05, maxx + 0.05),
                }
            )
        )
    except Exception as ex:
        record_error(f"خطا در برش داده {nc_path}", ex)
    return None
