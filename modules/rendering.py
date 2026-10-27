# modules/rendering.py — رندر heatmap با colormap حرفه‌ای دریایی
import os
import base64
import numpy as np
import xarray as xr
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from utils.logging_helpers import record_error

# --- Colormapهای اختصاصی PFZ ---
# از آبی عمیق (کم) → فیروزه‌ای → سبز → زرد → نارنجی → قرمز تیره (عالی)
_PFZ_COLORS = [
    "#0B1D36",
    "#0D3B66",
    "#1A6B9A",
    "#1B9E77",
    "#5CB85C",
    "#E6C229",
    "#F4A261",
    "#E76F51",
    "#9B2226",
]
PFZ_CMAP = LinearSegmentedColormap.from_list("pfz_ocean", _PFZ_COLORS, N=256)

# SST: آبی سرد → سفید → قرمز گرم
_SST_COLORS = [
    "#023E8A",
    "#0077B6",
    "#00B4D8",
    "#90E0EF",
    "#CAF0F8",
    "#FEEAAB",
    "#F4A261",
    "#E76F51",
    "#9B2226",
]
SST_CMAP = LinearSegmentedColormap.from_list("sst_thermal", _SST_COLORS, N=256)

# Chlorophyll: تیره → سبز روشن
_CHL_COLORS = [
    "#0D1B0F",
    "#1B4332",
    "#2D6A4F",
    "#40916C",
    "#52B788",
    "#95D5B2",
    "#D8F3DC",
    "#F0FFF0",
]
CHL_CMAP = LinearSegmentedColormap.from_list("chl_green", _CHL_COLORS, N=256)

# Current speed: تیره → فیروزه‌ای → زرد
_CUR_COLORS = [
    "#0B132B",
    "#1C2541",
    "#3A506B",
    "#5BC0BE",
    "#6FFFE9",
    "#F0F3BD",
    "#F4D35E",
]
CURRENT_CMAP = LinearSegmentedColormap.from_list("current_speed", _CUR_COLORS, N=256)

# SSH gradient: بنفش تیره → صورتی → زرد
_SSH_COLORS = [
    "#1A0A2E",
    "#2D1B4E",
    "#5B2C6F",
    "#9B59B6",
    "#E8A0BF",
    "#F7DC6F",
    "#FCF3CF",
]
SSH_CMAP = LinearSegmentedColormap.from_list("ssh_grad", _SSH_COLORS, N=256)

CMAP_REGISTRY = {
    "pfz": PFZ_CMAP,
    "pfz_ocean": PFZ_CMAP,
    "jet": PFZ_CMAP,  # جایگزین jet قدیمی با پالت دریایی
    "sst": SST_CMAP,
    "RdYlBu_r": SST_CMAP,
    "chl": CHL_CMAP,
    "YlGn": CHL_CMAP,
    "current": CURRENT_CMAP,
    "ssh": SSH_CMAP,
}


def _resolve_cmap(cmap_name: str):
    if cmap_name in CMAP_REGISTRY:
        return CMAP_REGISTRY[cmap_name]
    try:
        return plt.colormaps.get_cmap(cmap_name)
    except Exception:
        try:
            return cm.get_cmap(cmap_name)
        except Exception:
            return PFZ_CMAP


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


def render_pixel_perfect_heatmap(
    da, label, reg_name, cmap_name, out_dir, alpha=0.82, p_low=2, p_high=98
):
    """
    ساخت تصویر PNG شفاف با:
    - نرمال‌سازی percentile (کنتراست بهتر، بدون outlier)
    - colormap اختصاصی دریایی
    - alpha نرم روی لبه‌های معتبر
    """
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
        data = np.asarray(da.values, dtype=float).copy()
        lats = da[lat_name].values
        lons = da[lon_name].values

        grid_miny, grid_maxy = float(np.nanmin(lats)), float(np.nanmax(lats))
        grid_minx, grid_maxx = float(np.nanmin(lons)), float(np.nanmax(lons))

        valid = np.isfinite(data)
        if not valid.any():
            return None, None

        # نرمال‌سازی بر اساس percentile برای کنتراست قوی‌تر
        vals = data[valid]
        vmin = float(np.nanpercentile(vals, p_low))
        vmax = float(np.nanpercentile(vals, p_high))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        norm = (data - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)

        cmap = _resolve_cmap(cmap_name)
        rgba = cmap(np.nan_to_num(norm, nan=0.0))

        # شفافیت: پیکسل نامعتبر = 0، بقیه با alpha قابل تنظیم
        rgba[..., 3] = np.where(valid, alpha, 0.0)

        # کمی soft-edge: پیکسل‌های نزدیک NaN کمی شفاف‌تر
        try:
            from scipy.ndimage import binary_dilation, binary_erosion

            core = binary_erosion(valid, iterations=1)
            edge = valid & ~core
            rgba[edge, 3] = alpha * 0.55
        except Exception:
            pass

        rgba_img = np.flipud(rgba)

        os.makedirs(out_dir, exist_ok=True)
        safe_name = reg_name.replace(" ", "_")
        file_path = os.path.join(out_dir, f"{label}_{safe_name}.png")
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
            preferred_vars = ["analysed_sst", "sst", "chlor_a", "chl", "pfz_index", "pfz"]
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
