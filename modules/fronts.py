# modules/fronts.py — استخراج جبهه از گرادیان SST (نه کانتور PFZ نرم)
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from utils.logging_helpers import record_error


def _standardize_da(da):
    rename = {}
    for d in da.dims:
        if d.lower() in ("longitude", "x") and d != "lon":
            rename[d] = "lon"
        if d.lower() in ("latitude", "y") and d != "lat":
            rename[d] = "lat"
    return da.rename(rename) if rename else da


def _to_2d(da):
    da = da.squeeze()
    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})
    return da


def _load_sst_gradient(sst_nc_path, region_shp_path=None):
    """گرادیان قدر مطلق SST روی شبکه، با ماسک دریا و کلیپ منطقه."""
    if not sst_nc_path or not os.path.exists(sst_nc_path):
        return None, None, None

    with xr.open_dataset(sst_nc_path) as ds:
        ds = ds.load()
        rename = {}
        for d in list(ds.dims):
            if d.lower() in ("longitude", "x"):
                rename[d] = "lon"
            if d.lower() in ("latitude", "y"):
                rename[d] = "lat"
        if rename:
            ds = ds.rename(rename)
        sst_var = next(
            (v for v in ds.data_vars if "sst" in v.lower() or "temp" in v.lower()),
            list(ds.data_vars.keys())[0],
        )
        da = _to_2d(ds[sst_var])
        da = da.sortby("lat", ascending=True).sortby("lon", ascending=True)

    if region_shp_path and os.path.exists(region_shp_path):
        try:
            gdf = gpd.read_file(region_shp_path)
            if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            da = da.rio.write_crs("EPSG:4326").rio.clip(gdf.geometry, gdf.crs, drop=True)
        except Exception as e:
            print(f"SST clip warning: {e}")

    vals = np.asarray(da.values, dtype=float)
    ocean = np.isfinite(vals)
    if not ocean.any():
        return None, None, None

    # حذف نوار ساحلی برای جلوگیری از جبههٔ کاذب
    ocean_eroded = ndimage.binary_erosion(ocean, structure=np.ones((3, 3)), iterations=2)
    if not ocean_eroded.any():
        ocean_eroded = ocean

    fill = np.nanmean(vals[ocean])
    filled = np.where(ocean, vals, fill)
    # کمی نرم‌سازی برای کاهش نویز
    filled = ndimage.gaussian_filter(filled, sigma=0.8)
    dy, dx = np.gradient(filled)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    grad[~ocean_eroded] = np.nan

    lats = da["lat"].values if "lat" in da.coords else da.latitude.values
    lons = da["lon"].values if "lon" in da.coords else da.longitude.values
    return grad, lats, lons


def _load_pfz_field(nc_path):
    if not nc_path or not os.path.exists(nc_path):
        return None, None, None
    with xr.open_dataset(nc_path) as ds:
        var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
        da = _standardize_da(ds[var_key].load())
        da = _to_2d(da)
        lat_name = next((d for d in da.dims if d.lower() in ("lat", "latitude", "y")), None)
        lon_name = next((d for d in da.dims if d.lower() in ("lon", "longitude", "x")), None)
        if not lat_name or not lon_name:
            return None, None, None
        da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
        return (
            np.asarray(da.values, dtype=float),
            da[lat_name].values,
            da[lon_name].values,
        )


def _contour_lines(field, lats, lons, level):
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(lon_grid, lat_grid, field, levels=[level])
        lines = [
            LineString(poly)
            for segs in cs.allsegs
            for poly in segs
            if len(poly) >= 4
        ]
    finally:
        plt.close(fig)
    return lines


def _clip_and_filter_lines(lines, region_shp_path, min_km=5.0, max_km=400.0):
    """کلیپ به محدودهٔ منطقه + فیلتر طول غیرواقعی."""
    if not lines:
        return None
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

    if region_shp_path and os.path.exists(region_shp_path):
        try:
            region = gpd.read_file(region_shp_path)
            if region.crs is not None and str(region.crs) != "EPSG:4326":
                region = region.to_crs("EPSG:4326")
            # کمی بافر برای نگه داشتن جبهه‌های نزدیک مرز
            region_poly = region.geometry.buffer(0.05).unary_union
            gdf = gdf[gdf.intersects(region_poly)].copy()
            if gdf.empty:
                return None
            gdf["geometry"] = gdf.intersection(region_poly)
            gdf = gdf[~gdf.is_empty & gdf.geometry.notna()]
            # MultiLineString → تکه خط
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geometry.type.isin(["LineString", "LinearRing"])]
        except Exception as e:
            print(f"Front clip warning: {e}")

    if gdf.empty:
        return None

    gdf["Length_km"] = gdf.to_crs("EPSG:3857").geometry.length / 1000.0
    # فیلتر خطوط خیلی کوتاه (نویز) و خیلی بلند (ایزولین سراسری)
    gdf = gdf[(gdf["Length_km"] >= min_km) & (gdf["Length_km"] <= max_km)]
    return gdf if not gdf.empty else None


def generate_fronts_fallback(
    nc_path,
    user_threshold,
    region_name,
    sst_nc_path=None,
    region_shp_path=None,
):
    """
    استخراج خطوط جبههٔ حرارتی از گرادیان SST.
    اگر SST در دسترس نباشد، از صدک بالای PFZ به‌عنوان پشتیبان استفاده می‌شود.
    user_threshold در بازهٔ ۰٫۱–۱ روی صدک گرادیان نگاشت می‌شود
    (مثلاً ۰٫۵ → صدک ۷۵، ۰٫۸ → صدک ۹۰).
    """
    try:
        lines = []
        method = "none"
        active_threshold = float(user_threshold)

        # ---- روش اصلی: گرادیان SST ----
        grad, lats, lons = _load_sst_gradient(sst_nc_path, region_shp_path)
        if grad is not None and np.isfinite(grad).any():
            valid = grad[np.isfinite(grad)]
            # نگاشت آستانه کاربر به صدک (۰٫۱→۵۵٪ ، ۱٫۰→۹۸٪)
            pct = 55.0 + float(user_threshold) * 43.0
            pct = float(np.clip(pct, 55.0, 98.0))
            level = float(np.nanpercentile(valid, pct))
            active_threshold = level
            # چند سطح نزدیک برای پوشش بهتر
            for lv in (level, level * 0.92, float(np.nanpercentile(valid, min(pct + 5, 98)))):
                lines.extend(_contour_lines(grad, lats, lons, lv))
            method = f"sst_grad_p{pct:.0f}"

        # ---- پشتیبان: صدک بالای PFZ (نه ایزولین ۰٫۵ سراسری) ----
        if not lines:
            pfz, lats, lons = _load_pfz_field(nc_path)
            if pfz is not None and np.isfinite(pfz).any():
                valid = pfz[np.isfinite(pfz) & (pfz > 0)]
                if len(valid) > 0:
                    pct = 70.0 + float(user_threshold) * 25.0
                    pct = float(np.clip(pct, 70.0, 95.0))
                    level = float(np.nanpercentile(valid, pct))
                    active_threshold = level
                    lines = _contour_lines(pfz, lats, lons, level)
                    method = f"pfz_p{pct:.0f}"

        if not lines:
            return None

        gdf = _clip_and_filter_lines(
            lines, region_shp_path, min_km=5.0, max_km=350.0
        )
        if gdf is None or gdf.empty:
            # کمی شل‌تر
            gdf = _clip_and_filter_lines(
                lines, region_shp_path, min_km=2.0, max_km=500.0
            )
        if gdf is None or gdf.empty:
            return None

        gdf = gdf.copy()
        gdf["Region"] = region_name
        gdf["Threshold"] = active_threshold
        gdf["Method"] = method
        if "Length_km" not in gdf.columns:
            gdf["Length_km"] = gdf.to_crs("EPSG:3857").geometry.length / 1000.0

        # حذف خطوط تکراری/تقریباً هم‌پوشان خیلی ساده با نگه داشتن بلندترها
        gdf = gdf.sort_values("Length_km", ascending=False).reset_index(drop=True)
        return gdf

    except Exception as ex:
        record_error(f"خطا در استخراج جبهه برای {region_name}", ex)
    return None


def mask_pfz_by_fronts(da, fronts_gdf, buffer_deg=0.06):
    """برش لایه PFZ فقط در حریم اطراف خطوط جبهه."""
    if fronts_gdf is None or fronts_gdf.empty or da is None:
        return None
    try:
        lat_name = next(
            (d for d in da.dims if d.lower() in ["lat", "latitude", "y"]), None
        )
        lon_name = next(
            (d for d in da.dims if d.lower() in ["lon", "longitude", "x"]), None
        )
        if not lat_name or not lon_name:
            return None

        lats, lons = da[lat_name].values, da[lon_name].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        fronts_union = fronts_gdf.geometry.buffer(buffer_deg).unary_union

        try:
            from shapely.vectorized import contains

            mask = contains(fronts_union, lon_grid, lat_grid)
        except ImportError:
            from shapely.geometry import Point
            from shapely.prepared import prep

            prep_poly = prep(fronts_union)
            mask = np.array(
                [
                    prep_poly.contains(Point(x, y))
                    for x, y in zip(lon_grid.ravel(), lat_grid.ravel())
                ]
            ).reshape(lon_grid.shape)

        da_masked = da.copy()
        da_masked.values[~mask] = np.nan
        return da_masked
    except Exception as e:
        record_error("خطا در برش لایه PFZ روی جبهه‌ها", e)
        return None


def mask_land_from_pfz(da, land_gdf):
    """حذف مقادیر شبکه از روی مناطق خشکی و جزایر."""
    if land_gdf is None or land_gdf.empty or da is None:
        return da
    try:
        lat_name = next(
            (d for d in da.dims if d.lower() in ["lat", "latitude", "y"]), None
        )
        lon_name = next(
            (d for d in da.dims if d.lower() in ["lon", "longitude", "x"]), None
        )
        if not lat_name or not lon_name:
            return da
        lats, lons = da[lat_name].values, da[lon_name].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        land = land_gdf.to_crs("EPSG:4326") if land_gdf.crs else land_gdf
        land_union = land.geometry.unary_union
        try:
            from shapely.vectorized import contains

            on_land = contains(land_union, lon_grid, lat_grid)
        except ImportError:
            from shapely.geometry import Point
            from shapely.prepared import prep

            prep_land = prep(land_union)
            on_land = np.array(
                [
                    prep_land.contains(Point(x, y))
                    for x, y in zip(lon_grid.ravel(), lat_grid.ravel())
                ]
            ).reshape(lon_grid.shape)
        out = da.copy()
        out.values[on_land] = np.nan
        return out
    except Exception as e:
        record_error("خطا در ماسک خشکی", e)
        return da
