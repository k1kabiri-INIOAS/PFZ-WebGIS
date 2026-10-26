# modules/fronts.py — استخراج جبهه ساده و پایدار (نزدیک به نسخهٔ اصلی)
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from shapely.geometry import LineString

from utils.logging_helpers import record_error


def generate_fronts_fallback(
    nc_path,
    user_threshold,
    region_name,
    sst_nc_path=None,
    region_shp_path=None,
):
    """
    استخراج خطوط جبهه از شاخص PFZ منطقه (مثل قبل).
    - یک سطح آستانه
    - نرم‌سازی سبک
    - فیلتر طول برای حذف نویز و خطوط سراسری
    - کلیپ به shapefile منطقه
    """
    try:
        if not nc_path or not os.path.exists(nc_path):
            return None

        with xr.open_dataset(nc_path) as ds:
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key].load().squeeze()
            # یکنواخت‌سازی نام ابعاد
            rename = {}
            for d in da.dims:
                dl = d.lower()
                if dl in ("longitude", "x") and d != "lon":
                    rename[d] = "lon"
                if dl in ("latitude", "y") and d != "lat":
                    rename[d] = "lat"
            if rename:
                da = da.rename(rename)
            while da.ndim > 2:
                da = da.isel({da.dims[0]: 0})

            lat_name = "lat" if "lat" in da.dims else [d for d in da.dims if "lat" in d.lower()][0]
            lon_name = "lon" if "lon" in da.dims else [d for d in da.dims if "lon" in d.lower()][0]
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            data = np.asarray(da.values, dtype=float)
            lats = da[lat_name].values
            lons = da[lon_name].values

        valid = np.isfinite(data) & (data > 0)
        if not valid.any():
            return None

        # فرسایش لبه برای کاهش جبههٔ ساحلی کاذب
        eroded = ndimage.binary_erosion(valid, structure=np.ones((3, 3)), iterations=2)
        if not eroded.any():
            eroded = valid

        fill = float(np.nanmean(data[valid]))
        filled = np.where(valid, data, fill)
        # نرم‌سازی کمی قوی‌تر → جبهه کمتر و تمیزتر
        smoothed = ndimage.gaussian_filter(filled, sigma=1.2)
        smoothed = np.where(eroded, smoothed, np.nan)

        vals = smoothed[eroded]
        if len(vals) == 0:
            return None

        smooth_max = float(np.nanmax(vals))
        # آستانه کاربر مستقیماً روی شاخص PFZ (مثل قبل)
        active = float(user_threshold)
        if active >= smooth_max:
            active = smooth_max * 0.85
        if active <= 0:
            active = float(np.nanpercentile(vals, 75))

        # اگر شبکه خیلی بزرگ است، برای کانتور نمونه‌برداری کن (سرعت)
        step = 1
        if data.shape[0] * data.shape[1] > 400_000:
            step = 2
        data_c = smoothed[::step, ::step]
        lats_c = lats[::step]
        lons_c = lons[::step]
        lon_grid, lat_grid = np.meshgrid(lons_c, lats_c)

        fig, ax = plt.subplots()
        try:
            cs = ax.contour(lon_grid, lat_grid, data_c, levels=[active])
            lines = [
                LineString(poly)
                for segs in cs.allsegs
                for poly in segs
                if len(poly) >= 5
            ]
        finally:
            plt.close(fig)

        if not lines:
            # یک بار با آستانه پایین‌تر امتحان کن
            active2 = max(active * 0.75, float(np.nanpercentile(vals, 60)))
            fig, ax = plt.subplots()
            try:
                cs = ax.contour(lon_grid, lat_grid, data_c, levels=[active2])
                lines = [
                    LineString(poly)
                    for segs in cs.allsegs
                    for poly in segs
                    if len(poly) >= 5
                ]
                active = active2
            finally:
                plt.close(fig)

        if not lines:
            return None

        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

        # کلیپ به منطقه
        if region_shp_path and os.path.exists(region_shp_path):
            try:
                region = gpd.read_file(region_shp_path)
                if region.crs is not None and str(region.crs) != "EPSG:4326":
                    region = region.to_crs("EPSG:4326")
                poly = region.geometry.buffer(0.03).unary_union
                gdf = gdf[gdf.intersects(poly)].copy()
                if gdf.empty:
                    return None
                gdf["geometry"] = gdf.intersection(poly)
                gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                gdf = gdf[gdf.geometry.type.isin(["LineString", "LinearRing"])]
                gdf = gdf[~gdf.is_empty & gdf.geometry.notna()]
            except Exception as e:
                print(f"Front clip warning: {e}")

        if gdf.empty:
            return None

        gdf["Length_km"] = gdf.to_crs("EPSG:3857").geometry.length / 1000.0
        # حذف نویز کوتاه و خطوط سراسری غیرواقعی
        gdf = gdf[(gdf["Length_km"] >= 8.0) & (gdf["Length_km"] <= 250.0)]
        if gdf.empty:
            return None

        gdf = gdf.copy()
        gdf["Region"] = region_name
        gdf["Threshold"] = active
        gdf = gdf.sort_values("Length_km", ascending=False).reset_index(drop=True)

        # حداکثر تعداد جبهه در هر منطقه (جلوگیری از شلوغی)
        max_fronts = 25
        if len(gdf) > max_fronts:
            gdf = gdf.head(max_fronts)

        return gdf

    except Exception as ex:
        record_error(f"خطا در استخراج جبهه برای {region_name}", ex)
    return None


def mask_pfz_by_fronts(da, fronts_gdf, buffer_deg=0.06):
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
