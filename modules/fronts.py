# modules/fronts.py — استخراج و ماسک جبهه‌های صیادی
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from shapely.geometry import LineString

from utils.logging_helpers import record_error


def generate_fronts_fallback(nc_path, user_threshold, region_name):
    """استخراج خطوط جبهه از فایل NetCDF شاخص PFZ."""
    try:
        if not nc_path or not __import__("os").path.exists(nc_path):
            return None
        with xr.open_dataset(nc_path) as ds:
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key].load()
            lat_name = next(
                (d for d in da.dims if d.lower() in ["lat", "latitude", "y"]), None
            )
            lon_name = next(
                (d for d in da.dims if d.lower() in ["lon", "longitude", "x"]), None
            )
            if not lat_name or not lon_name:
                return None
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            if da.ndim > 2:
                for d in [d for d in da.dims if d not in [lat_name, lon_name]]:
                    da = da.isel({d: 0})
            data = da.values.copy()
            lats = da[lat_name].values
            lons = da[lon_name].values

        valid_mask = ~np.isnan(data) & (data > 0)
        if not valid_mask.any():
            return None

        data_smoothed = ndimage.gaussian_filter(
            np.where(valid_mask, data, float(np.nanmean(data[valid_mask]))),
            sigma=1.0,
        ).astype(float)
        eroded_mask = ndimage.binary_erosion(
            valid_mask, structure=np.ones((3, 3)), iterations=1
        )
        if not eroded_mask.any():
            eroded_mask = valid_mask

        data_smoothed[~eroded_mask] = np.nan
        valid_smoothed = data_smoothed[eroded_mask]
        if len(valid_smoothed) == 0:
            return None

        smooth_max = float(np.nanmax(valid_smoothed))
        active_threshold = (
            user_threshold if user_threshold < smooth_max else smooth_max * 0.80
        )
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = [
                LineString(poly)
                for segs in cs.allsegs
                for poly in segs
                if len(poly) > 2
            ]
            plt.close(fig)
            return extracted

        lines = (
            extract_lines(active_threshold)
            or extract_lines(smooth_max * 0.60)
            or extract_lines(smooth_max * 0.40)
        )

        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts["Region"] = region_name
            gdf_fronts["Threshold"] = active_threshold
            gdf_fronts["Length_km"] = (
                gdf_fronts.to_crs("EPSG:3857").geometry.length / 1000
            )
            filtered = gdf_fronts[gdf_fronts["Length_km"] > 0.5]
            return filtered if not filtered.empty else None
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
        land_union = land_gdf.unary_union

        try:
            from shapely.vectorized import contains

            land_mask = contains(land_union, lon_grid, lat_grid)
        except ImportError:
            from shapely.geometry import Point
            from shapely.prepared import prep

            prep_poly = prep(land_union)
            land_mask = np.array(
                [
                    prep_poly.contains(Point(x, y))
                    for x, y in zip(lon_grid.ravel(), lat_grid.ravel())
                ]
            ).reshape(lon_grid.shape)

        da_out = da.copy()
        da_out.values[land_mask] = np.nan
        return da_out
    except Exception as e:
        record_error("خطا در ماسک خشکی", e)
        return da
