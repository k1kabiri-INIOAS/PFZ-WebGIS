# File Path: modules/processor.py
# Description: Ocean-only PFZ processing with land masking, ROI clipping, and file handle cleanup.

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import rioxarray

def process_pfz_pipeline(*args, **kwargs):
    all_args = list(args)
    shapefile_path = all_args[0] if len(all_args) > 0 else kwargs.get('shapefile_path')
    sst_nc_path = all_args[1] if len(all_args) > 1 else kwargs.get('sst_nc_path')
    chl_nc_path = all_args[2] if len(all_args) > 2 else kwargs.get('chl_nc_path')
    out_dir = all_args[3] if len(all_args) > 3 else kwargs.get('output_dir', 'outputs')
    sst_weight = float(all_args[4]) if len(all_args) > 4 else float(kwargs.get('sst_weight', 0.5))
    chl_weight = float(all_args[5]) if len(all_args) > 5 else float(kwargs.get('chl_weight', 0.5))

    os.makedirs(out_dir, exist_ok=True)
    nc_out = os.path.join(out_dir, "pfz_output.nc")
    tif_out = os.path.join(out_dir, "pfz_output.tif")
    fronts_geojson = os.path.join(out_dir, "pfz_fronts.geojson")

    try:
        # استفاده از Context Manager برای آزاد کردن قفل فایل‌ها
        with xr.open_dataset(sst_nc_path) as ds_sst_raw, xr.open_dataset(chl_nc_path) as ds_chl_raw:
            ds_sst = ds_sst_raw.load()
            ds_chl = ds_chl_raw.load()

        def standardize_ds(ds):
            rename_dict = {}
            for dim in ['longitude', 'x']:
                if dim in ds.dims: rename_dict[dim] = 'lon'
            for dim in ['latitude', 'y']:
                if dim in ds.dims: rename_dict[dim] = 'lat'
            if rename_dict:
                ds = ds.rename(rename_dict)
            return ds

        ds_sst = standardize_ds(ds_sst)
        ds_chl = standardize_ds(ds_chl)

        sst_var = next((v for v in ds_sst.data_vars if 'sst' in v.lower() or 'temp' in v.lower()), list(ds_sst.data_vars.keys())[0])
        chl_var = next((v for v in ds_chl.data_vars if 'chl' in v.lower()), list(ds_chl.data_vars.keys())[0])

        ds_chl = ds_chl.interp_like(ds_sst, method='nearest')

        sst_da = ds_sst[sst_var].squeeze()
        chl_da = ds_chl[chl_var].squeeze()

        while sst_da.ndim > 2: sst_da = sst_da[0]
        while chl_da.ndim > 2: chl_da = chl_da[0]

        # برش مکانی بر اساس شیپ‌فایل منطقه (در صورت وجود)
        if shapefile_path and os.path.exists(shapefile_path):
            try:
                gdf_roi = gpd.read_file(shapefile_path)
                if gdf_roi.crs is not None and gdf_roi.crs != "EPSG:4326":
                    gdf_roi = gdf_roi.to_crs("EPSG:4326")
                sst_da = sst_da.rio.write_crs("EPSG:4326").rio.clip(gdf_roi.geometry, gdf_roi.crs, drop=False)
                chl_da = chl_da.rio.write_crs("EPSG:4326").rio.clip(gdf_roi.geometry, gdf_roi.crs, drop=False)
            except Exception as clip_err:
                print(f"ROI Clipping Warning: {clip_err}")

        sst_vals = sst_da.values.copy()
        chl_vals = chl_da.values.copy()

        # ماسک‌سازی پهنه دریا (حذف تمام مقادیر خشکی)
        ocean_mask = ~np.isnan(sst_vals) & ~np.isnan(chl_vals)
        if not ocean_mask.any():
            raise ValueError("هیچ پیکسل دریایی معتبری در محدوده یافت نشد.")

        # محاسبه گرادیان حرارتی فقط در پهنه دریا
        sst_filled = np.where(ocean_mask, sst_vals, np.nanmean(sst_vals[ocean_mask]))
        dy, dx = np.gradient(sst_filled)
        sst_grad = np.sqrt(dx**2 + dy**2)
        sst_grad[~ocean_mask] = 0.0  # صفر کردن گرادیان روی خشکی و خط ساحلی

        # نرمال‌سازی گرادیان SST روی دریا
        grad_ocean = sst_grad[ocean_mask]
        g_min, g_max = np.nanmin(grad_ocean), np.nanmax(grad_ocean)
        norm_sst_grad = np.zeros_like(sst_grad)
        if g_max > g_min:
            norm_sst_grad[ocean_mask] = (sst_grad[ocean_mask] - g_min) / (g_max - g_min)

        # نرمال‌سازی کلروفیل روی دریا
        chl_ocean = chl_vals[ocean_mask]
        chl_log = np.log1p(np.maximum(chl_vals, 0))
        c_min, c_max = np.nanmin(chl_log[ocean_mask]), np.nanmax(chl_log[ocean_mask])
        norm_chl = np.zeros_like(chl_vals)
        if c_max > c_min:
            norm_chl[ocean_mask] = (chl_log[ocean_mask] - c_min) / (c_max - c_min)

        # محاسبه شاخص وزن‌دار PFZ فقط برای دریا
        tot_w = sst_weight + chl_weight
        tot_w = 1.0 if tot_w <= 0 else tot_w

        pfz_index_arr = np.full_like(sst_vals, np.nan)
        pfz_index_arr[ocean_mask] = (sst_weight * norm_sst_grad[ocean_mask] + chl_weight * norm_chl[ocean_mask]) / tot_w

        lon_arr = ds_sst.lon.values if 'lon' in ds_sst.coords else ds_sst.longitude.values
        lat_arr = ds_sst.lat.values if 'lat' in ds_sst.coords else ds_sst.latitude.values

        # استخراج کانتور فقط از پیکسل‌های دریا
        contour_data = np.nan_to_num(pfz_index_arr, nan=0.0)
        fig, ax = plt.subplots()
        cs = ax.contour(lon_arr, lat_arr, contour_data, levels=[0.45])
        
        lines = []
        for segs in cs.allsegs:
            for poly in segs:
                if len(poly) > 1:
                    lines.append(LineString(poly))
        plt.close(fig)

        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        gdf.to_file(fronts_geojson, driver="GeoJSON")

        # ذخیره NetCDF و GeoTIFF
        ds_out = xr.Dataset(
            {
                "pfz": (["lat", "lon"], pfz_index_arr),
                "pfz_index": (["lat", "lon"], pfz_index_arr)
            },
            coords={"lon": lon_arr, "lat": lat_arr}
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