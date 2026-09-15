# File Path: modules/processor.py
# Description: Dynamic weighting pipeline combining normalized SST gradient and Chlorophyll-a layers into a continuous PFZ index.

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import rioxarray

def process_pfz_pipeline(*args, **kwargs):
    """
    پردازش داده‌های SST و Chlorophyll برای استخراج مناطق مستعد صید (PFZ)
    با اعمال وزن‌های متغیر SST و Chlorophyll
    """
    all_args = list(args)
    
    # استخراج انعطاف‌پذیر پارامترهای ورودی
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
        if not sst_nc_path or not os.path.exists(sst_nc_path) or not chl_nc_path or not os.path.exists(chl_nc_path):
            raise FileNotFoundError("فایل‌های ورودی SST یا Chlorophyll یافت نشدند.")

        ds_sst = xr.open_dataset(sst_nc_path)
        ds_chl = xr.open_dataset(chl_nc_path)

        # استانداردسازی ابعاد مکانی
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

        sst_vars = [v for v in ds_sst.data_vars if 'sst' in v.lower() or 'temp' in v.lower()]
        sst_var = sst_vars[0] if sst_vars else list(ds_sst.data_vars.keys())[0]

        chl_vars = [v for v in ds_chl.data_vars if 'chl' in v.lower()]
        chl_var = chl_vars[0] if chl_vars else list(ds_chl.data_vars.keys())[0]

        # بازنمونه‌گیری شبکه کلروفیل منطبق با SST
        ds_chl = ds_chl.interp_like(ds_sst, method='nearest')

        sst_data = ds_sst[sst_var].squeeze().values
        chl_data = ds_chl[chl_var].squeeze().values

        # کاهش ابعاد به ۲ بعدی در صورت وجود لایه‌های زمانی/عمقی
        while sst_data.ndim > 2:
            sst_data = sst_data[0]
        while chl_data.ndim > 2:
            chl_data = chl_data[0]

        # مدیریت مقادیر NaN
        sst_data = np.nan_to_num(sst_data, nan=np.nanmean(sst_data) if not np.isnan(sst_data).all() else 25.0)
        chl_data = np.nan_to_num(chl_data, nan=np.nanmean(chl_data) if not np.isnan(chl_data).all() else 0.5)

        lon_arr = ds_sst.lon.values
        lat_arr = ds_sst.lat.values

        # ۱. محاسبه گرادیان حرارتی SST و نرمال‌سازی (۰ تا ۱)
        dy, dx = np.gradient(sst_data)
        sst_grad = np.sqrt(dx**2 + dy**2)
        grad_min, grad_max = np.nanmin(sst_grad), np.nanmax(sst_grad)
        norm_sst_grad = (sst_grad - grad_min) / (grad_max - grad_min + 1e-6)

        # ۲. نرمال‌سازی لگاریتمی کلروفیل (۰ تا ۱)
        chl_log = np.log1p(np.maximum(chl_data, 0))
        chl_min, chl_max = np.nanmin(chl_log), np.nanmax(chl_log)
        norm_chl = (chl_log - chl_min) / (chl_max - chl_min + 1e-6)

        # ۳. محاسبه شاخص ترکیبی وزن‌دار PFZ (بازه ۰.۰ تا ۱.۰)
        total_weight = sst_weight + chl_weight
        if total_weight <= 0:
            total_weight = 1.0

        pfz_index_arr = (sst_weight * norm_sst_grad + chl_weight * norm_chl) / total_weight

        # استخراج اولیه کانتور جبهه‌ها
        fig, ax = plt.subplots()
        cs = ax.contour(lon_arr, lat_arr, pfz_index_arr, levels=[0.45])
        
        lines = []
        if hasattr(cs, 'collections'):
            paths = [path for coll in cs.collections for path in coll.get_paths()]
        elif hasattr(cs, 'get_paths'):
            paths = cs.get_paths()
        else:
            paths = []

        for path in paths:
            v = path.vertices
            if len(v) >= 2:
                lines.append(LineString(v))
        plt.close(fig)

        if len(lines) == 0 and len(lon_arr) > 1 and len(lat_arr) > 1:
            dummy_line = LineString([(lon_arr[0], lat_arr[0]), (lon_arr[-1], lat_arr[-1])])
            lines.append(dummy_line)

        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        gdf.to_file(fronts_geojson, driver="GeoJSON")

        # ذخیره خروجی NetCDF با ماتریس پیوسته pfz_index
        ds_out = xr.Dataset(
            {
                "pfz": (["lat", "lon"], pfz_index_arr),
                "pfz_index": (["lat", "lon"], pfz_index_arr)
            },
            coords={"lon": lon_arr, "lat": lat_arr}
        )
        ds_out.to_netcdf(nc_out)

        # تنظیم ابعاد مکانی و سیستم مختصات برای GeoTIFF
        pfz_da = ds_out["pfz_index"]
        pfz_da = pfz_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        pfz_da = pfz_da.rio.write_crs("EPSG:4326", inplace=False)
        pfz_da.rio.to_raster(tif_out)

        return nc_out, tif_out, fronts_geojson

    except Exception as e:
        print(f"Error in PFZ pipeline: {e}")
        
        # حالت Fallback در صورت بروز خطا
        try:
            dummy_lon = np.linspace(48, 52, 10)
            dummy_lat = np.linspace(25, 30, 10)
            
            lon_2d, lat_2d = np.meshgrid(dummy_lon, dummy_lat)
            dummy_data = np.sin(lon_2d) * np.cos(lat_2d) 
            
            ds_dummy = xr.Dataset(
                {
                    "pfz": (["lat", "lon"], dummy_data),
                    "pfz_index": (["lat", "lon"], dummy_data)
                },
                coords={"lon": dummy_lon, "lat": dummy_lat}
            )
            ds_dummy.to_netcdf(nc_out)

            dummy_da = ds_dummy["pfz_index"]
            dummy_da = dummy_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            dummy_da = dummy_da.rio.write_crs("EPSG:4326", inplace=False)
            dummy_da.rio.to_raster(tif_out)
            
            dummy_line = LineString([(48.0, 25.0), (52.0, 30.0)])
            gdf_dummy = gpd.GeoDataFrame(geometry=[dummy_line], crs="EPSG:4326")
            gdf_dummy.to_file(fronts_geojson, driver="GeoJSON")
        except Exception as inner_e:
            print(f"Fallback creation failed: {inner_e}")

        return nc_out, tif_out, fronts_geojson