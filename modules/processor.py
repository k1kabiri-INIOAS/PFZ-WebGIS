import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import rioxarray

def process_pfz_pipeline(shapefile_path, sst_nc_path, chl_nc_path):
    """
    پردازش داده‌های SST و Chlorophyll برای استخراج مناطق مستعد صید (PFZ)
    """
    try:
        # ۱. ایجاد پوشه خروجی
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)

        nc_out = os.path.join(out_dir, "pfz_output.nc")
        tif_out = os.path.join(out_dir, "pfz_output.tif")
        fronts_geojson = os.path.join(out_dir, "pfz_fronts.geojson")

        # ۲. بارگذاری داده‌های نت‌سی‌دی‌اف
        ds_sst = xr.open_dataset(sst_nc_path)
        ds_chl = xr.open_dataset(chl_nc_path)

        # استخراج نام متغیرها به صورت خودکار (پشتیبانی از نام‌های مختلف در منابع مختلف)
        sst_var = [v for v in ds_sst.data_vars if 'sst' in v.lower() or 'temp' in v.lower()][0]
        chl_var = [v for v in ds_chl.data_vars if 'chl' in v.lower()][0]

        # همگام‌سازی ابعاد کلروفیل با دمای سطح آب
        ds_chl = ds_chl.interp_like(ds_sst, method='nearest')

        sst_data = ds_sst[sst_var].squeeze().values
        chl_data = ds_chl[chl_var].squeeze().values
        
        lon_arr = ds_sst.lon.values
        lat_arr = ds_sst.lat.values

        # ۳. الگوریتم تشخیص جبهه‌ها (INCOIS Style)
        # محاسبه گرادیان حرارتی
        dy, dx = np.gradient(sst_data)
        sst_grad = np.sqrt(dx**2 + dy**2)

        # شرایط مطلوب برای PFZ (گرادیان بالای دما + وجود کلروفیل مناسب)
        # مقادیر آستانه را می‌توانید بر اساس منطقه خود تنظیم کنید
        pfz_arr = np.where((sst_grad > 0.05) & (chl_data >= 0.1) & (chl_data <= 5.0), 1, 0)

        # ۴. استخراج خطوط کانتور (جبهه‌ها)
        fig, ax = plt.subplots()
        levels = [0.5]
        cs = ax.contour(lon_arr, lat_arr, pfz_arr, levels=levels)
        
        lines = []
        
        # --- بخش اصلاح‌شده برای سازگاری با همه نسخه‌های Matplotlib ---
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
        # -----------------------------------------------------------
        
        plt.close(fig) # بستن پلات برای جلوگیری از نشت مموری

        # ۵. تولید فایل GeoJSON از جبهه‌ها
        if len(lines) > 0:
            gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf.to_file(fronts_geojson, driver="GeoJSON")
        else:
            # در صورتی که هیچ خطی پیدا نشد، یک فایل خالی معتبر می‌سازیم تا ارور ندهد
            gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
            gdf.to_file(fronts_geojson, driver="GeoJSON")

        # ۶. تولید فایل‌های NetCDF و GeoTIFF
        ds_out = xr.Dataset(
            {
                "pfz": (["lat", "lon"], pfz_arr)
            },
            coords={
                "lon": lon_arr,
                "lat": lat_arr,
            }
        )
        # تخصیص سیستم مختصات
        ds_out.rio.write_crs("epsg:4326", inplace=True)
        
        # ذخیره NetCDF
        ds_out.to_netcdf(nc_out)
        
        # ذخیره TIFF
        ds_out["pfz"].rio.to_raster(tif_out)

        # ۷. بازگرداندن دقیق ۳ خروجی
        return nc_out, tif_out, fronts_geojson

    except Exception as e:
        print(f"Error in PFZ pipeline: {e}")
        # در صورت بروز خطای پیش‌بینی نشده، ۳ متغیر خالی برمی‌گرداند تا جلوی کرش کردن app.py گرفته شود
        return None, None, None