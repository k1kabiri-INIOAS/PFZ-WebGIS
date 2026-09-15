# File Path: app.py
# Description: Streamlit WebGIS application for Multi-Region Ocean PFZ mapping with dynamic per-region weightings and threshold controls.

import os
import sys
import tempfile
import zipfile
import traceback
import logging
import warnings
import pandas as pd
import streamlit as st
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import folium
from shapely.geometry import LineString
from streamlit_folium import st_folium
import scipy.ndimage as ndimage

from modules.fetcher import fetch_near_realtime_data
from modules.processor import process_pfz_pipeline

warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

st.set_page_config(page_title="PFZ Management System", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# مقداردهی متغیرهای Session State برای ماندگاری اطلاعات
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
for key in ["nc_out_list", "combined_fronts_gdf", "combined_region_gdf", "minx", "miny", "maxx", "maxy"]:
    if key not in st.session_state:
        st.session_state[key] = None

def record_error(msg, exc=None):
    full_msg = msg
    if exc:
        full_msg += f"\n{traceback.format_exc()}"
    print(f"[PFZ-LOG-ERROR] {full_msg}", flush=True)
    logging.error(full_msg)
    st.session_state.error_logs.append(full_msg)

def log_process(msg_type, msg_text, status_obj=None):
    st.session_state.process_logs.append((msg_type, msg_text))
    if status_obj:
        status_obj.write(msg_text)

st.title("🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ)")

# نمایش خطاهای سیستمی
if st.session_state.error_logs:
    st.error("⚠️ خطاهایی در حین اجرای برنامه رخ داده است:")
    all_logs_str = "\n".join(st.session_state.error_logs)
    st.code(all_logs_str, language="text")
    if st.button("🗑️ پاک‌کردن تاریخچه خطاها"):
        st.session_state.error_logs = []
        st.rerun()

st.sidebar.header("تنظیمات پردازش و مدل")

uploaded_shapefile_zip = st.sidebar.file_uploader(
    "آپلود فایل فشرده شیپ‌فایل مناطق (.zip)", 
    type="zip"
)

output_dir = os.path.join(tempfile.gettempdir(), "Data_Processed")
os.makedirs(output_dir, exist_ok=True)

# استخراج فایل‌های شیپ‌فایل و ساخت منوی پویا برای هر منطقه
region_configs = {}
if uploaded_shapefile_zip is not None:
    extract_path = os.path.join(output_dir, "extracted_shapes")
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(uploaded_shapefile_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    shp_files = []
    for r, d, files in os.walk(extract_path):
        for f in files:
            if f.endswith('.shp') and not f.startswith('._'):
                shp_files.append(os.path.join(r, f))
    
    if shp_files:
        st.sidebar.subheader("⚙️ تنظیمات اختصاصی هر منطقه")
        for shp_path in sorted(shp_files):
            reg_name = os.path.splitext(os.path.basename(shp_path))[0].replace("_", " ").title()
            
            with st.sidebar.expander(f"📌 {reg_name}", expanded=True):
                sst_w = st.slider(f"وزن SST ({reg_name})", 0.0, 1.0, 0.6, 0.05, key=f"sst_{reg_name}")
                chl_w = round(1.0 - sst_w, 2)
                st.caption(f"وزن کلروفیل-آ: **{chl_w}**")
                
                thresh = st.slider(f"آستانه حساسیت ({reg_name})", 0.1, 1.0, 0.50, 0.05, key=f"thresh_{reg_name}")
                
                region_configs[reg_name] = {
                    "shp_path": shp_path,
                    "sst_weight": sst_w,
                    "chl_weight": chl_w,
                    "threshold": thresh
                }
    else:
        st.sidebar.error("هیچ فایل .shp معتبری در فایل ZIP یافت نشد.")

def generate_fronts_fallback(nc_path, user_threshold, region_name):
    try:
        if not nc_path or not os.path.exists(nc_path):
            record_error(f"فایل NetCDF وجود ندارد: {nc_path}")
            return None

        with xr.open_dataset(nc_path) as ds:
            if "pfz_index" not in ds:
                record_error("متغیر 'pfz_index' در فایل NetCDF یافت نشد.")
                return None
            
            da = ds["pfz_index"]
            lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
            lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            
            if not lat_name or not lon_name:
                record_error("ابعاد مکانی (lat/lon) به درستی در فایل NetCDF یافت نشد.")
                return None
                
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            
            if da.ndim > 2:
                non_spatial_dims = [d for d in da.dims if d not in [lat_name, lon_name]]
                for d in non_spatial_dims:
                    da = da.isel({d: 0})
            
            data = da.values.copy()
            
        valid_mask = ~np.isnan(data) & (data > 0)
        if not valid_mask.any():
            return None

        mean_val = float(np.nanmean(data[valid_mask]))
        data_filled = np.where(valid_mask, data, mean_val)
        data_smoothed = ndimage.gaussian_filter(data_filled, sigma=1.0).astype(float)

        eroded_mask = ndimage.binary_erosion(valid_mask, structure=np.ones((3, 3)), iterations=1)
        if not eroded_mask.any():
            eroded_mask = valid_mask

        data_smoothed[~eroded_mask] = np.nan
        valid_smoothed = data_smoothed[eroded_mask]
        if len(valid_smoothed) == 0:
            return None

        smooth_max = float(np.nanmax(valid_smoothed))
        active_threshold = user_threshold
        if active_threshold >= smooth_max:
            active_threshold = smooth_max * 0.80

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = []
            for segs in cs.allsegs:
                for poly in segs:
                    if len(poly) > 2:
                        extracted.append(LineString(poly))
            plt.close(fig)
            return extracted

        lines = extract_lines(active_threshold)
        if not lines:
            fallback_percents = [0.60, 0.40, 0.20]
            for pct in fallback_percents:
                test_t = smooth_max * pct
                if test_t <= 0: continue
                lines = extract_lines(test_t)
                if lines:
                    active_threshold = test_t
                    break
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Region'] = region_name
            gdf_fronts['Threshold'] = active_threshold
            gdf_fronts = gdf_fronts.to_crs("EPSG:3857")
            gdf_fronts['Length_km'] = gdf_fronts.geometry.length / 1000
            gdf_fronts = gdf_fronts.to_crs("EPSG:4326")
            
            gdf_fronts = gdf_fronts[gdf_fronts['Length_km'] > 0.5]
            return gdf_fronts if not gdf_fronts.empty else None
            
    except Exception as ex:
        record_error(f"خطا در استخراج جبهه برای منطقه {region_name}", ex)
    return None

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if not region_configs:
        st.error("لطفاً فایل فشرده شیپ‌فایل مناطق (.zip) را آپلود کنید.")
    else:
        st.session_state.process_logs = []
        with st.status("🚀 شروع فرآیند پردازش داده‌های مکانی چندمنطقه‌ای...", expanded=True) as status:
            try:
                # ۱. ادغام محدوده مناطق برای محاسبه Bounding Box کلی
                log_process("info", "در حال استخراج و خواندن شیپ‌فایل‌های منطقه‌ای...", status)
                all_gdfs = []
                for reg_name, cfg in region_configs.items():
                    temp_gdf = gpd.read_file(cfg["shp_path"])
                    if temp_gdf.crs is not None and temp_gdf.crs != "EPSG:4326":
                        temp_gdf = temp_gdf.to_crs("EPSG:4326")
                    temp_gdf["Region"] = reg_name
                    all_gdfs.append(temp_gdf)

                combined_region_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
                minx, miny, maxx, maxy = combined_region_gdf.total_bounds

                # ۲. دریافت یکباره داده‌های ماهواره‌ای برای محدوده کلی
                log_process("info", "در حال برقراری ارتباط با سرور و دریافت داده‌های SST و CHL کلی...", status)
                try:
                    sst_nc_path, chl_nc_path, _ = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                except Exception as fetch_ex:
                    record_error("خطا در ماژول fetch_near_realtime_data", fetch_ex)
                    sst_nc_path, chl_nc_path = None, None

                if sst_nc_path and chl_nc_path:
                    log_process("success", "داده‌های ماهواره‌ای با موفقیت دریافت شدند.", status)
                    
                    all_front_gdfs = []
                    nc_out_list = []

                    # ۳. پردازش جداگانه هر منطقه با وزن‌ها و آستانه‌های اختصاصی
                    for reg_name, cfg in region_configs.items():
                        log_process("info", f"در حال پردازش **{reg_name}** (وزن SST: {cfg['sst_weight']} | آستانه: {cfg['threshold']})...", status)
                        
                        reg_out_dir = os.path.join(output_dir, reg_name.replace(" ", "_"))
                        try:
                            nc_out, _, _ = process_pfz_pipeline(
                                cfg["shp_path"], sst_nc_path, chl_nc_path, reg_out_dir, 
                                cfg["sst_weight"], cfg["chl_weight"]
                            )
                        except Exception as proc_ex:
                            record_error(f"خطا در پردازش مدل منطقه {reg_name}", proc_ex)
                            nc_out = None

                        if nc_out:
                            nc_out_list.append((reg_name, nc_out, cfg["shp_path"]))
                            reg_fronts_gdf = generate_fronts_fallback(nc_out, cfg["threshold"], reg_name)
                            if reg_fronts_gdf is not None:
                                all_front_gdfs.append(reg_fronts_gdf)
                                log_process("success", f"جبهه‌های منطقه {reg_name} استخراج گردید.", status)
                            else:
                                log_process("warning", f"جبهه‌ای در منطقه {reg_name} یافت نشد.", status)

                    st.session_state.combined_fronts_gdf = pd.concat(all_front_gdfs, ignore_index=True) if all_front_gdfs else None
                    st.session_state.combined_region_gdf = combined_region_gdf
                    st.session_state.nc_out_list = nc_out_list
                    st.session_state.minx, st.session_state.miny, st.session_state.maxx, st.session_state.maxy = minx, miny, maxx, maxy
                    st.session_state.analysis_done = True
                    status.update(label="تمام مراحل پردازش با موفقیت به پایان رسید!", state="complete")
                else:
                    log_process("error", "فایل‌های SST یا CHL دریافت نشدند.", status)
                    status.update(label="پردازش متوقف شد", state="error")

            except Exception as global_ex:
                record_error("خطای کلی در جریان اجرای برنامه", global_ex)
                log_process("error", f"خطای سیستمی رخ داد: {global_ex}", status)
                status.update(label="اجرای برنامه با خطا متوقف شد", state="error")

if st.session_state.process_logs:
    with st.expander("📝 گزارش مراحل پردازش", expanded=True):
        for msg_type, text in st.session_state.process_logs:
            if msg_type == "success":
                st.success(text)
            elif msg_type == "error":
                st.error(text)
            elif msg_type == "warning":
                st.warning(text)
            else:
                st.info(text)

if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه و لایه پس‌زمینه")
    
    m = folium.Map(
        location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], 
        zoom_start=6, tiles="OpenStreetMap"
    )
    
    # رسم مرز مناطق
    folium.GeoJson(
        st.session_state.combined_region_gdf,
        name="Region Boundaries",
        style_function=lambda x: {'color': '#0000FF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'},
        tooltip=folium.GeoJsonTooltip(fields=['Region'], aliases=['منطقه:'])
    ).add_to(m)
    
    # رسم خطوط جبهه‌ها
    if st.session_state.combined_fronts_gdf is not None and not st.session_state.combined_fronts_gdf.empty:
        folium.GeoJson(
            st.session_state.combined_fronts_gdf,
            name="PFZ Front Lines",
            style_function=lambda x: {'color': '#FF0000', 'weight': 3.5, 'opacity': 1.0},
            tooltip=folium.GeoJsonTooltip(fields=['Region', 'Length_km', 'Threshold'], aliases=['منطقه:', 'طول (km):', 'آستانه:'])
        ).add_to(m)
        st.success(f"🎯 تعداد {len(st.session_state.combined_fronts_gdf)} جبهه صیادی در مجموع مناطق استخراج و رسم شد.")

    m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)