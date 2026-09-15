# File Path: app.py
# Description: Multi-region PFZ WebGIS with dynamic UI controls for independent regional weights and thresholds.

import os
import sys
import tempfile
import zipfile
import traceback
import logging
import warnings
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

st.set_page_config(page_title="Multi-Region PFZ System", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if "error_logs" not in st.session_state:
    st.session_state.error_logs = []
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
for key in ["combined_fronts_gdf", "combined_region_gdf", "total_bounds"]:
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

st.title("🌊 سامانه هوشمند چندمنطقه‌ای تشخیص مناطق مستعد صید (PFZ)")

if st.session_state.error_logs:
    st.error("⚠️ خطاهایی در حین اجرای برنامه رخ داده است:")
    st.code("\n".join(st.session_state.error_logs), language="text")
    if st.button("🗑️ پاک‌کردن تاریخچه خطاها"):
        st.session_state.error_logs = []
        st.rerun()

st.sidebar.header("📁 بارگذاری داده‌های منطقه‌ای")

uploaded_zip = st.sidebar.file_uploader(
    "آپلود فایل ZIP حاوی شیپ‌فایل مناطق (مانند Persian_Gulf.shp, Oman_Sea.shp)", 
    type="zip"
)

output_dir = os.path.join(tempfile.gettempdir(), "Data_Processed_Multi")
os.makedirs(output_dir, exist_ok=True)

# استخراج شیپ‌فایل‌های موجود در ZIP و ساخت تنظیمات پویا
region_configs = {}
extracted_shp_paths = []

if uploaded_zip is not None:
    extract_path = os.path.join(output_dir, "extracted_shapes")
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    for root, _, files in os.walk(extract_path):
        for f in files:
            if f.endswith('.shp') and not f.startswith('._'):
                extracted_shp_paths.append(os.path.join(root, f))
    
    if extracted_shp_paths:
        st.sidebar.subheader("⚙️ تنظیمات اختصاصی هر منطقه")
        for shp_path in extracted_shp_paths:
            region_name = os.path.splitext(os.path.basename(shp_path))[0].replace("_", " ").title()
            
            with st.sidebar.expander(f"📌 تنظیمات: {region_name}", expanded=True):
                sst_w = st.slider(f"وزن SST ({region_name})", 0.0, 1.0, 0.6, 0.05, key=f"sst_{region_name}")
                chl_w = round(1.0 - sst_w, 2)
                st.caption(f"وزن کلروفیل-آ: **{chl_w}**")
                
                thresh = st.slider(f"آستانه جبهه‌یابی ({region_name})", 0.1, 1.0, 0.45, 0.05, key=f"thresh_{region_name}")
                
                region_configs[region_name] = {
                    "shp_path": shp_path,
                    "sst_weight": sst_w,
                    "chl_weight": chl_w,
                    "threshold": thresh
                }
    else:
        st.sidebar.error("هیچ فایل .shp معتبری در زیپ یافت نشد.")

def generate_region_fronts(nc_path, output_geojson_path, user_threshold, region_name):
    try:
        if not nc_path or not os.path.exists(nc_path):
            return None

        with xr.open_dataset(nc_path) as ds:
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key]
            
            lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
            lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            
            if da.ndim > 2:
                for d in [dim for dim in da.dims if dim not in [lat_name, lon_name]]:
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
        active_threshold = user_threshold if user_threshold < smooth_max else smooth_max * 0.80

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        fig, ax = plt.subplots()
        cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[active_threshold])
        extracted = []
        for segs in cs.allsegs:
            for poly in segs:
                if len(poly) > 2:
                    extracted.append(LineString(poly))
        plt.close(fig)

        if extracted:
            gdf_fronts = gpd.GeoDataFrame(geometry=extracted, crs="EPSG:4326")
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

if st.sidebar.button("🚀 اجرای تحلیل چندمنطقه‌ای"):
    if not region_configs:
        st.error("لطفاً فایل ZIP حاوی شیپ‌فایل‌های منطقه را آپلود کنید.")
    else:
        st.session_state.process_logs = []
        with st.status("🌐 پردازش مجزای مناطق اقیانوسی...", expanded=True) as status:
            try:
                # ۱. محاسبه Bounding Box کلی برای دریافت داده‌های ماهواره‌ای
                all_gdfs = []
                for reg_name, cfg in region_configs.items():
                    temp_gdf = gpd.read_file(cfg["shp_path"])
                    if temp_gdf.crs is not None and temp_gdf.crs != "EPSG:4326":
                        temp_gdf = temp_gdf.to_crs("EPSG:4326")
                    temp_gdf["Region"] = reg_name
                    all_gdfs.append(temp_gdf)
                
                combined_region_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
                minx, miny, maxx, maxy = combined_region_gdf.total_bounds
                
                log_process("info", "در حال دریافت داده‌های SST و CHL برای محدوده کلی...", status)
                sst_nc_path, chl_nc_path, _ = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)

                if sst_nc_path and chl_nc_path:
                    all_front_gdfs = []

                    # ۲. پردازش جداگانه برای هر منطقه با وزن‌ها و آستانه اختصاصی
                    for reg_name, cfg in region_configs.items():
                        log_process("info", f"در حال پردازش مستقل: **{reg_name}** (وزن SST: {cfg['sst_weight']} | آستانه: {cfg['threshold']})...", status)
                        
                        reg_out_dir = os.path.join(output_dir, reg_name.replace(" ", "_"))
                        nc_out, _, _ = process_pfz_pipeline(
                            cfg["shp_path"], sst_nc_path, chl_nc_path, reg_out_dir, 
                            cfg["sst_weight"], cfg["chl_weight"]
                        )
                        
                        if nc_out:
                            reg_fronts_gdf = generate_region_fronts(nc_out, None, cfg["threshold"], reg_name)
                            if reg_fronts_gdf is not None:
                                all_front_gdfs.append(reg_fronts_gdf)
                                log_process("success", f"جبهه‌های منطقه {reg_name} با موفقیت استخراج شد.", status)
                            else:
                                log_process("warning", f"جبهه‌ای در منطقه {reg_name} با آستانه تعیین‌شده یافت نشد.", status)

                    if all_front_gdfs:
                        st.session_state.combined_fronts_gdf = pd.concat(all_front_gdfs, ignore_index=True)
                    else:
                        st.session_state.combined_fronts_gdf = None

                    st.session_state.combined_region_gdf = combined_region_gdf
                    st.session_state.total_bounds = (minx, miny, maxx, maxy)
                    st.session_state.analysis_done = True
                    status.update(label="پردازش تمام مناطق با موفقیت کامل شد!", state="complete")
                else:
                    log_process("error", "فایل‌های ماهواره‌ای دریافت نشدند.", status)
                    status.update(label="خطا در دریافت داده‌ها", state="error")

            except Exception as global_ex:
                record_error("خطای سیستمی در تحلیل چندمنطقه‌ای", global_ex)
                status.update(label="توقف برنامه به دلیل خطا", state="error")

if st.session_state.process_logs:
    with st.expander("📝 گزارش مراحل پردازش", expanded=True):
        for msg_type, text in st.session_state.process_logs:
            if msg_type == "success": st.success(text)
            elif msg_type == "error": st.error(text)
            elif msg_type == "warning": st.warning(text)
            else: st.info(text)

if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.subheader("🗺️ نقشه یکپارچه مناطق و جبهه‌های استخراج‌شده")
    minx, miny, maxx, maxy = st.session_state.total_bounds
    
    m = folium.Map(location=[(miny + maxy)/2, (minx + maxx)/2], zoom_start=6, tiles="OpenStreetMap")
    
    # نمایش مرز مناطق با رنگ‌های متمایز
    folium.GeoJson(
        st.session_state.combined_region_gdf,
        name="مرز مناطق",
        style_function=lambda x: {'color': '#2B5B84', 'fillColor': '#2B5B84', 'fillOpacity': 0.05, 'weight': 2, 'dashArray': '4, 4'}
    ).add_to(m)

    # نمایش جبهه‌ها
    if st.session_state.combined_fronts_gdf is not None:
        folium.GeoJson(
            st.session_state.combined_fronts_gdf,
            name="خطوط جبهه صیادی (PFZ)",
            style_function=lambda x: {'color': '#E63946', 'weight': 3.5, 'opacity': 0.9},
            tooltip=folium.GeoJsonTooltip(fields=['Region', 'Length_km', 'Threshold'], aliases=['منطقه:', 'طول (km):', 'آستانه:'])
        ).add_to(m)
        st.success(f"🎯 در مجموع تعداد {len(st.session_state.combined_fronts_gdf)} خط جبهه در تمامی مناطق استخراج شد.")

    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)