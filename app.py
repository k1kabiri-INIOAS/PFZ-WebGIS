# File Path: app.py
# Description: Streamlit WebGIS application for Multi-Region Ocean PFZ mapping with accurate Heatmap Geo-referencing and dual (Jalali/Gregorian) date displays.

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

# تابع تبدیل تاریخ میلادی به شمسی (بدون نیاز به کتابخانه خارجی)
def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy if gm > 2 else gy - 1
    days = (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) + g_d_m[gm - 1] + gd - 1
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + (days // 31)
        jd = 1 + (days % 31)
    else:
        jm = 7 + ((days - 186) // 30)
        jd = 1 + ((days - 186) % 30)
    return jy, jm, jd

def parse_date_formats(date_str):
    if not date_str:
        return None, None
    try:
        dt = pd.to_datetime(date_str)
        greg_str = dt.strftime("%Y-%m-%d")
        jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
        jalali_str = f"{jy}/{jm:02d}/{jd:02d}"
        return greg_str, jalali_str
    except Exception:
        return str(date_str), None

# مقداردهی متغیرهای Session State برای ماندگاری اطلاعات
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
for key in ["nc_out_list", "combined_fronts_gdf", "combined_region_gdf", "minx", "miny", "maxx", "maxy", "latest_date"]:
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

                log_process("info", "در حال برقراری ارتباط با سرور و دریافت داده‌های SST و CHL کلی...", status)
                try:
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                except Exception as fetch_ex:
                    record_error("خطا در ماژول fetch_near_realtime_data", fetch_ex)
                    sst_nc_path, chl_nc_path, latest_date = None, None, None

                if sst_nc_path and chl_nc_path:
                    greg_d, jalali_d = parse_date_formats(latest_date)
                    log_process("success", f"داده‌های ماهواره‌ای با موفقیت دریافت شدند. (تاریخ شمسی: {jalali_d} | Date: {greg_d})", status)
                    
                    all_front_gdfs = []
                    nc_out_list = []

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
                    st.session_state.latest_date = latest_date
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
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه و لایه‌های نقشه حرارتی (Heatmap)")
    
    greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
    
    # نمایش تاریخ‌های شمسی و میلادی بالای نقشه
    if greg_str and jalali_str:
        st.info(f"📅 **تاریخ اخذ داده:** `{jalali_str}` | **Date:** `{greg_str}`")

    m = folium.Map(
        location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], 
        zoom_start=6, tiles="OpenStreetMap"
    )
    
    # ۱. تولید و افزودن دقیق لایه Heatmap با جئو‌رفرنس صحیح برای هر منطقه
    if st.session_state.nc_out_list:
        for reg_name, nc_out, reg_shp_path in st.session_state.nc_out_list:
            if nc_out and os.path.exists(nc_out):
                try:
                    with xr.open_dataset(nc_out) as ds_res:
                        var_key = "pfz_index" if "pfz_index" in ds_res else list(ds_res.data_vars.keys())[0]
                        pfz_da = ds_res[var_key].load()
                        
                        lat_name_plot = next((d for d in pfz_da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
                        lon_name_plot = next((d for d in pfz_da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
                        
                        if da_dim := [d for d in pfz_da.dims if d not in [lat_name_plot, lon_name_plot]]:
                            for d in da_dim:
                                pfz_da = pfz_da.isel({d: 0})
                        
                        lats = pfz_da[lat_name_plot].values
                        lons = pfz_da[lon_name_plot].values
                        
                        grid_minx, grid_maxx = float(np.nanmin(lons)), float(np.nanmax(lons))
                        grid_miny, grid_maxy = float(np.nanmin(lats)), float(np.nanmax(lats))

                        data_arr = pfz_da.values.copy()
                        
                        # در صورت معکوس بودن محور lat، تصویر جهت درستی داشته باشد
                        if lats[0] > lats[-1]:
                            data_arr = np.flipud(data_arr)

                        # ساخت شکل دقیق بدون Margin برای جئورفرنس دقیق
                        fig = plt.figure(figsize=(10, 10), dpi=200)
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.set_axis_off()
                        fig.patch.set_alpha(0)
                        
                        ax.imshow(
                            data_arr, 
                            cmap="jet", 
                            extent=[grid_minx, grid_maxx, grid_miny, grid_maxy], 
                            origin='lower', 
                            aspect='auto',
                            interpolation='nearest'
                        )
                        ax.set_xlim(grid_minx, grid_maxx)
                        ax.set_ylim(grid_miny, grid_maxy)
                        
                        overlay_path = os.path.join(output_dir, f"pfz_overlay_{reg_name.replace(' ', '_')}.png")
                        fig.savefig(overlay_path, dpi=200, transparent=True)
                        plt.close(fig)
                        
                        folium.raster_layers.ImageOverlay(
                            image=overlay_path,
                            bounds=[[grid_miny, grid_minx], [grid_maxy, grid_maxx]],
                            opacity=0.6,
                            name=f"Heatmap Index ({reg_name})"
                        ).add_to(m)
                except Exception as img_ex:
                    record_error(f"خطا در رندر تصویر Heatmap برای {reg_name}", img_ex)

    # ۲. رسم مرز مناطق
    folium.GeoJson(
        st.session_state.combined_region_gdf,
        name="Region Boundaries",
        style_function=lambda x: {'color': '#0000FF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'},
        tooltip=folium.GeoJsonTooltip(fields=['Region'], aliases=['منطقه:'])
    ).add_to(m)
    
    # ۳. رسم خطوط جبهه‌ها
    if st.session_state.combined_fronts_gdf is not None and not st.session_state.combined_fronts_gdf.empty:
        folium.GeoJson(
            st.session_state.combined_fronts_gdf,
            name="PFZ Front Lines",
            style_function=lambda x: {'color': '#FF0000', 'weight': 3.5, 'opacity': 1.0},
            tooltip=folium.GeoJsonTooltip(fields=['Region', 'Length_km', 'Threshold'], aliases=['منطقه:', 'طول (km):', 'آستانه:'])
        ).add_to(m)
        st.success(f"🎯 تعداد {len(st.session_state.combined_fronts_gdf)} جبهه صیادی در مجموع مناطق استخراج و رسم شد.")

    # ۴. کادر شناور روی نقشه با نمایش همزمان تاریخ شمسی و میلادی
    if greg_str and jalali_str:
        date_box_html = f'''
            <div style="position: fixed; 
                        bottom: 25px; left: 20px; width: 220px; height: 50px; 
                        z-index:9999; font-size:12px; background-color: rgba(255, 255, 255, 0.92); 
                        border: 2px solid #2B5B84; border-radius: 6px; 
                        padding: 4px; font-weight: bold; text-align: center; color: #1E3A8A; line-height: 1.4;">
                تاریخ اخذ داده: {jalali_str}<br>
                <span style="font-family: Arial, sans-serif; color: #333333;">Date: {greg_str}</span>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(date_box_html))

    m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)