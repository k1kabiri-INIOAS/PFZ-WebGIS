import os
import sys
import tempfile
import zipfile
import traceback
import logging
import streamlit as st
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import folium
from shapely.geometry import LineString
from streamlit_folium import st_folium
import scipy.ndimage as ndimage
import warnings

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
    st.session_state.process_logs = []  # برای ذخیره و نمایش دائمی پیغام‌های موفقیت/وضعیت
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
for key in ["nc_out", "tif_out", "fronts_geojson", "gdf", "minx", "miny", "maxx", "maxy"]:
    if key not in st.session_state:
        st.session_state[key] = None

def record_error(msg, exc=None):
    full_msg = msg
    if exc:
        full_msg += f"\n{traceback.format_exc()}"
    print(f"[PFZ-LOG-ERROR] {full_msg}", flush=True)
    logging.error(full_msg)
    st.session_state.error_logs.append(full_msg)

# تابع کمکی برای ثبت و لاگ کردن همزمان در Status و Session State
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
    "آپلود فایل فشرده شیپ‌فایل منطقه (.zip)", 
    type="zip"
)

# استفاده از مسیر موقت و ایمن برای جلوگیری از خطای Permission Denied
output_dir = os.path.join(tempfile.gettempdir(), "Data_Processed")
os.makedirs(output_dir, exist_ok=True)

st.sidebar.subheader("پارامترهای مدل")
sst_weight = st.sidebar.slider("وزن جبهه‌های حرارتی SST", 0.0, 1.0, 0.5, 0.1)
chl_weight = st.sidebar.slider("وزن کلروفیل-آ (Chlorophyll-a)", 0.0, 1.0, 0.5, 0.1)

st.sidebar.subheader("تنظیمات استخراج عوارض")
pfz_threshold = st.sidebar.slider("آستانه حساسیت جبهه‌ها (Threshold)", 0.1, 1.0, 0.45, 0.05)

def generate_fronts_fallback(nc_path, output_geojson_path, user_threshold):
    try:
        if not nc_path or not os.path.exists(nc_path):
            record_error(f"فایل NetCDF وجود ندارد: {nc_path}")
            return False

        ds = xr.open_dataset(nc_path)
        if "pfz_index" not in ds:
            record_error("متغیر 'pfz_index' در فایل NetCDF یافت نشد.")
            return False
        
        da = ds["pfz_index"]
        
        # شناسایی ایمن ابعاد مکانی صرف نظر از ترتیب آنها
        lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
        lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        
        if not lat_name or not lon_name:
            record_error("ابعاد مکانی (lat/lon) به درستی در فایل NetCDF یافت نشد.")
            return False
            
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
        # حذف ایمن ابعاد غیرمکانی (مانند time) در صورت وجود
        if da.ndim > 2:
            non_spatial_dims = [d for d in da.dims if d not in [lat_name, lon_name]]
            for d in non_spatial_dims:
                da = da.isel({d: 0})
        
        data = da.values
            
        valid_mask = ~np.isnan(data)
        if not valid_mask.any():
            record_error("ماتریس محاسباتی pfz_index کلاً از داده‌های NaN تشکیل شده است.")
            return False
            
        raw_max = float(np.nanmax(data))
        raw_min = float(np.nanmin(data))
        
        if np.isnan(raw_max) or raw_max == raw_min:
            record_error("داده‌های ماتریس یکنواخت هستند و امکان استخراج جبهه (کانتور) وجود ندارد.")
            return False

        data_filled = np.nan_to_num(data, nan=0.0)
        data_smoothed = ndimage.gaussian_filter(data_filled, sigma=1.0)
        data_smoothed[~valid_mask] = np.nan
        
        valid_smoothed = data_smoothed[valid_mask]
        smooth_max = float(np.nanmax(valid_smoothed))
        
        active_threshold = user_threshold
        if active_threshold >= smooth_max:
            active_threshold = smooth_max * 0.85

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = []
            
            # استفاده از ساختار جدید Matplotlib >= 3.8 جهت جلوگیری از خطای collections
            for segs in cs.allsegs:
                for poly in segs:
                    if len(poly) > 1:
                        extracted.append(LineString(poly))
            
            plt.close(fig)
            return extracted

        lines = extract_lines(active_threshold)
        
        if not lines:
            fallback_percents = [0.70, 0.50, 0.30, 0.10]
            for pct in fallback_percents:
                test_t = smooth_max * pct
                if test_t <= 0: continue
                lines = extract_lines(test_t)
                if lines:
                    active_threshold = test_t
                    break
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Threshold'] = active_threshold
            gdf_fronts = gdf_fronts.to_crs("EPSG:3857")
            gdf_fronts['Length_km'] = gdf_fronts.geometry.length / 1000
            gdf_fronts = gdf_fronts.to_crs("EPSG:4326")
            
            gdf_fronts = gdf_fronts[gdf_fronts['Length_km'] > 0.5]
            
            if not gdf_fronts.empty:
                os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
                gdf_fronts.to_file(output_geojson_path, driver="GeoJSON")
                return True
            else:
                record_error("خطوط جبهه یافت شدند اما همگی کوتاه‌تر از 0.5 کیلومتر بوده و فیلتر شدند.")
                return False
        else:
            record_error(f"هیچ خط کانتوری در آستانه‌های مختلف پیدا نشد.")
            return False
            
    except Exception as ex:
        record_error("خطای غیرمنتظره در generate_fronts_fallback", ex)
    return False

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if uploaded_shapefile_zip is None:
        st.error("لطفاً فایل فشرده شیپ‌فایل منطقه (.zip) را آپلود کنید.")
    else:
        st.session_state.process_logs = []  # پاک کردن لاگ‌های قبلی
        with st.status("🚀 شروع فرآیند پردازش داده‌های مکانی...", expanded=True) as status:
            try:
                # مرحله ۱: استخراج شیپ‌فایل
                log_process("info", "در حال استخراج و خواندن فایل شیپ‌فایل منطقه...", status)
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(uploaded_shapefile_zip, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    
                    shp_files = [os.path.join(r, f) for r, d, files in os.walk(tmpdir) for f in files if f.endswith('.shp')]
                    
                    if not shp_files:
                        record_error("فایل .shp در داخل فایل ZIP پیدا نشد.")
                        log_process("error", "فایل شیپ‌فایل یافت نشد.", status)
                        status.update(label="پردازش متوقف شد", state="error")
                    else:
                        log_process("success", "فایل منطقه با موفقیت بارگذاری شد.", status)
                        shapefile_path = shp_files[0]
                        
                        gdf = gpd.read_file(shapefile_path)
                        if gdf.crs is not None and gdf.crs != "EPSG:4326":
                            gdf = gdf.to_crs("EPSG:4326")
                        minx, miny, maxx, maxy = gdf.total_bounds
                        
                        # مرحله ۲: دریافت داده‌های ماهواره‌ای
                        log_process("info", "در حال برقراری ارتباط با سرور و دریافت داده‌های SST و CHL...", status)
                        try:
                            sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                        except Exception as fetch_ex:
                            record_error("خطا در ماژول fetch_near_realtime_data", fetch_ex)
                            sst_nc_path, chl_nc_path = None, None
                        
                        if sst_nc_path and chl_nc_path:
                            log_process("success", "داده‌های ماهواره‌ای با موفقیت دریافت شدند.", status)
                            
                            # مرحله ۳: پردازش مدل
                            log_process("info", "در حال پردازش مدل و محاسبه شاخص PFZ...", status)
                            try:
                                nc_out, tif_out, fronts_geojson = process_pfz_pipeline(
                                    shapefile_path, sst_nc_path, chl_nc_path, output_dir, sst_weight, chl_weight
                                )
                            except Exception as proc_ex:
                                record_error("خطا در ماژول process_pfz_pipeline", proc_ex)
                                nc_out, fronts_geojson = None, None

                            if nc_out:
                                log_process("success", "مدل شاخص PFZ با موفقیت پردازش شد.", status)
                                
                                # مرحله ۴: استخراج جبهه‌ها
                                log_process("info", "در حال استخراج خطوط جبهه‌های حرارتی...", status)
                                target_geojson = os.path.join(output_dir, "pfz_fronts.geojson")
                                fallback_success = generate_fronts_fallback(nc_out, target_geojson, user_threshold=pfz_threshold)
                                
                                if fallback_success:
                                    fronts_geojson = target_geojson
                                    log_process("success", "جبهه‌های صیادی استخراج و فایل GeoJSON تولید شد.", status)
                                else:
                                    log_process("warning", "پردازش پایان یافت اما هیچ جبهه‌ای استخراج نشد.", status)
                                
                                # نهایی‌سازی استیت‌ها
                                st.session_state.analysis_done = True
                                st.session_state.nc_out = nc_out
                                st.session_state.fronts_geojson = fronts_geojson
                                st.session_state.gdf = gdf
                                st.session_state.minx, st.session_state.miny, st.session_state.maxx, st.session_state.maxy = minx, miny, maxx, maxy
                                status.update(label="تمام مراحل پردازش با موفقیت به پایان رسید!", state="complete")
                            else:
                                log_process("error", "خطا در خروجی‌های پردازش مدل رخ داد.", status)
                                status.update(label="پردازش متوقف شد", state="error")
                        else:
                            log_process("error", "فایل‌های SST یا CHL دریافت نشدند (ارور سرور یا عدم وجود داده).", status)
                            record_error("فایل‌های SST یا CHL دریافت نشدند.")
                            status.update(label="پردازش متوقف شد", state="error")
                            
            except Exception as global_ex:
                record_error("خطای کلی در جریان اجرای برنامه", global_ex)
                log_process("error", f"خطای سیستمی رخ داد: {global_ex}", status)
                status.update(label="اجرای برنامه با خطا متوقف شد", state="error")

# نمایش دائمی پیغام‌های مراحل پردازش پس از اتمام (محو نمی‌شوند)
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

if st.session_state.analysis_done and st.session_state.gdf is not None:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه و لایه پس‌زمینه")
    
    m = folium.Map(
        location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], 
        zoom_start=6, tiles="OpenStreetMap"
    )
    
    folium.GeoJson(
        st.session_state.gdf,
        name="Region Boundary",
        style_function=lambda x: {'color': '#0000FF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'}
    ).add_to(m)
    
    if st.session_state.nc_out and os.path.exists(st.session_state.nc_out):
        try:
            ds_res = xr.open_dataset(st.session_state.nc_out)
            if "pfz_index" in ds_res:
                pfz_da = ds_res["pfz_index"]
                
                # مدیریت ابعاد اضافی در زمان نمایش نقشه
                if pfz_da.ndim > 2:
                    lat_name_plot = next((d for d in pfz_da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
                    lon_name_plot = next((d for d in pfz_da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
                    non_spatial_dims = [d for d in pfz_da.dims if d not in [lat_name_plot, lon_name_plot]]
                    for d in non_spatial_dims:
                        pfz_da = pfz_da.isel({d: 0})
                
                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                pfz_da.plot.imshow(ax=ax, cmap="jet", alpha=0.5, add_colorbar=False)
                
                overlay_path = os.path.join(output_dir, "pfz_overlay.png")
                fig.savefig(overlay_path, dpi=150, transparent=True, pad_inches=0)
                plt.close(fig)
                
                folium.raster_layers.ImageOverlay(
                    image=overlay_path,
                    bounds=[[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]],
                    opacity=0.6,
                    name="PFZ Index Heatmap"
                ).add_to(m)
        except Exception as img_ex:
            record_error("خطا در رندر تصویر Heatmap روی نقشه", img_ex)

    if st.session_state.fronts_geojson and os.path.exists(st.session_state.fronts_geojson):
        try:
            fronts_gdf = gpd.read_file(st.session_state.fronts_geojson)
            if not fronts_gdf.empty:
                folium.GeoJson(
                    fronts_gdf,
                    name="PFZ Front Lines",
                    style_function=lambda x: {'color': '#FF0000', 'weight': 4.0, 'opacity': 1.0}
                ).add_to(m)
                st.success(f"🎯 تعداد {len(fronts_gdf)} جبهه صیادی با موفقیت استخراج و رسم شد.")
        except Exception as geojson_ex:
            record_error("خطا در خواندن فایل GeoJSON جبهه‌ها", geojson_ex)

    m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)