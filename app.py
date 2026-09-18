# File Path: app.py
# Description: Streamlit WebGIS application for Ocean PFZ with automatic default shapefile loading, intelligent region-based hyperparameter defaults, and interactive navigation links.

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 

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
from PIL import Image
import folium
from shapely.geometry import LineString
from streamlit_folium import st_folium
import scipy.ndimage as ndimage
from branca.element import MacroElement
from jinja2 import Template

from modules.fetcher import fetch_near_realtime_data
from modules.processor import process_pfz_pipeline

warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

st.set_page_config(page_title="سامانه مدیریت PFZ 🐟", page_icon="🐟", layout="wide")

# ==========================================
# ۱. تنظیمات پیش‌فرض مناطق (گیت‌هاب)
# ==========================================
DEFAULT_ZIP_PATH = "default_shapes.zip"  # مسیر فایل زیپ پیش‌فرض در مخزن گیت‌هاب

def get_region_defaults(reg_name):
    """
    تعیین خودکار مقادیر پیش‌فرض بر اساس نام منطقه
    """
    name_lower = reg_name.lower()
    if "oman" in name_lower:
        return 0.6, 0.5  # (SST Weight, Threshold) برای دریای عمان
    elif "persian" in name_lower or "gulf" in name_lower:
        return 0.7, 0.4  # (SST Weight, Threshold) برای خلیج فارس
    return 0.6, 0.5      # مقادیر fallback پیش‌فرض برای سایر مناطق

# ==========================================
# ۲. کلاس کنترل سفارشی نقشه
# ==========================================
class CustomMapFeatures(MacroElement):
    _template = Template("""
    {% macro script(this, kwargs) %}
    var map = {{ this._parent.get_name() }};
    let useDDM = false;
    let lastLatLng = null;
    let currentMarker = null;

    function toDDM(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const decimalMinutes = ((absolute - degrees) * 60).toFixed(3);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${decimalMinutes}' ${direction}`;
    }

    function updateCoordDisplay(latlng) {
      const displayElement = document.getElementById('coord-text');
      if (!displayElement || !latlng) return;
      if (useDDM) {
        displayElement.innerHTML = `${toDDM(latlng.lat, true)} | ${toDDM(latlng.lng, false)}`;
      } else {
        displayElement.innerHTML = `Lat: ${latlng.lat.toFixed(5)} | Lng: ${latlng.lng.toFixed(5)}`;
      }
    }

    const coordControl = L.control({ position: 'bottomright' });
    coordControl.onAdd = function (map) {
      const div = L.DomUtil.create('div', 'coord-box');
      div.style.padding = '8px 12px';
      div.style.background = 'rgba(255, 255, 255, 0.95)';
      div.style.border = '2px solid #2B5B84';
      div.style.borderRadius = '8px';
      div.style.fontSize = '13px';
      div.style.direction = 'ltr';
      div.style.fontFamily = 'monospace';
      div.style.display = 'flex';
      div.style.alignItems = 'center';
      div.style.gap = '10px';
      div.style.boxShadow = '0 2px 6px rgba(0,0,0,0.3)';
      div.style.zIndex = '1000';

      div.innerHTML = `
        <span id="coord-text" style="font-weight: bold; color: #333;">Lat: -- | Lng: --</span>
        <button id="toggle-coord-btn" style="cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: bold; border: 1px solid #007bff; background: #007bff; color: white; border-radius: 4px;">DDM</button>
      `;

      L.DomEvent.disableClickPropagation(div);
      setTimeout(() => {
        const btnEl = document.getElementById('toggle-coord-btn');
        if (btnEl) {
          btnEl.addEventListener('click', () => {
            useDDM = !useDDM;
            btnEl.innerText = useDDM ? 'DD' : 'DDM';
            if (lastLatLng) updateCoordDisplay(lastLatLng);
          });
        }
      }, 100);
      return div;
    };
    coordControl.addTo(map);

    map.on('mousemove', function (e) {
      lastLatLng = e.latlng;
      updateCoordDisplay(e.latlng);
    });

    map.on('click', function (e) {
      const latlng = e.latlng;
      if (currentMarker) {
        currentMarker.setLatLng(latlng);
      } else {
        currentMarker = L.marker(latlng).addTo(map);
      }
      
      const latDDM = toDDM(latlng.lat, true);
      const lngDDM = toDDM(latlng.lng, false);
      const copyText = latDDM + '  |  ' + lngDDM;
      const latFixed = latlng.lat.toFixed(5);
      const lngFixed = latlng.lng.toFixed(5);

      const navionicsUrl = `https://maps.garmin.com/en-US/marine/#13/${latFixed}/${lngFixed}`;
      const geoUrl = `geo:${latFixed},${lngFixed}?q=${latFixed},${lngFixed}(PFZ+Target)`;
      const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${latFixed},${lngFixed}`;
      const openSeaMapUrl = `https://map.openseamap.org/?zoom=13&lat=${latFixed}&lon=${lngFixed}`;
      const windyUrl = `https://www.windy.com/?${latFixed},${lngFixed},11`;
      
      const popupHtml = `
        <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A; min-width:230px; padding: 2px;">
          <div style="margin-bottom:6px;">${latDDM}<br>${lngDDM}</div>
          <input type="text" id="coord-input-box" value="${copyText}" readonly style="width: 100%; text-align: center; font-family: monospace; font-size: 11px; padding: 4px; margin-bottom: 6px; border: 1px solid #007bff; border-radius: 4px; background: #f0f4f8; color: #333;" />
          <button id="popup-copy-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #007bff; color: white; border-radius: 4px; width: 100%; font-weight:bold; margin-bottom: 5px;">📋 کپی کُد مختصات (Copy)</button>
          <a href="${navionicsUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #002B49; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">⚓ باز کردن در Navionics / Garmin Marine</a>
          <a href="${geoUrl}" style="display: block; padding: 5px 8px; font-size: 11px; background: #28a745; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">📲 ارسال به Garmin App / GPS (موبایل)</a>
          <a href="${gmapsUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #4285F4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">📍 باز کردن در گوگل مپ</a>
          <a href="${openSeaMapUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #007791; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">🌐 باز کردن در OpenSeaMap</a>
          <a href="${windyUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #1B65B4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center;">🌊 باز کردن در Windy</a>
        </div>
      `;
      
      currentMarker.bindPopup(popupHtml).openPopup();
      lastLatLng = latlng;
      updateCoordDisplay(latlng);

      setTimeout(() => {
        const copyBtn = document.getElementById('popup-copy-btn');
        const inputBox = document.getElementById('coord-input-box');
        if (copyBtn && inputBox) {
          L.DomEvent.disableClickPropagation(copyBtn);
          L.DomEvent.disableClickPropagation(inputBox);
          copyBtn.addEventListener('click', function() {
            inputBox.select();
            document.execCommand('copy');
            copyBtn.innerText = 'کپی شد!';
            setTimeout(() => { copyBtn.innerText = '📋 کپی کُد مختصات (Copy)'; }, 2000);
          });
        }
      }, 150);
    });
    {% endmacro %}
    """)
    def __init__(self):
        super().__init__()

# استایل‌دهی RTL
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input { font-family: 'Vazirmatn', sans-serif; }
    .main-title { font-size: 2rem !important; color: #1E3A8A; font-weight: bold; margin-bottom: 1rem; text-align: right !important; }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = (gy + 1) if gm > 2 else gy
    days = 355666 + (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) + gd + g_d_m[gm - 1]
    jy = -1595 + (33 * (days // 12053))
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
    if not date_str: return None, None
    try:
        dt = pd.to_datetime(date_str)
        greg_str = dt.strftime("%Y-%m-%d")
        jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
        return greg_str, f"{jy}/{jm:02d}/{jd:02d}"
    except Exception:
        return str(date_str), None

if "error_logs" not in st.session_state: st.session_state.error_logs = []
if "process_logs" not in st.session_state: st.session_state.process_logs = []
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
for key in ["nc_out_list", "sst_nc_path", "chl_nc_path", "combined_fronts_gdf", "combined_region_gdf", "minx", "miny", "maxx", "maxy", "latest_date"]:
    if key not in st.session_state: st.session_state[key] = None

def record_error(msg, exc=None):
    full_msg = f"{msg}\n{traceback.format_exc()}" if exc else msg
    logging.error(full_msg)
    st.session_state.error_logs.append(full_msg)

def log_process(msg_type, msg_text, status_obj=None):
    st.session_state.process_logs.append((msg_type, msg_text))
    if status_obj: status_obj.write(msg_text)

st.markdown('<div class="main-title">🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ) 🐟</div>', unsafe_allow_html=True)

if st.session_state.error_logs:
    st.error("⚠️ خطاهایی در حین اجرای برنامه رخ داده است:")
    st.code("\n".join(st.session_state.error_logs), language="text")
    if st.button("🗑️ پاک‌کردن تاریخچه خطاها"):
        st.session_state.error_logs = []
        st.rerun()

st.sidebar.header("⚙️ تنظیمات پردازش و مدل")

# آپلود فایل زیپ (اختیاری)
uploaded_shapefile_zip = st.sidebar.file_uploader(
    "آپلود شیپ‌فایل جدید (.zip) - اختیاری", 
    type="zip",
    help="در صورت عدم آپلود، شیپ‌فایل پیش‌فرض پروژه (خلیج فارس و عمان) بارگذاری می‌شود."
)

output_dir = os.path.join(tempfile.gettempdir(), "Data_Processed")
os.makedirs(output_dir, exist_ok=True)

# مدیریت منبع شیپ‌فایل (پیش‌فرض یا آپلودی)
target_zip_file = None
if uploaded_shapefile_zip is not None:
    target_zip_file = uploaded_shapefile_zip
    st.sidebar.success("📂 زیپ‌فایل سفارشی آپلود شد.")
elif os.path.exists(DEFAULT_ZIP_PATH):
    target_zip_file = DEFAULT_ZIP_PATH
    st.sidebar.info("📁 استفاده از شیپ‌فایل پیش‌فرض (خلیج فارس و عمان)")
else:
    st.sidebar.warning("⚠️ هیچ فایل پیش‌فرضی در مخزن یافت نشد. لطفاً فایل زیپ را آپلود کنید.")

region_configs = {}
if target_zip_file is not None:
    extract_path = os.path.join(output_dir, "extracted_shapes")
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(target_zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    shp_files = [os.path.join(r, f) for r, d, files in os.walk(extract_path) for f in files if f.endswith('.shp') and not f.startswith('._')]
    
    if shp_files:
        st.sidebar.subheader("📌 تنظیمات پارامترهای هر منطقه")
        for shp_path in sorted(shp_files):
            reg_name = os.path.splitext(os.path.basename(shp_path))[0].replace("_", " ").title()
            
            # دریافت هوشمند مقادیر پیش‌فرض براساس نام منطقه
            def_sst, def_thresh = get_region_defaults(reg_name)
            
            with st.sidebar.expander(f"منطقه: {reg_name}", expanded=True):
                sst_w = st.slider(f"وزن SST ({reg_name})", 0.0, 1.0, def_sst, 0.05, key=f"sst_{reg_name}")
                chl_w = round(1.0 - sst_w, 2)
                st.caption(f"وزن کلروفیل-آ: **{chl_w}**")
                
                thresh = st.slider(f"آستانه حساسیت ({reg_name})", 0.1, 1.0, def_thresh, 0.05, key=f"thresh_{reg_name}")
                
                region_configs[reg_name] = {
                    "shp_path": shp_path,
                    "sst_weight": sst_w,
                    "chl_weight": chl_w,
                    "threshold": thresh
                }

def generate_fronts_fallback(nc_path, user_threshold, region_name):
    try:
        if not nc_path or not os.path.exists(nc_path): return None
        with xr.open_dataset(nc_path) as ds:
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key].load()
            lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
            lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            if not lat_name or not lon_name: return None
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            lats, lons = da[lat_name].values, da[lon_name].values
            if da.ndim > 2:
                for d in [d for d in da.dims if d not in [lat_name, lon_name]]: da = da.isel({d: 0})
            data = da.values.copy()
            
        valid_mask = ~np.isnan(data) & (data > 0)
        if not valid_mask.any(): return None

        data_filled = np.where(valid_mask, data, float(np.nanmean(data[valid_mask])))
        data_smoothed = ndimage.gaussian_filter(data_filled, sigma=1.0).astype(float)
        eroded_mask = ndimage.binary_erosion(valid_mask, structure=np.ones((3, 3)), iterations=1)
        data_smoothed[~eroded_mask] = np.nan
        
        valid_smoothed = data_smoothed[eroded_mask]
        if len(valid_smoothed) == 0: return None

        smooth_max = float(np.nanmax(valid_smoothed))
        active_threshold = user_threshold if user_threshold < smooth_max else smooth_max * 0.80
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = [LineString(poly) for segs in cs.allsegs for poly in segs if len(poly) > 2]
            plt.close(fig)
            return extracted

        lines = extract_lines(active_threshold)
        if not lines:
            for pct in [0.60, 0.40, 0.20]:
                test_t = smooth_max * pct
                if test_t <= 0: continue
                lines = extract_lines(test_t)
                if lines: break
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Region'], gdf_fronts['Threshold'] = region_name, active_threshold
            gdf_fronts = gdf_fronts.to_crs("EPSG:3857")
            gdf_fronts['Length_km'] = gdf_fronts.geometry.length / 1000
            gdf_fronts = gdf_fronts.to_crs("EPSG:4326")
            return gdf_fronts[gdf_fronts['Length_km'] > 0.5]
    except Exception as ex:
        record_error(f"خطا در استخراج جبهه برای {region_name}", ex)
    return None

def render_pixel_perfect_heatmap(da, label, reg_name, cmap_name, out_dir):
    try:
        lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
        lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        if not lat_name or not lon_name: return None, None
        for d in [d for d in da.dims if d not in [lat_name, lon_name]]: da = da.isel({d: 0})

        da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
        lats, lons, data_arr = da[lat_name].values, da[lon_name].values, da.values.copy().astype(float)
        ny, nx = data_arr.shape
        if ny < 2 or nx < 2: return None, None

        dx = float(np.abs(lons[1] - lons[0])) / 2.0 if len(lons) > 1 else 0.025
        dy = float(np.abs(lats[1] - lats[0])) / 2.0 if len(lats) > 1 else 0.025

        valid_mask = ~np.isnan(data_arr) & (data_arr > 0)
        if not valid_mask.any(): return None, None

        vmin, vmax = float(np.nanmin(data_arr[valid_mask])), float(np.nanmax(data_arr[valid_mask]))
        norm_arr = (data_arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(data_arr)

        rgba_img = plt.get_cmap(cmap_name)(norm_arr)
        rgba_img[~valid_mask] = [0.0, 0.0, 0.0, 0.0]
        img = Image.fromarray((np.flipud(rgba_img) * 255.0).clip(0, 255).astype(np.uint8), 'RGBA')

        file_path = os.path.join(out_dir, f"{label}_{reg_name.replace(' ', '_')}.png")
        img.save(file_path)
        return file_path, [[float(lats[0]) - dy, float(lons[0]) - dx], [float(lats[-1]) + dy, float(lons[-1]) + dx]]
    except Exception as ex:
        record_error(f"خطا در رندر پیکسل برای {label} در {reg_name}", ex)
        return None, None

def load_and_crop_dataset(nc_path, shp_path):
    if not nc_path or not os.path.exists(nc_path): return None
    try:
        with xr.open_dataset(nc_path) as ds: da = ds[list(ds.data_vars.keys())[0]].load()
        gdf = gpd.read_file(shp_path)
        if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
        lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        if lat_name and lon_name:
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            return da.sel({lat_name: slice(miny - 0.05, maxy + 0.05), lon_name: slice(minx - 0.05, maxx + 0.05)})
    except Exception as ex: record_error(f"خطا در برش داده {nc_path}", ex)
    return None

if st.sidebar.button("🚀 دریافت داده‌های به‌روز و اجرای تحلیل"):
    if not region_configs:
        st.error("لطفاً یک شیپ‌فایل معتبر آپلود کنید یا فایل default_shapes.zip را در گیت‌هاب بگذارید.")
    else:
        st.session_state.process_logs = []
        with st.status("🚀 شروع فرآیند پردازش داده‌های مکانی...", expanded=True) as status:
            try:
                log_process("info", "در حال استخراج و خواندن شیپ‌فایل‌های منطقه‌ای...", status)
                all_gdfs = []
                for reg_name, cfg in region_configs.items():
                    temp_gdf = gpd.read_file(cfg["shp_path"])
                    if temp_gdf.crs and temp_gdf.crs != "EPSG:4326": temp_gdf = temp_gdf.to_crs("EPSG:4326")
                    temp_gdf["Region"] = reg_name
                    all_gdfs.append(temp_gdf)

                combined_region_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
                minx, miny, maxx, maxy = combined_region_gdf.total_bounds

                log_process("info", "در حال دریافت داده‌های SST و CHL...", status)
                sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)

                if sst_nc_path and chl_nc_path:
                    greg_d, jalali_d = parse_date_formats(latest_date)
                    log_process("success", f"داده‌های ماهواره‌ای دریافت شد. (تاریخ: {jalali_d} | {greg_d})", status)
                    
                    all_front_gdfs, nc_out_list = [], []
                    for reg_name, cfg in region_configs.items():
                        log_process("info", f"در حال پردازش **{reg_name}** (وزن SST: {cfg['sst_weight']} | آستانه: {cfg['threshold']})...", status)
                        reg_out_dir = os.path.join(output_dir, reg_name.replace(" ", "_"))
                        nc_out, _, _ = process_pfz_pipeline(cfg["shp_path"], sst_nc_path, chl_nc_path, reg_out_dir, cfg["sst_weight"], cfg["chl_weight"])

                        if nc_out:
                            nc_out_list.append((reg_name, nc_out, cfg["shp_path"]))
                            reg_fronts_gdf = generate_fronts_fallback(nc_out, cfg["threshold"], reg_name)
                            if reg_fronts_gdf is not None: all_front_gdfs.append(reg_fronts_gdf)

                    st.session_state.combined_fronts_gdf = pd.concat(all_front_gdfs, ignore_index=True) if all_front_gdfs else None
                    st.session_state.combined_region_gdf = combined_region_gdf
                    st.session_state.nc_out_list = nc_out_list
                    st.session_state.sst_nc_path, st.session_state.chl_nc_path = sst_nc_path, chl_nc_path
                    st.session_state.latest_date = latest_date
                    st.session_state.minx, st.session_state.miny, st.session_state.maxx, st.session_state.maxy = minx, miny, maxx, maxy
                    st.session_state.analysis_done = True
                    status.update(label="تمام مراحل پردازش با موفقیت به پایان رسید!", state="complete")
            except Exception as global_ex:
                record_error("خطای کلی در جریان اجرای برنامه", global_ex)
                status.update(label="اجرای برنامه با خطا متوقف شد", state="error")

if st.session_state.process_logs:
    with st.expander("📝 گزارش مراحل پردازش", expanded=True):
        for msg_type, text in st.session_state.process_logs:
            if msg_type == "success": st.success(text)
            elif msg_type == "error": st.error(text)
            elif msg_type == "warning": st.warning(text)
            else: st.info(text)

# ==========================================
# ۳. رندر نقشه
# ==========================================
if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه و لایه‌های نقشه حرارتی (Heatmap)")
    try:
        greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
        m = folium.Map(location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], zoom_start=6, tiles=None)
        
        folium.TileLayer('OpenStreetMap', name='نقشه خیابانی (OSM)').add_to(m)
        folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='تصاویر ماهواره‌ای گوگل', overlay=False).add_to(m)
        folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', name='نقشه ترکیبی گوگل', overlay=False).add_to(m)
        CustomMapFeatures().add_to(m)

        if st.session_state.nc_out_list:
            for reg_name, nc_out, reg_shp_path in st.session_state.nc_out_list:
                if nc_out and os.path.exists(nc_out):
                    with xr.open_dataset(nc_out) as ds_pfz:
                        da_pfz = ds_pfz[list(ds_pfz.data_vars.keys())[0]].load()
                        img_path, bounds = render_pixel_perfect_heatmap(da_pfz, "PFZ", reg_name, "jet", output_dir)
                        if img_path: folium.raster_layers.ImageOverlay(image=img_path, bounds=bounds, opacity=0.65, name=f"PFZ ({reg_name})", show=True).add_to(m)

                if st.session_state.sst_nc_path:
                    da_sst = load_and_crop_dataset(st.session_state.sst_nc_path, reg_shp_path)
                    if da_sst is not None:
                        img_path_sst, bounds_sst = render_pixel_perfect_heatmap(da_sst, "SST", reg_name, "coolwarm", output_dir)
                        if img_path_sst: folium.raster_layers.ImageOverlay(image=img_path_sst, bounds=bounds_sst, opacity=0.65, name=f"SST ({reg_name})", show=False).add_to(m)

                if st.session_state.chl_nc_path:
                    da_chl = load_and_crop_dataset(st.session_state.chl_nc_path, reg_shp_path)
                    if da_chl is not None:
                        img_path_chl, bounds_chl = render_pixel_perfect_heatmap(da_chl, "Chlorophyll-a", reg_name, "YlGn", output_dir)
                        if img_path_chl: folium.raster_layers.ImageOverlay(image=img_path_chl, bounds=bounds_chl, opacity=0.65, name=f"Chlorophyll-a ({reg_name})", show=False).add_to(m)

        folium.GeoJson(st.session_state.combined_region_gdf, name="Region Boundaries", style_function=lambda x: {'color': '#0000FF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'}).add_to(m)
        
        if st.session_state.combined_fronts_gdf is not None and not st.session_state.combined_fronts_gdf.empty:
            folium.GeoJson(st.session_state.combined_fronts_gdf, name="PFZ Front Lines", style_function=lambda x: {'color': '#FF0000', 'weight': 3.5, 'opacity': 1.0}).add_to(m)

        m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
        folium.LayerControl().add_to(m)
        st_folium(m, width=1100, height=600, returned_objects=[])

    except Exception as map_render_err:
        st.error("⚠️ خطا در پردازش و رندر نقشه:")
        st.exception(map_render_err)