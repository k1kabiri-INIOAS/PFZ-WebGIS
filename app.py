# File Path: app.py
# Description: Streamlit WebGIS application for Multi-Region Ocean PFZ mapping with pixel-perfect PIL heatmaps, RTL layout, accurate Jalali date conversion, multi-basemap support, custom coordinate display, copy features, multi-platform navigation integration (GPS/Garmin, Google Maps, OpenSeaMap, Navionics, Windy), and robust map render error tracking.

import os
# غیرفعال کردن قفل فایل‌های NetCDF/HDF5 برای جلوگیری از خطای Resource temporarily unavailable (Errno 11)
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
# ۱. کلاس ساخت کنترل سفارشی روی نقشه (JavaScript اختصاصی)
# ==========================================
class CustomMapFeatures(MacroElement):
    """
    تزریق کدهای جاوااسکریپت به نقشه جهت:
    - نمایش لحظه‌ای مختصات
    - سوئیچ بین فرمت‌های DD و DDM (درجه و دقیقه اعشاری)
    - کپی تضمینی متن مختصات در پاپ‌آپ
    - دکمه‌های اتصال مستقیم به سرویس‌های ناوبری (Garmin/GPS, Google Maps, OpenSeaMap, Navionics, Windy)
    - ثبت مارکر تعاملی با کلیک روی نقشه
    """
    _template = Template("""
    {% macro script(this, kwargs) %}
    
    var map = {{ this._parent.get_name() }};
    
    let useDDM = false;
    let lastLatLng = null;
    let currentMarker = null;

    // تابع تبدیل فرمت اعشاری (DD) به درجه و دقیقه اعشاری (DDM)
    function toDDM(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const decimalMinutes = ((absolute - degrees) * 60).toFixed(3);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${decimalMinutes}' ${direction}`;
    }

    // تابع بروزرسانی متن باکس مختصات گوشه صفحه
    function updateCoordDisplay(latlng) {
      const displayElement = document.getElementById('coord-text');
      if (!displayElement || !latlng) return;

      if (useDDM) {
        displayElement.innerHTML = `${toDDM(latlng.lat, true)} | ${toDDM(latlng.lng, false)}`;
      } else {
        displayElement.innerHTML = `Lat: ${latlng.lat.toFixed(5)} | Lng: ${latlng.lng.toFixed(5)}`;
      }
    }

    // ساخت کنترل (باکس گوشه پایین سمت راست)
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
        <button id="toggle-coord-btn" title="تغییر فرمت نمایش" style="
          cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: bold;
          border: 1px solid #007bff; background: #007bff; color: white; border-radius: 4px;
        ">DDM</button>
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

    // رویداد حرکت ماوس (آپدیت نمایش مختصات)
    map.on('mousemove', function (e) {
      lastLatLng = e.latlng;
      updateCoordDisplay(e.latlng);
    });

    // رویداد کلیک روی نقشه (ایجاد مارکر و پاپ‌آپ شامل کپی مختصات و لینک‌های چندگانه)
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
      
      // لینک‌های دسترسی به سرویس‌ها و پروتکل‌های مختلف
      const geoUrl = `geo:${latlng.lat},${latlng.lng}?q=${latlng.lat},${latlng.lng}(PFZ+Target)`;
      const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${latlng.lat},${latlng.lng}`;
      const openSeaMapUrl = `https://map.openseamap.org/?zoom=12&lat=${latlng.lat}&lon=${latlng.lng}`;
      const navionicsUrl = `https://webapp.navionics.com/?lat=${latlng.lat}&lon=${latlng.lng}&zoom=12`;
      const windyUrl = `https://www.windy.com/?${latlng.lat},${latlng.lng},11`;
      
      const popupHtml = `
        <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A; min-width:225px; padding: 2px; max-height: 310px; overflow-y: auto;">
          <div style="margin-bottom:6px;">${latDDM}<br>${lngDDM}</div>
          <input type="text" id="coord-input-box" value="${copyText}" readonly style="width: 100%; text-align: center; font-family: monospace; font-size: 11px; padding: 4px; margin-bottom: 6px; border: 1px solid #007bff; border-radius: 4px; background: #f0f4f8; color: #333;" />
          
          <button id="popup-copy-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #007bff; color: white; border-radius: 4px; width: 100%; font-weight:bold; margin-bottom: 5px;">
            انتخاب و کپی مختصات (Copy)
          </button>
          
          <a href="${geoUrl}" style="display: block; padding: 5px 8px; font-size: 11px; background: #28a745; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">
            🎯 باز کردن در GPS / Garmin (App)
          </a>

          <a href="${gmapsUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #4285F4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">
            📍 باز کردن در گوگل مپ
          </a>

          <a href="${openSeaMapUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #007791; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">
            ⚓ باز کردن در OpenSeaMap
          </a>

          <a href="${navionicsUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #002B49; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">
            🗺️ باز کردن در Navionics
          </a>

          <a href="${windyUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #1B65B4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center;">
            🌊 باز کردن در Windy
          </a>
        </div>
      `;
      
      currentMarker.bindPopup(popupHtml).openPopup();
      lastLatLng = latlng;
      updateCoordDisplay(latlng);

      // اتصال رویداد کپی به عناصر داخل پاپ‌آپ
      setTimeout(() => {
        const copyBtn = document.getElementById('popup-copy-btn');
        const inputBox = document.getElementById('coord-input-box');
        
        if (copyBtn && inputBox) {
          L.DomEvent.disableClickPropagation(copyBtn);
          L.DomEvent.disableClickPropagation(inputBox);

          const doCopy = function() {
            inputBox.select();
            inputBox.setSelectionRange(0, 99999);
            
            try {
              var successful = document.execCommand('copy');
              if (successful) {
                copyBtn.innerText = 'کپی شد! (Copied)';
                copyBtn.style.background = '#17a2b8';
                setTimeout(() => {
                  copyBtn.innerText = 'انتخاب و کپی مختصات (Copy)';
                  copyBtn.style.background = '#007bff';
                }, 2000);
              } else {
                throw new Error("ExecCommand failed");
              }
            } catch (err) {
              if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(copyText).then(() => {
                  copyBtn.innerText = 'کپی شد! (Copied)';
                  copyBtn.style.background = '#17a2b8';
                  setTimeout(() => {
                    copyBtn.innerText = 'انتخاب و کپی مختصات (Copy)';
                    copyBtn.style.background = '#007bff';
                  }, 2000);
                }).catch(() => {
                  copyBtn.innerText = 'متن انتخاب شد (Ctrl+C)';
                });
              } else {
                copyBtn.innerText = 'متن انتخاب شد (Ctrl+C)';
              }
            }
          };

          copyBtn.addEventListener('click', doCopy);
          inputBox.addEventListener('click', function() {
            inputBox.select();
          });
        }
      }, 150);
    });

    {% endmacro %}
    """)
    def __init__(self):
        super().__init__()


# تزریق استایل RTL و فونت‌های فارسی
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    .stApp, [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input {
        font-family: 'Vazirmatn', sans-serif;
    }

    .material-symbols-rounded, 
    .material-symbols-outlined, 
    [data-testid="stIconMaterial"], 
    i.material-icons,
    .stIcon,
    svg,
    svg * {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        direction: ltr !important;
    }

    .main-title {
        font-size: 2rem !important;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: right !important;
    }
    
    .stMarkdown, .stSelectbox, .stSlider {
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

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

if "error_logs" not in st.session_state:
    st.session_state.error_logs = []
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
for key in ["nc_out_list", "sst_nc_path", "chl_nc_path", "combined_fronts_gdf", "combined_region_gdf", "minx", "miny", "maxx", "maxy", "latest_date"]:
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

st.markdown('<div class="main-title">🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ) 🐟</div>', unsafe_allow_html=True)

if st.session_state.error_logs:
    st.error("⚠️ خطاهایی در حین اجرای برنامه رخ داده است:")
    all_logs_str = "\n".join(st.session_state.error_logs)
    st.code(all_logs_str, language="text")
    if st.button("🗑️ پاک‌کردن تاریخچه خطاها"):
        st.session_state.error_logs = []
        st.rerun()

st.sidebar.header("⚙️ تنظیمات پردازش و مدل")

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
        st.sidebar.subheader("📌 تنظیمات اختصاصی هر منطقه")
        for shp_path in sorted(shp_files):
            reg_name = os.path.splitext(os.path.basename(shp_path))[0].replace("_", " ").title()
            
            with st.sidebar.expander(f"منطقه: {reg_name}", expanded=True):
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
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key].load()
            
            lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
            lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            
            if not lat_name or not lon_name:
                record_error("ابعاد مکانی (lat/lon) به درستی در فایل NetCDF یافت نشد.")
                return None
                
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            lats = da[lat_name].values
            lons = da[lon_name].values
            
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

def render_pixel_perfect_heatmap(da, label, reg_name, cmap_name, out_dir):
    try:
        lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
        lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        if not lat_name or not lon_name:
            return None, None

        if extra_dims := [d for d in da.dims if d not in [lat_name, lon_name]]:
            for d in extra_dims:
                da = da.isel({d: 0})

        da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)

        lats = da[lat_name].values
        lons = da[lon_name].values
        data_arr = da.values.copy().astype(float)

        ny, nx = data_arr.shape
        if ny < 2 or nx < 2:
            return None, None

        dx = float(np.abs(lons[1] - lons[0])) / 2.0 if len(lons) > 1 else 0.025
        dy = float(np.abs(lats[1] - lats[0])) / 2.0 if len(lats) > 1 else 0.025

        grid_minx = float(lons[0]) - dx
        grid_maxx = float(lons[-1]) + dx
        grid_miny = float(lats[0]) - dy
        grid_maxy = float(lats[-1]) + dy

        valid_mask = ~np.isnan(data_arr) & (data_arr > 0)
        if not valid_mask.any():
            return None, None

        vmin, vmax = float(np.nanmin(data_arr[valid_mask])), float(np.nanmax(data_arr[valid_mask]))
        norm_arr = np.zeros_like(data_arr)
        if vmax > vmin:
            norm_arr = (data_arr - vmin) / (vmax - vmin)

        colormap = plt.get_cmap(cmap_name)
        rgba_img = colormap(norm_arr)
        rgba_img[~valid_mask] = [0.0, 0.0, 0.0, 0.0]

        rgba_img = np.flipud(rgba_img)

        img_uint8 = (rgba_img * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'RGBA')

        file_path = os.path.join(out_dir, f"{label}_{reg_name.replace(' ', '_')}.png")
        img.save(file_path)

        bounds = [[grid_miny, grid_minx], [grid_maxy, grid_maxx]]
        return file_path, bounds
    except Exception as ex:
        record_error(f"خطا در رندر پیکسل برای {label} در {reg_name}", ex)
        return None, None

def load_and_crop_dataset(nc_path, shp_path):
    if not nc_path or not os.path.exists(nc_path):
        return None
    try:
        with xr.open_dataset(nc_path) as ds:
            var_key = list(ds.data_vars.keys())[0]
            da = ds[var_key].load()
        
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        minx, miny, maxx, maxy = gdf.total_bounds
        lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
        lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        
        if lat_name and lon_name:
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            da_cropped = da.sel({lat_name: slice(miny - 0.05, maxy + 0.05), lon_name: slice(minx - 0.05, maxx + 0.05)})
            return da_cropped
    except Exception as ex:
        record_error(f"خطا در برش داده {nc_path}", ex)
    return None

if st.sidebar.button("🚀 دریافت داده‌های به‌روز و اجرای تحلیل"):
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

                log_process("info", "در حال برقراری ارتباط با سرور و دریافت داده‌های SST و CHL...", status)
                try:
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                except Exception as fetch_ex:
                    record_error("خطا در ماژول fetch_near_realtime_data", fetch_ex)
                    sst_nc_path, chl_nc_path, latest_date = None, None, None

                if sst_nc_path and chl_nc_path:
                    greg_d, jalali_d = parse_date_formats(latest_date)
                    log_process("success", f"داده‌های ماهواره‌ای با موفقیت دریافت شدند. (تاریخ اخذ داده: {jalali_d} | Data Acquisition Date: {greg_d})", status)
                    
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
                    st.session_state.sst_nc_path = sst_nc_path
                    st.session_state.chl_nc_path = chl_nc_path
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

# ==========================================
# ۲. بخش نمایش نقشه تعاملی با مدیریت خطا
# ==========================================
if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه و لایه‌های نقشه حرارتی (Heatmap)")
    
    try:
        greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
        
        if greg_str and jalali_str:
            persian_digits = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
            jalali_str_fa = jalali_str.translate(persian_digits)
            st.info(f"📅 **تاریخ اخذ داده:** `{jalali_str_fa}` | **Data Acquisition Date:** `{greg_str}`")

        # ساخت نقشه پایه بدون کاشی پیش‌فرض
        m = folium.Map(
            location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], 
            zoom_start=6, 
            tiles=None
        )
        
        # 🌐 افزودن سرویس‌های نقشه پس‌زمینه (Basemaps)
        folium.TileLayer('OpenStreetMap', name='نقشه خیابانی (OSM)').add_to(m)
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='تصاویر ماهواره‌ای گوگل (Satellite)',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google Hybrid',
            name='نقشه ترکیبی گوگل (Hybrid)',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Esri Topo',
            name='توپوگرافی (Esri Topo)',
            overlay=False,
            control=True
        ).add_to(m)

        # 📍 تزریق کنترل سفارشی مختصات، کپی، و دکمه‌های ناوبری
        CustomMapFeatures().add_to(m)
        
        # ۱. بارگذاری و نمایش مجزای لایه‌های PFZ, SST, Chlorophyll-a
        if st.session_state.nc_out_list:
            for reg_name, nc_out, reg_shp_path in st.session_state.nc_out_list:
                
                # (الف) لایه PFZ
                if nc_out and os.path.exists(nc_out):
                    try:
                        with xr.open_dataset(nc_out) as ds_pfz:
                            var_key = "pfz_index" if "pfz_index" in ds_pfz else list(ds_pfz.data_vars.keys())[0]
                            da_pfz = ds_pfz[var_key].load()
                            
                            img_path, bounds = render_pixel_perfect_heatmap(da_pfz, "PFZ", reg_name, "jet", output_dir)
                            if img_path and bounds:
                                folium.raster_layers.ImageOverlay(
                                    image=img_path,
                                    bounds=bounds,
                                    opacity=0.65,
                                    name=f"PFZ ({reg_name})",
                                    show=True
                                ).add_to(m)
                    except Exception as pfz_ex:
                        record_error(f"خطا در ایجاد لایه PFZ منطقه {reg_name}", pfz_ex)

                # (ب) لایه SST
                if st.session_state.sst_nc_path:
                    try:
                        da_sst = load_and_crop_dataset(st.session_state.sst_nc_path, reg_shp_path)
                        if da_sst is not None:
                            img_path_sst, bounds_sst = render_pixel_perfect_heatmap(da_sst, "SST", reg_name, "coolwarm", output_dir)
                            if img_path_sst and bounds_sst:
                                folium.raster_layers.ImageOverlay(
                                    image=img_path_sst,
                                    bounds=bounds_sst,
                                    opacity=0.65,
                                    name=f"SST ({reg_name})",
                                    show=False
                                ).add_to(m)
                    except Exception as sst_ex:
                        record_error(f"خطا در ایجاد لایه SST منطقه {reg_name}", sst_ex)

                # (ج) لایه Chlorophyll-a
                if st.session_state.chl_nc_path:
                    try:
                        da_chl = load_and_crop_dataset(st.session_state.chl_nc_path, reg_shp_path)
                        if da_chl is not None:
                            img_path_chl, bounds_chl = render_pixel_perfect_heatmap(da_chl, "Chlorophyll-a", reg_name, "YlGn", output_dir)
                            if img_path_chl and bounds_chl:
                                folium.raster_layers.ImageOverlay(
                                    image=img_path_chl,
                                    bounds=bounds_chl,
                                    opacity=0.65,
                                    name=f"Chlorophyll-a ({reg_name})",
                                    show=False
                                ).add_to(m)
                    except Exception as chl_ex:
                        record_error(f"خطا در ایجاد لایه Chlorophyll-a منطقه {reg_name}", chl_ex)

        # ۲. رسم مرز مناطق
        folium.GeoJson(
            st.session_state.combined_region_gdf,
            name="Region Boundaries",
            style_function=lambda x: {'color': '#0000FF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'}
        ).add_to(m)
        
        # ۳. رسم خطوط جبهه‌ها
        if st.session_state.combined_fronts_gdf is not None and not st.session_state.combined_fronts_gdf.empty:
            folium.GeoJson(
                st.session_state.combined_fronts_gdf,
                name="PFZ Front Lines",
                style_function=lambda x: {'color': '#FF0000', 'weight': 3.5, 'opacity': 1.0}
            ).add_to(m)
            st.success(f"🎯 تعداد {len(st.session_state.combined_fronts_gdf)} جبهه صیادی در مجموع مناطق استخراج و رسم شد.")

        # ۴. کادر شناور روی نقشه با تاریخ شمسی و میلادی
        if greg_str and jalali_str:
            persian_digits = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
            jalali_str_fa = jalali_str.translate(persian_digits)
            
            date_box_html = f'''
                <div style="position: fixed; 
                            bottom: 25px; left: 20px; width: 250px; height: 50px; 
                            z-index:9999; font-size:12px; background-color: rgba(255, 255, 255, 0.92); 
                            border: 2px solid #2B5B84; border-radius: 6px; 
                            padding: 4px; font-weight: bold; text-align: center; color: #1E3A8A; line-height: 1.4;
                            direction: rtl; font-family: 'Vazirmatn', sans-serif;">
                    تاریخ اخذ داده: {jalali_str_fa}<br>
                    <span style="font-family: Arial, sans-serif; color: #333333; font-size: 11px;">Data Acquisition Date: {greg_str}</span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(date_box_html))

        m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
        folium.LayerControl().add_to(m)
        
        # نمایش نهایی نقشه
        st_folium(m, width=1100, height=600, returned_objects=[])

    except Exception as map_render_err:
        st.error("⚠️ خطا در پردازش و رندر نقشه:")
        st.exception(map_render_err)