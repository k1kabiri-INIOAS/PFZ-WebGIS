# File Path: app_2.py
# Description: Streamlit WebGIS application for Multi-Region Ocean PFZ mapping with pixel-perfect PIL heatmaps, RTL layout, accurate Jalali date conversion, multi-basemap support, custom coordinate display with copy/DMS/DDM features, and interactive map marker placement.

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
    - سوئیچ بین سه فرمت DD و DMS و DDM
    - کپی در Clipboard و نمایش پیام Copied!
    - ثبت مارکر تعاملی با کلیک روی نقشه به همراه دکمه کپی
    """
    _template = Template("""
    {% macro script(this, kwargs) %}
    
    var map = {{ this._parent.get_name() }};
    
    let coordFormat = 0; // 0: DD, 1: DMS, 2: DDM
    let lastLatLng = null;
    let currentMarker = null;

    // تابع تبدیل فرمت اعشاری (DD) به درجه-دقیقه-ثانیه (DMS)
    function toDMS(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const minutesNotTruncated = (absolute - degrees) * 60;
      const minutes = Math.floor(minutesNotTruncated);
      const seconds = ((minutesNotTruncated - minutes) * 60).toFixed(1);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${minutes}' ${seconds}" ${direction}`;
    }

    // تابع تبدیل فرمت اعشاری به درجه-دقیقه اعشاری (DDM) - ویژه قایق‌رانان
    function toDDM(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const minutes = ((absolute - degrees) * 60).toFixed(3);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${minutes}' ${direction}`;
    }

    // تابع فرمت‌دهی مختصات بر اساس فرمت انتخابی
    function formatCoord(latlng, formatType) {
      if (formatType === 1) {
        return `${toDMS(latlng.lat, true)} | ${toDMS(latlng.lng, false)}`;
      } else if (formatType === 2) {
        return `${toDDM(latlng.lat, true)} | ${toDDM(latlng.lng, false)}`;
      } else {
        return `Lat: ${latlng.lat.toFixed(5)} | Lng: ${latlng.lng.toFixed(5)}`;
      }
    }

    // تابع بروزرسانی متن باکس مختصات
    function updateCoordDisplay(latlng) {
      const displayElement = document.getElementById('coord-text');
      if (!displayElement || !latlng) return;
      displayElement.innerHTML = formatCoord(latlng, coordFormat);
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
        <span id="copy-toast" style="display: none; color: #28a745; font-weight: bold; font-size: 11px;">Copied!</span>
        <span id="coord-text" title="برای کپی کلیک کنید" style="cursor: pointer; user-select: none; font-weight: bold; color: #333;">Lat: -- | Lng: --</span>
        <button id="toggle-coord-btn" title="تغییر فرمت نمایش" style="
          cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: bold;
          border: 1px solid #007bff; background: #007bff; color: white; border-radius: 4px; min-width: 45px;
        ">DD</button>
      `;

      L.DomEvent.disableClickPropagation(div);

      setTimeout(() => {
        const textEl = document.getElementById('coord-text');
        const btnEl = document.getElementById('toggle-coord-btn');
        const toastEl = document.getElementById('copy-toast');

        if (textEl) {
          textEl.addEventListener('click', () => {
            const text = textEl.innerText;
            if (!text || text.includes('--')) return;
            navigator.clipboard.writeText(text).then(() => {
              if (toastEl) {
                toastEl.style.display = 'inline';
                setTimeout(() => { toastEl.style.display = 'none'; }, 1500);
              }
            });
          });
        }

        if (btnEl) {
          btnEl.addEventListener('click', () => {
            coordFormat = (coordFormat + 1) % 3;
            const labels = ['DD', 'DMS', 'DDM'];
            btnEl.innerText = labels[coordFormat];
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

    // رویداد کلیک روی نقشه (ایجاد/جابه‌جایی مارکر، مختصات DDM و دکمه کپی)
    map.on('click', function (e) {
      const latlng = e.latlng;
      if (currentMarker) {
        currentMarker.setLatLng(latlng);
      } else {
        currentMarker = L.marker(latlng).addTo(map);
      }
      
      const ddmLat = toDDM(latlng.lat, true);
      const ddmLng = toDDM(latlng.lng, false);
      const popupContent = `
        <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A; min-width:140px;">
          <div style="margin-bottom: 8px;">
            ${ddmLat}<br>${ddmLng}
          </div>
          <button onclick="navigator.clipboard.writeText('${ddmLat}, ${ddmLng}'); this.innerText='Copied!'; setTimeout(() => this.innerText='Copy', 1500);" 
            style="cursor: pointer; padding: 4px 8px; font-size: 11px; font-weight: bold; border: 1px solid #28a745; background: #28a745; color: white; border-radius: 4px; width:100%;">
            Copy
          </button>
        </div>
      `;
      currentMarker.bindPopup(popupContent).openPopup();
      lastLatLng = latlng;
      updateCoordDisplay(latlng);
    });

    {% endmacro %}
    """)
    def __init__(self):
        super().__init__()


# تزریق استایل RTL و فونت‌های فارسی با محافظت از آیکون‌های Material Streamlit
st.markdown("""
    <style>
    /* ایمپورت فونت‌های فارسی و آیکون‌های متریال */
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    /* تنظیم راست‌چین شدن و فونت پایه برای بدنه */
    .stApp, [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    /* اعمال فونت فارسی فقط به عناصر متنی مشخص تا آیکون‌ها در امان بمانند */
    p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input {
        font-family: 'Vazirmatn', sans-serif;
    }

    /* 🔴 محافظت قطعی از کلاس‌ها و تگ‌های سازنده آیکون در استریم‌لیت */
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

    /* عنوان اصلی برنامه */
    .main-title {
        font-size: 2rem !important;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: right !important;
    }
    
    /* تراز کردن متن داخل سلکتورها و دراپ‌داون‌ها */
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

def to_persian_numbers(text):
    if not text:
        return text
    mapping = {
        '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴', 
        '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
    }
    return "".join(mapping.get(c, c) for c in str(text))

def parse_date_formats(date_str):
    if not date_str:
        return None, None
    try:
        dt = pd.to_datetime(date_str)
        greg_str = dt.strftime("%Y-%m-%d")
        jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
        jalali_str = to_persian_numbers(f"{jy}/{jm:02d}/{jd:02d}")
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

# عنوان اصلی برنامه همراه با آیکون موج و ماهی
st.markdown('<div class="main-title">🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ) 🐟</div>', unsafe_allow_html=True)

if st.session_state.error_logs:
    st.error("⚠️ خطاهایی در حین اجرای برنامه رخ داده است:")
    all_logs_str = "\n".join(st.session_state.error_logs)
    st.code(all_logs_str, language="textتغییرات خواسته‌شده بر روی نسخه `app_2.py` با موفقیت اعمال شد. موارد زیر برای رسیدن به اهداف شما پیاده‌سازی شده‌اند:

*   **حل مشکل کپی شدن مختصات:** محیط‌های اجرای Streamlit (به‌دلیل استفاده از Iframe) در بسیاری از مرورگرها دسترسی مستقیم به `navigator.clipboard` را مسدود می‌کنند[cite: 1]. به همین دلیل یک تابع سراسری قدرتمندتر با سیستم جایگزین (Fallback) مبتنی بر `document.execCommand` اضافه شد تا مختصات تحت هر شرایطی به درستی کپی شود.
*   **اضافه شدن کپی در مارکر کلیک:** با کلیک روی هر نقطه از نقشه، پاپ‌آپ بازشده اکنون دارای دکمه «کپی مختصات» است که از همان سیستم قدرتمند کپی استفاده می‌کند.
*   **پشتیبانی از فرمت DDM:** منطق جاوااسکریپت برای پشتیبانی از ۳ فرمت مختلف (DD و DMS و DDM) بازنویسی شد و دکمه گوشه نقشه به صورت چرخشی بین این سه فرمت جابه‌جا می‌شود[cite: 1].
*   **بومی‌سازی فونت و اعداد تقویم:** برای تبدیل تاریخ به اعداد فارسی، تابع کمکی `to_persian_numerals` نوشته شد[cite: 1]. کادر تاریخ نیز برای اعمال قطعی فونت «وزیرمتن» به‌روزرسانی گردید.

کد نهایی فایل `app_2.py` در زیر قرار دارد:

```python
# File Path: app_2.py
# Description: Streamlit WebGIS application for Multi-Region Ocean PFZ mapping with pixel-perfect PIL heatmaps, RTL layout, accurate Jalali date conversion, multi-basemap support, custom coordinate display with copy/DMS/DDM features, and interactive map marker placement.

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
    - سوئیچ بین فرمت‌های DD, DMS, و DDM
    - کپی در Clipboard با پشتیبانی کامل در Iframe و نمایش پیام Copied!
    - ثبت مارکر تعاملی با کلیک روی نقشه با امکان کپی
    """
    _template = Template("""
    {% macro script(this, kwargs) %}
    
    var map = {{ this._parent.get_name() }};
    
    let coordFormat = 0; // 0: DD, 1: DMS, 2: DDM
    let lastLatLng = null;
    let currentMarker = null;

    // تابع سراسری برای کپی کردن متن
    window.copyToClipboard = function(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(showToast).catch(() => fallbackCopy(text));
        } else {
            fallbackCopy(text);
        }
    };

    function fallbackCopy(text) {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand('copy');
            showToast();
        } catch (err) {
            console.error('Fallback copy failed', err);
        }
        document.body.removeChild(textArea);
    }

    function showToast() {
        const toastEl = document.getElementById('copy-toast');
        if (toastEl) {
            toastEl.style.display = 'inline';
            setTimeout(() => { toastEl.style.display = 'none'; }, 1500);
        }
    }

    // تبدیل فرمت اعشاری (DD) به درجه-دقیقه-ثانیه (DMS)
    function toDMS(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const minutesNotTruncated = (absolute - degrees) * 60;
      const minutes = Math.floor(minutesNotTruncated);
      const seconds = ((minutesNotTruncated - minutes) * 60).toFixed(1);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${minutes}' ${seconds}" ${direction}`;
    }

    // تبدیل فرمت اعشاری (DD) به درجه اعشار-دقیقه (DDM)
    function toDDM(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const minutes = ((absolute - degrees) * 60).toFixed(3);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${minutes}' ${direction}`;
    }

    // تابع فرمت‌دهی خروجی بر اساس وضعیت انتخاب‌شده
    function formatCoord(latlng) {
      if (coordFormat === 1) {
          return `${toDMS(latlng.lat, true)} | ${toDMS(latlng.lng, false)}`;
      } else if (coordFormat === 2) {
          return `${toDDM(latlng.lat, true)} | ${toDDM(latlng.lng, false)}`;
      } else {
          return `Lat: ${latlng.lat.toFixed(5)} | Lng: ${latlng.lng.toFixed(5)}`;
      }
    }

    // تابع بروزرسانی متن باکس مختصات
    function updateCoordDisplay(latlng) {
      const displayElement = document.getElementById('coord-text');
      if (!displayElement || !latlng) return;
      displayElement.innerHTML = formatCoord(latlng);
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
        <span id="copy-toast" style="display: none; color: #28a745; font-weight: bold; font-size: 11px;">Copied!</span>
        <span id="coord-text" title="برای کپی کلیک کنید" style="cursor: pointer; user-select: none; font-weight: bold; color: #333;">Lat: -- | Lng: --</span>
        <button id="toggle-coord-btn" title="تغییر فرمت نمایش" style="
          cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: bold;
          border: 1px solid #007bff; background: #007bff; color: white; border-radius: 4px;
        ">DD</button>
      `;

      L.DomEvent.disableClickPropagation(div);

      setTimeout(() => {
        const textEl = document.getElementById('coord-text');
        const btnEl = document.getElementById('toggle-coord-btn');

        if (textEl) {
          textEl.addEventListener('click', () => {
            const text = textEl.innerText;
            if (!text || text.includes('--')) return;
            window.copyToClipboard(text);
          });
        }

        if (btnEl) {
          btnEl.addEventListener('click', () => {
            coordFormat = (coordFormat + 1) % 3;
            const labels = ['DD', 'DMS', 'DDM'];
            btnEl.innerText = labels[coordFormat];
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

    // رویداد کلیک روی نقشه (ایجاد/جابه‌جایی مارکر و نمایش پاپ‌آپ مختصات)
    map.on('click', function (e) {
      const latlng = e.latlng;
      if (currentMarker) {
        currentMarker.setLatLng(latlng);
      } else {
        currentMarker = L.marker(latlng).addTo(map);
      }
      
      const latDD = latlng.lat.toFixed(5);
      const lngDD = latlng.lng.toFixed(5);
      const copyText = latDD + ', ' + lngDD;
      
      const popupHtml = `
          <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A;">
              Lat: ${latDD}<br>Lng: ${lngDD}<br>
              <button onclick="window.copyToClipboard('${copyText}')" style="margin-top:8px; padding:4px 8px; font-family: 'Vazirmatn', sans-serif; font-size: 11px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                  کپی مختصات
              </button>
          </div>
      `;
      
      currentMarker.bindPopup(popupHtml).openPopup();
      lastLatLng = latlng;
      updateCoordDisplay(latlng);
    });

    {% endmacro %}
    """)
    def __init__(self):
        super().__init__()


# تزریق استایل RTL و فونت‌های فارسی با محافظت از آیکون‌های Material Streamlit
st.markdown("""
    <style>
    /* ایمپورت فونت‌های فارسی و آیکون‌های متریال */
    @import url('[https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css](https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css)');
    @import url('[https://fonts.googleapis.com/css2?family=Material](https://fonts.googleapis.com/css2?family=Material)