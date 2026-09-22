# File Path: app.py
# Description: Streamlit WebGIS application with fine-tuned visual offset (0.005 deg) for alignment.

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 

import sys
import tempfile
import zipfile
import logging
import warnings
import base64
import json
import sqlite3
import hashlib
import datetime
import pandas as pd
import streamlit as st
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import folium
from folium.plugins import Fullscreen
from shapely.geometry import LineString, Point
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
# تنظیمات میزان شیفت دیداری نقشه (بر حسب درجه جغرافیایی)
# ==========================================
OFFSET_LON_DEG = 0.05  # شیفت جزئی طولی
OFFSET_LAT_DEG = 0.1  # شیفت جزئی عرضی

# ==========================================
# ۰. سیستم پایگاه داده، لاگ کاربران و احراز هویت
# ==========================================
DB_PATH = "users.db"
output_dir = os.path.join(tempfile.gettempdir(), "Data_Processed")
os.makedirs(output_dir, exist_ok=True)

STATE_FILE = os.path.join(output_dir, "app_state.json")
FRONTS_FILE = os.path.join(output_dir, "latest_fronts.geojson")
REGIONS_FILE = os.path.join(output_dir, "latest_regions.geojson")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, 
        password TEXT, 
        role TEXT, 
        oauth_provider TEXT,
        first_name TEXT,
        last_name TEXT,
        phone TEXT,
        organization TEXT
    )''')
    
    for col, col_type in [('first_name', 'TEXT'), ('last_name', 'TEXT'), ('phone', 'TEXT'), ('organization', 'TEXT')]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    c.execute('''CREATE TABLE IF NOT EXISTS user_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, action TEXT, timestamp TEXT)''')
    
    c.execute("SELECT * FROM users WHERE username='k1_kabiri'")
    if not c.fetchone():
        hashed_pw = hashlib.sha256('Keivan@010976'.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, role, oauth_provider, first_name, last_name) VALUES (?, ?, ?, ?, ?, ?)", 
                  ('k1_kabiri', hashed_pw, 'admin', 'local', 'کیسان', 'کبیری'))
    
    conn.commit()
    conn.close()

def log_user_activity(username, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)", (username, action, dt_str))
    conn.commit()
    conn.close()

def create_user(username, password, role='user', provider='local', first_name="", last_name="", phone="", organization=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest() if password else ""
        c.execute("""INSERT INTO users (username, password, role, oauth_provider, first_name, last_name, phone, organization) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                  (username, hashed_pw, role, provider, first_name, last_name, phone, organization))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        c.execute("""UPDATE users SET first_name=?, last_name=?, phone=?, organization=? WHERE username=?""", 
                  (first_name, last_name, phone, organization, username))
        conn.commit()
        return True
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hashlib.sha256(password.encode()).hexdigest():
        return user[1]
    return None

def login_or_register_email_user(email, first_name, last_name, phone, organization):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=?", (email,))
    user = c.fetchone()
    conn.close()
    if user:
        create_user(email, "", role=user[0], provider='email', first_name=first_name, last_name=last_name, phone=phone, organization=organization)
        return user[0]
    else:
        create_user(email, "", role='user', provider='email', first_name=first_name, last_name=last_name, phone=phone, organization=organization)
        return 'user'

init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# ==========================================
# استایل‌دهی کلی
# ==========================================
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input { font-family: 'Vazirmatn', sans-serif; }
    .main-title { font-size: 2.0rem !important; color: #1E3A8A; font-weight: bold; margin-bottom: 1rem; text-align: right !important; }
    .stMarkdown, .stSelectbox, .stSlider { text-align: right; }
    </style>
""", unsafe_allow_html=True)

if st.session_state.logged_in and st.session_state.role != 'admin':
    st.markdown("""
        <style>
        .stDeployButton {display: none !important;}
        [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
        header [data-testid="baseButton-header"] {display: none !important;}
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# ذخیره و بازیابی وضعیت برنامه
# ==========================================
def save_shared_state():
    if st.session_state.combined_fronts_gdf is not None:
        st.session_state.combined_fronts_gdf.to_file(FRONTS_FILE, driver="GeoJSON")
    if st.session_state.combined_region_gdf is not None:
        st.session_state.combined_region_gdf.to_file(REGIONS_FILE, driver="GeoJSON")
    
    with open(STATE_FILE, "w") as f:
        json.dump({
            "latest_date": st.session_state.latest_date,
            "minx": st.session_state.minx,
            "miny": st.session_state.miny,
            "maxx": st.session_state.maxx,
            "maxy": st.session_state.maxy,
            "nc_out_list": st.session_state.nc_out_list,
            "sst_nc_path": st.session_state.sst_nc_path,
            "chl_nc_path": st.session_state.chl_nc_path
        }, f)

def load_shared_state():
    if os.path.exists(STATE_FILE) and os.path.exists(REGIONS_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            st.session_state.latest_date = state.get("latest_date")
            st.session_state.minx = state.get("minx")
            st.session_state.miny = state.get("miny")
            st.session_state.maxx = state.get("maxx")
            st.session_state.maxy = state.get("maxy")
            st.session_state.nc_out_list = state.get("nc_out_list")
            st.session_state.sst_nc_path = state.get("sst_nc_path")
            st.session_state.chl_nc_path = state.get("chl_nc_path")
            
            st.session_state.combined_region_gdf = gpd.read_file(REGIONS_FILE)
            if os.path.exists(FRONTS_FILE):
                st.session_state.combined_fronts_gdf = gpd.read_file(FRONTS_FILE)
            else:
                st.session_state.combined_fronts_gdf = None
                
            st.session_state.analysis_done = True
        except Exception as e:
            record_error("خطا در بارگذاری آخرین وضعیت نقشه", e)

def image_to_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

# ==========================================
# کلاس کنترل سفارشی نقشه (مختصات DDM)
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

      const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${latFixed},${lngFixed}`;
      const openSeaMapUrl = `https://map.openseamap.org/?zoom=13&lat=${latFixed}&lon=${lngFixed}`;
      const windyUrl = `https://www.windy.com/?${latFixed},${lngFixed},11`;
      
      const popupHtml = `
        <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A; min-width:230px; padding: 2px;">
          <div style="margin-bottom:6px;">${latDDM}<br>${lngDDM}</div>
          <input type="text" id="coord-input-box" value="${copyText}" readonly style="width: 100%; text-align: center; font-family: monospace; font-size: 11px; padding: 4px; margin-bottom: 6px; border: 1px solid #007bff; border-radius: 4px; background: #f0f4f8; color: #333;" />
          <button id="popup-copy-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #007bff; color: white; border-radius: 4px; width: 100%; font-weight:bold; margin-bottom: 5px;">📋 کپی کُد مختصات (Copy)</button>
          <button id="popup-share-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #6c757d; color: white; border-radius: 4px; font-weight: bold; text-align: center; width: 100%; margin-bottom: 5px;">🔀 اشتراک‌گذاری مختصات (Share With)</button>
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
        const shareBtn = document.getElementById('popup-share-btn');
        const inputBox = document.getElementById('coord-input-box');
        if (copyBtn && inputBox) {
          L.DomEvent.disableClickPropagation(copyBtn);
          L.DomEvent.disableClickPropagation(inputBox);
          copyBtn.addEventListener('click', function() {
            inputBox.select();
            try {
              document.execCommand('copy');
              copyBtn.innerText = 'کپی شد! (Copied)';
              copyBtn.style.background = '#17a2b8';
              setTimeout(() => { copyBtn.innerText = '📋 کپی کُد مختصات (Copy)'; copyBtn.style.background = '#007bff'; }, 2000);
            } catch (err) {
              copyBtn.innerText = 'متن انتخاب شد (Ctrl+C)';
            }
          });
        }
        if (shareBtn) {
          L.DomEvent.disableClickPropagation(shareBtn);
          shareBtn.addEventListener('click', () => {
            if (navigator.share) {
              navigator.share({ title: 'مختصات نقطه صیادی PFZ', text: copyText, url: gmapsUrl }).catch(() => {});
            } else {
              window.open(gmapsUrl, '_blank');
            }
          });
        }
      }, 150);
    });
    {% endmacro %}
    """)
    def __init__(self):
        super().__init__()

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

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
        jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
        return dt.strftime("%Y-%m-%d"), f"{jy}/{jm:02d}/{jd:02d}"
    except Exception:
        return str(date_str), None

if "error_logs" not in st.session_state: st.session_state.error_logs = []
if "process_logs" not in st.session_state: st.session_state.process_logs = []
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
for key in ["nc_out_list", "sst_nc_path", "chl_nc_path", "combined_fronts_gdf", "combined_region_gdf", "minx", "miny", "maxx", "maxy", "latest_date"]:
    if key not in st.session_state: st.session_state[key] = None

def record_error(msg, exc=None):
    full_msg = f"{msg}"
    logging.warning(f"[PFZ-LOG] {full_msg}")
    st.session_state.error_logs.append(full_msg)

def log_process(msg_type, msg_text, status_obj=None):
    st.session_state.process_logs.append((msg_type, msg_text))
    if status_obj: status_obj.write(msg_text)

# ==========================================
# صفحه ورود کاربران
# ==========================================
if not st.session_state.logged_in:
    st.markdown('<div class="main-title">🌊 ورود به سامانه هوشمند مناطق مستعد صید (PFZ)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔐 ورود به سیستم", "📧 ورود با ایمیل و مشخصات"])
        
        with tab1:
            with st.form("login_form"):
                login_user = st.text_input("👤 نام کاربری")
                login_pass = st.text_input("🔑 رمز عبور", type="password")
                submit_login = st.form_submit_button("ورود به سامانه", use_container_width=True)
                
                if submit_login:
                    role = authenticate_user(login_user, login_pass)
                    if role:
                        st.session_state.logged_in = True
                        st.session_state.username = login_user
                        st.session_state.role = role
                        log_user_activity(login_user, "LOGIN")
                        st.success(f"خوش آمدید {login_user}!")
                        st.rerun()
                    else:
                        st.error("نام کاربری یا رمز عبور اشتباه است.")

        with tab2:
            st.markdown("### ورود سریع با ایمیل و اطلاعات شخصی")
            with st.form("email_login_form"):
                email_input = st.text_input("📧 آدرس ایمیل (Email Address)")
                first_name_input = st.text_input("👤 نام *")
                last_name_input = st.text_input("👤 نام خانوادگی *")
                phone_input = st.text_input("📞 شماره تلفن (اختیاری)")
                org_input = st.text_input("🏢 نام سازمان (اختیاری)")
                submit_email_login = st.form_submit_button("تایید و ورود به سامانه", use_container_width=True)
                
                if submit_email_login:
                    if not email_input or "@" not in email_input:
                        st.error("لطفا یک آدرس ایمیل معتبر وارد کنید.")
                    elif not first_name_input.strip() or not last_name_input.strip():
                        st.error("لطفا فیلدهای اجباری (نام و نام خانوادگی) را پر کنید.")
                    else:
                        role = login_or_register_email_user(email_input, first_name_input.strip(), last_name_input.strip(), phone_input.strip(), org_input.strip())
                        st.session_state.logged_in = True
                        st.session_state.username = email_input
                        st.session_state.role = role
                        log_user_activity(email_input, f"LOGIN (Email: {first_name_input} {last_name_input} - Org: {org_input})")
                        st.success(f"ورود موفق با ایمیل: {email_input}")
                        st.rerun()
    st.stop()

# ==========================================
# بدنه اصلی برنامه
# ==========================================
load_shared_state()

st.sidebar.markdown(f"### 👤 سلام **{st.session_state.username}**")
st.sidebar.caption(f"🛡️ سطح دسترسی: **{'مدیر سیستم (Admin)' if st.session_state.role == 'admin' else 'کاربر عادی (User)'}**")
if st.sidebar.button("🚪 خروج (Logout)", use_container_width=True):
    log_user_activity(st.session_state.username, "LOGOUT")
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")

map_theme = st.sidebar.radio("🎨 تم نقشه (Map Theme)", ["☀️ روز (Light)", "🌙 شب (Dark)"], index=0)

st.sidebar.markdown("---")

DEFAULT_SHAPES_PATH = "default_shapes.zip"
region_configs = {}

if st.session_state.role == 'admin':
    st.sidebar.header("⚙️ تنظیمات پردازش و مدل")
    uploaded_shapefile_zip = st.sidebar.file_uploader("آپلود فایل شیپ‌فایل مناطق (.zip) - اختیاری", type="zip")
    
    extract_path = os.path.join(output_dir, "extracted_shapes")
    os.makedirs(extract_path, exist_ok=True)
    
    zip_to_extract = uploaded_shapefile_zip if uploaded_shapefile_zip else DEFAULT_SHAPES_PATH if os.path.exists(DEFAULT_SHAPES_PATH) else None
    
    if zip_to_extract is not None:
        try:
            with zipfile.ZipFile(zip_to_extract, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            shp_files = [os.path.join(r, f) for r, d, files in os.walk(extract_path) for f in files if f.endswith('.shp') and not f.startswith('._')]
            
            if shp_files:
                st.sidebar.subheader("📌 تنظیمات اختصاصی هر منطقه")
                DEFAULT_REGION_DEFAULTS = {"Persian Gulf": {"sst_w": 0.70, "thresh": 0.40}}
                
                for shp_path in sorted(shp_files):
                    reg_name = os.path.splitext(os.path.basename(shp_path))[0].replace("_", " ").title()
                    def_sst = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get("sst_w", 0.60)
                    def_thresh = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get("thresh", 0.50)
                    
                    with st.sidebar.expander(f"منطقه: {reg_name}", expanded=True):
                        sst_w = st.slider(f"وزن SST ({reg_name})", 0.0, 1.0, def_sst, 0.05, key=f"sst_{reg_name}")
                        chl_w = round(1.0 - sst_w, 2)
                        st.caption(f"وزن کلروفیل-آ: **{chl_w}**")
                        thresh = st.slider(f"آستانه حساسیت ({reg_name})", 0.1, 1.0, def_thresh, 0.05, key=f"thresh_{reg_name}")
                        
                        region_configs[reg_name] = {"shp_path": shp_path, "sst_weight": sst_w, "chl_weight": chl_w, "threshold": thresh}
        except Exception as ex:
            record_error("خطا در استخراج شیپ‌فایل", ex)

st.markdown('<div class="main-title">🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ) 🐟</div>', unsafe_allow_html=True)

# ==========================================
# بخش مدیریت و لاگ ادمین
# ==========================================
if st.session_state.role == 'admin':
    with st.expander("📊 گزارش ورود و خروج کاربران و اطلاعات ثبت‌نامی", expanded=False):
        tab_log1, tab_log2 = st.tabs(["📝 لاگ فعالیت‌ها", "👥 لیست کاربران ثبت‌نام‌شده"])
        
        with tab_log1:
            try:
                conn = sqlite3.connect(DB_PATH)
                logs_df = pd.read_sql_query("SELECT username AS 'نام کاربری/ایمیل', action AS 'عملیات', timestamp AS 'زمان' FROM user_logs ORDER BY id DESC LIMIT 50", conn)
                conn.close()
                if not logs_df.empty:
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("هنوز گزارشی ثبت نشده است.")
            except Exception as e:
                st.error(f"خطا در خواندن لاگ کاربران: {e}")
                
        with tab_log2:
            try:
                conn = sqlite3.connect(DB_PATH)
                users_df = pd.read_sql_query("SELECT username AS 'ایمیل/نام‌کاربری', first_name AS 'نام', last_name AS 'نام خانوادگی', phone AS 'تلفن', organization AS 'سازمان', role AS 'نقش' FROM users", conn)
                conn.close()
                if not users_df.empty:
                    st.dataframe(users_df, use_container_width=True)
                else:
                    st.info("کاربری ثبت نشده است.")
            except Exception as e:
                st.error(f"خطا در خواندن لیست کاربران: {e}")

# ==========================================
# توابع پردازش جبهه و رندر حرارتی
# ==========================================
def apply_visual_offset_to_bounds(bounds, lon_offset=0.0, lat_offset=0.0):
    """اعمال شیفت دیداری خالص بر روی Bounding Box لایه Folium."""
    if not bounds:
        return bounds
    sw, ne = bounds
    return [
        [sw[0] + lat_offset, sw[1] + lon_offset],
        [ne[0] + lat_offset, ne[1] + lon_offset]
    ]

def generate_fronts_fallback(nc_path, user_threshold, region_name):
    try:
        if not nc_path or not os.path.exists(nc_path): return None
        with xr.open_dataset(nc_path) as ds:
            var_key = "pfz_index" if "pfz_index" in ds else list(ds.data_vars.keys())[0]
            da = ds[var_key].load()
            lat_name, lon_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None), next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            if not lat_name or not lon_name: return None
            da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
            if da.ndim > 2:
                for d in [d for d in da.dims if d not in [lat_name, lon_name]]: da = da.isel({d: 0})
            data, lats, lons = da.values.copy(), da[lat_name].values, da[lon_name].values
            
        valid_mask = ~np.isnan(data) & (data > 0)
        if not valid_mask.any(): return None

        data_smoothed = ndimage.gaussian_filter(np.where(valid_mask, data, float(np.nanmean(data[valid_mask]))), sigma=1.0).astype(float)
        eroded_mask = ndimage.binary_erosion(valid_mask, structure=np.ones((3, 3)), iterations=1)
        if not eroded_mask.any(): eroded_mask = valid_mask

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

        lines = extract_lines(active_threshold) or extract_lines(smooth_max * 0.60) or extract_lines(smooth_max * 0.40)
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Region'] = region_name
            gdf_fronts['Threshold'] = active_threshold
            gdf_fronts['Length_km'] = gdf_fronts.to_crs("EPSG:3857").geometry.length / 1000
            return gdf_fronts[gdf_fronts['Length_km'] > 0.5] if not gdf_fronts[gdf_fronts['Length_km'] > 0.5].empty else None
    except Exception as ex: record_error(f"خطا در استخراج جبهه برای {region_name}", ex)
    return None

def mask_pfz_by_fronts(da, fronts_gdf, buffer_deg=0.06):
    """برش لایه PFZ تنها در حریم/محدوده اطراف خطوط جبهه صیادی با مختصات واقعی"""
    if fronts_gdf is None or fronts_gdf.empty or da is None:
        return None
    try:
        lat_name, lon_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None), next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        if not lat_name or not lon_name: return None

        lats, lons = da[lat_name].values, da[lon_name].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        fronts_union = fronts_gdf.geometry.buffer(buffer_deg).unary_union
        
        try:
            from shapely.vectorized import contains
            mask = contains(fronts_union, lon_grid, lat_grid)
        except ImportError:
            from shapely.prepared import prep
            prep_poly = prep(fronts_union)
            mask = np.array([prep_poly.contains(Point(x, y)) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]).reshape(lon_grid.shape)

        da_masked = da.copy()
        da_masked.values[~mask] = np.nan
        return da_masked
    except Exception as e:
        record_error("خطا در برش لایه PFZ روی جبهه‌ها", e)
        return None

def render_pixel_perfect_heatmap(da, label, reg_name, cmap_name, out_dir):
    try:
        lat_name, lon_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None), next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        if not lat_name or not lon_name: return None, None
        if extra_dims := [d for d in da.dims if d not in [lat_name, lon_name]]:
            for d in extra_dims: da = da.isel({d: 0})

        da = da.sortby(lat_name, ascending=True).sortby(lon_name, ascending=True)
        lats, lons, data_arr = da[lat_name].values, da[lon_name].values, da.values.copy().astype(float)
        
        dx, dy = float(np.abs(lons[1] - lons[0])) / 2.0 if len(lons) > 1 else 0.025, float(np.abs(lats[1] - lats[0])) / 2.0 if len(lats) > 1 else 0.025
        grid_minx, grid_maxx, grid_miny, grid_maxy = float(lons[0]) - dx, float(lons[-1]) + dx, float(lats[0]) - dy, float(lats[-1]) + dy

        valid_mask = ~np.isnan(data_arr) & (data_arr > 0)
        if not valid_mask.any(): return None, None

        vmin, vmax = float(np.nanmin(data_arr[valid_mask])), float(np.nanmax(data_arr[valid_mask]))
        norm_arr = (data_arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(data_arr)
        
        rgba_img = plt.get_cmap(cmap_name)(norm_arr)
        rgba_img[~valid_mask] = [0.0, 0.0, 0.0, 0.0]
        rgba_img = np.flipud(rgba_img)

        file_path = os.path.join(out_dir, f"{label}_{reg_name.replace(' ', '_')}.png")
        Image.fromarray((rgba_img * 255.0).clip(0, 255).astype(np.uint8), 'RGBA').save(file_path)
        return file_path, [[grid_miny, grid_minx], [grid_maxy, grid_maxx]]
    except Exception as ex:
        record_error(f"خطا در رندر پیکسل برای {label}", ex)
        return None, None

def load_and_crop_dataset(nc_path, shp_path):
    if not nc_path or not os.path.exists(nc_path): return None
    try:
        with xr.open_dataset(nc_path) as ds:
            target_var = None
            preferred_vars = ['analysed_sst', 'sst', 'chlor_a', 'chl', 'pfz_index']
            for v in preferred_vars:
                if v in ds.data_vars:
                    target_var = v
                    break
            if not target_var:
                target_var = list(ds.data_vars.keys())[0] if ds.data_vars else None
                
            if not target_var:
                return None
            da = ds[target_var].load()

        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        lat_name, lon_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None), next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
        return da.sortby(lat_name).sortby(lon_name).sel({lat_name: slice(miny - 0.05, maxy + 0.05), lon_name: slice(minx - 0.05, maxx + 0.05)}) if lat_name and lon_name else None
    except Exception as ex:
        record_error(f"خطا در برش داده {nc_path}", ex)
    return None

# دکمه اجرا در منوی سمت راست برای ادمین
if st.session_state.role == 'admin':
    if st.sidebar.button("🚀 دریافت داده‌های به‌روز و اجرای تحلیل", use_container_width=True):
        if not region_configs:
            st.error("تنظیمات مناطق بارگذاری نشده است.")
        else:
            st.session_state.process_logs = []
            with st.status("🚀 شروع فرآیند پردازش داده‌های مکانی چندمنطقه‌ای...", expanded=True) as status:
                try:
                    log_process("info", "در حال خواندن شیپ‌فایل‌های منطقه‌ای...", status)
                    all_gdfs = []
                    for reg_name, cfg in region_configs.items():
                        temp_gdf = gpd.read_file(cfg["shp_path"]).to_crs("EPSG:4326")
                        temp_gdf["Region"] = reg_name
                        all_gdfs.append(temp_gdf)

                    combined_region_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
                    minx, miny, maxx, maxy = combined_region_gdf.total_bounds

                    log_process("info", "در حال دریافت داده‌های SST و CHL از سرور کوپرنیکوس (Copernicus Marine)...", status)
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)

                    if sst_nc_path and chl_nc_path:
                        log_process("success", "داده‌های ماهواره‌ای با موفقیت از سرور دریافت شدند.", status)
                        all_front_gdfs, nc_out_list = [], []

                        for reg_name, cfg in region_configs.items():
                            log_process("info", f"در حال پردازش **{reg_name}**...", status)
                            try:
                                nc_out, _, _ = process_pfz_pipeline(cfg["shp_path"], sst_nc_path, chl_nc_path, os.path.join(output_dir, reg_name.replace(" ", "_")), cfg["sst_weight"], cfg["chl_weight"])
                            except Exception as proc_ex:
                                record_error(f"خطا در مدل منطقه {reg_name}", proc_ex)
                                nc_out = None

                            if nc_out:
                                nc_out_list.append((reg_name, nc_out, cfg["shp_path"]))
                                reg_fronts_gdf = generate_fronts_fallback(nc_out, cfg["threshold"], reg_name)
                                if reg_fronts_gdf is not None:
                                    all_front_gdfs.append(reg_fronts_gdf)
                                    log_process("success", f"جبهه‌های منطقه {reg_name} استخراج گردید.", status)

                        st.session_state.combined_fronts_gdf = pd.concat(all_front_gdfs, ignore_index=True) if all_front_gdfs else None
                        st.session_state.combined_region_gdf = combined_region_gdf
                        st.session_state.nc_out_list = nc_out_list
                        st.session_state.sst_nc_path = sst_nc_path
                        st.session_state.chl_nc_path = chl_nc_path
                        st.session_state.latest_date = latest_date
                        st.session_state.minx, st.session_state.miny, st.session_state.maxx, st.session_state.maxy = minx, miny, maxx, maxy
                        st.session_state.analysis_done = True
                        
                        save_shared_state()
                        status.update(label="تمام مراحل پردازش با موفقیت به پایان رسید!", state="complete")
                    else:
                        log_process("error", "فایل‌های SST یا CHL دریافت نشدند.", status)
                        status.update(label="پردازش متوقف شد", state="error")

                except Exception as global_ex:
                    record_error("خطای کلی در جریان اجرای برنامه", global_ex)
                    status.update(label="اجرای برنامه با خطا متوقف شد", state="error")

    if st.session_state.process_logs:
        with st.expander("📝 گزارش مختصر مراحل پردازش", expanded=False):
            for msg_type, text in st.session_state.process_logs:
                st.success(text) if msg_type == "success" else st.error(text) if msg_type == "error" else st.info(text)

# ==========================================
# ۳. رندر نقشه تعاملی و لایه‌های رنگی PFZ
# ==========================================
if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.markdown('<h3 style="text-align: right; color: #1E3A8A; font-weight: bold; margin-top: 1rem;">🗺️ نقشه تعاملی خطوط جبهه و لایه‌های پایه</h3>', unsafe_allow_html=True)

    try:
        is_dark = "شب" in map_theme
        
        m = folium.Map(
            location=[(st.session_state.miny + st.session_state.maxy)/2, (st.session_state.minx + st.session_state.maxx)/2], 
            zoom_start=6, 
            tiles=None
        )
        
        if is_dark:
            folium.TileLayer('CartoDB dark_matter', name='نقشه تاریک (Dark Mode)', show=True).add_to(m)
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='تصاویر ماهواره‌ای گوگل (Satellite)', overlay=False, control=True, show=False).add_to(m)
            folium.TileLayer('OpenStreetMap', name='نقشه خیابانی (OSM)', show=False).add_to(m)
        else:
            folium.TileLayer('OpenStreetMap', name='نقشه خیابانی (OSM)', show=True).add_to(m)
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='تصاویر ماهواره‌ای گوگل (Satellite)', overlay=False, control=True, show=False).add_to(m)
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', name='نقشه ترکیبی گوگل (Hybrid)', overlay=False, control=True, show=False).add_to(m)
            folium.TileLayer('CartoDB dark_matter', name='نقشه تاریک (Dark Mode)', show=False).add_to(m)

        CustomMapFeatures().add_to(m)
        
        Fullscreen(
            position="topright",
            title="حالت تمام‌صفحه (Fullscreen)",
            title_cancel="خروج از حالت تمام‌صفحه",
            force_separate_button=True
        ).add_to(m)
        
        # ----------------------------------------------------
        # الف) رندر لایه‌های رنگی پهنه‌بندی (با شیفت دیداری 0.005)
        # ----------------------------------------------------
        if st.session_state.nc_out_list:
            for reg_name, nc_out, reg_shp_path in st.session_state.nc_out_list:
                if nc_out and os.path.exists(nc_out):
                    try:
                        with xr.open_dataset(nc_out) as ds_pfz:
                            var_key = "pfz_index" if "pfz_index" in ds_pfz else list(ds_pfz.data_vars.keys())[0]
                            da_pfz = ds_pfz[var_key].load()
                            
                            # ۱. لایه اصلی پهنه‌بندی PFZ برای کل منطقه
                            img_path, bounds = render_pixel_perfect_heatmap(da_pfz, "PFZ_Full", reg_name, "jet", output_dir)
                            if img_path and bounds and os.path.exists(img_path):
                                shifted_bounds = apply_visual_offset_to_bounds(bounds, lon_offset=OFFSET_LON_DEG, lat_offset=OFFSET_LAT_DEG)
                                encoded_img = image_to_base64(img_path)
                                if encoded_img:
                                    folium.raster_layers.ImageOverlay(
                                        image=encoded_img, 
                                        bounds=shifted_bounds, 
                                        opacity=0.70, 
                                        name=f"🌊 پهنه‌بندی کل منطقه - PFZ Index ({reg_name})", 
                                        show=True
                                    ).add_to(m)

                            # ۲. لایه هوشمند PFZ محدود به محدوده جبهه‌ها
                            if st.session_state.combined_fronts_gdf is not None:
                                da_masked = mask_pfz_by_fronts(da_pfz, st.session_state.combined_fronts_gdf)
                                if da_masked is not None:
                                    m_path, m_bounds = render_pixel_perfect_heatmap(da_masked, "PFZ_Fronts_Masked", reg_name, "jet", output_dir)
                                    if m_path and m_bounds and os.path.exists(m_path):
                                        shifted_m_bounds = apply_visual_offset_to_bounds(m_bounds, lon_offset=OFFSET_LON_DEG, lat_offset=OFFSET_LAT_DEG)
                                        enc_masked = image_to_base64(m_path)
                                        if enc_masked:
                                            folium.raster_layers.ImageOverlay(
                                                image=enc_masked,
                                                bounds=shifted_m_bounds,
                                                opacity=0.85,
                                                name=f"🎯 الگوی رنگی PFZ فقط در حریم جبهه‌ها ({reg_name})",
                                                show=True
                                            ).add_to(m)

                    except Exception as pfz_ex: record_error("خطا لایه PFZ", pfz_ex)

                # ۳. لایه دمای سطح دریا (SST)
                if st.session_state.sst_nc_path:
                    try:
                        da_sst = load_and_crop_dataset(st.session_state.sst_nc_path, reg_shp_path)
                        if da_sst is not None:
                            img_path_sst, bounds_sst = render_pixel_perfect_heatmap(da_sst, "SST", reg_name, "coolwarm", output_dir)
                            if img_path_sst and bounds_sst and os.path.exists(img_path_sst):
                                shifted_sst_bounds = apply_visual_offset_to_bounds(bounds_sst, lon_offset=OFFSET_LON_DEG, lat_offset=OFFSET_LAT_DEG)
                                encoded_img_sst = image_to_base64(img_path_sst)
                                if encoded_img_sst:
                                    folium.raster_layers.ImageOverlay(
                                        image=encoded_img_sst, 
                                        bounds=shifted_sst_bounds, 
                                        opacity=0.65, 
                                        name=f"🌡️ دمای سطح دریا - SST ({reg_name})", 
                                        show=False
                                    ).add_to(m)
                    except Exception as sst_ex: record_error("خطا لایه SST", sst_ex)

                # ۴. لایه کلروفیل-آ (Chlorophyll-a)
                if st.session_state.chl_nc_path:
                    try:
                        da_chl = load_and_crop_dataset(st.session_state.chl_nc_path, reg_shp_path)
                        if da_chl is not None:
                            img_path_chl, bounds_chl = render_pixel_perfect_heatmap(da_chl, "Chlorophyll-a", reg_name, "YlGn", output_dir)
                            if img_path_chl and bounds_chl and os.path.exists(img_path_chl):
                                shifted_chl_bounds = apply_visual_offset_to_bounds(bounds_chl, lon_offset=OFFSET_LON_DEG, lat_offset=OFFSET_LAT_DEG)
                                encoded_img_chl = image_to_base64(img_path_chl)
                                if encoded_img_chl:
                                    folium.raster_layers.ImageOverlay(
                                        image=encoded_img_chl, 
                                        bounds=shifted_chl_bounds, 
                                        opacity=0.65, 
                                        name=f"🌱 غلظت کلروفیل - Chlorophyll-a ({reg_name})", 
                                        show=False
                                    ).add_to(m)
                    except Exception as chl_ex: record_error("خطا لایه Chl", chl_ex)

        # ----------------------------------------------------
        # ب) خطوط جبهه صیادی (Fronts) - قرمز رنگ
        # ----------------------------------------------------
        if st.session_state.combined_fronts_gdf is not None and not st.session_state.combined_fronts_gdf.empty:
            fronts_fg = folium.FeatureGroup(name="🚩 خطوط جبهه صیادی (Front Lines - Red)", show=True)
            folium.GeoJson(
                st.session_state.combined_fronts_gdf,
                style_function=lambda x: {'color': '#FF0000', 'weight': 3.5, 'opacity': 0.95}
            ).add_to(fronts_fg)
            fronts_fg.add_to(m)

        # ----------------------------------------------------
        # ج) مرز مناطق
        # ----------------------------------------------------
        regions_fg = folium.FeatureGroup(name="📌 محدوده مناطق (Regions)", show=False)
        folium.GeoJson(
            st.session_state.combined_region_gdf,
            style_function=lambda x: {'color': '#007BFF', 'fillColor': 'transparent', 'weight': 2, 'dashArray': '5, 5'}
        ).add_to(regions_fg)
        regions_fg.add_to(m)

        # ----------------------------------------------------
        # د) Legend متناسب با تم
        # ----------------------------------------------------
        bg_color = "rgba(20, 20, 20, 0.90)" if is_dark else "rgba(255, 255, 255, 0.95)"
        text_color = "#FFFFFF" if is_dark else "#1E3A8A"
        item_text_color = "#DDDDDD" if is_dark else "#333333"
        border_color = "#444444" if is_dark else "#2B5B84"

        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 85px; left: 20px; width: 185px; 
                    z-index:9999; font-size:11px; background-color: {bg_color}; 
                    border: 2px solid {border_color}; border-radius: 8px; 
                    padding: 8px; font-weight: bold; color: {text_color};
                    direction: rtl; font-family: 'Vazirmatn', sans-serif;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);">
            <div style="text-align: center; border-bottom: 1px solid #777; padding-bottom: 4px; margin-bottom: 6px; font-size: 12px; font-weight: bold;">
                🎯 پتانسیل صید (PFZ)
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                <span style="width: 14px; height: 14px; background: #800000; display: inline-block; border-radius: 3px;"></span>
                <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">عالی (بسیار بالا)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                <span style="width: 14px; height: 14px; background: #FF0000; display: inline-block; border-radius: 3px;"></span>
                <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">خوب (بالا)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                <span style="width: 14px; height: 14px; background: #00FF00; display: inline-block; border-radius: 3px;"></span>
                <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">متوسط</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                <span style="width: 14px; height: 14px; background: #00FFFF; display: inline-block; border-radius: 3px;"></span>
                <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">کم</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="width: 14px; height: 14px; background: #000080; display: inline-block; border-radius: 3px;"></span>
                <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">بسیار کم / ناچیز</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
        if greg_str and jalali_str:
            persian_digits = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
            jalali_str_fa = jalali_str.translate(persian_digits)
            
            date_box_html = f'''
                <div style="position: fixed; 
                            bottom: 25px; left: 20px; width: 250px; height: 50px; 
                            z-index:9999; font-size:12px; background-color: {bg_color}; 
                            border: 2px solid {border_color}; border-radius: 6px; 
                            padding: 4px; font-weight: bold; text-align: center; color: {text_color}; line-height: 1.4;
                            direction: rtl; font-family: 'Vazirmatn', sans-serif;">
                    تاریخ اخذ داده: {jalali_str_fa}<br>
                    <span style="font-family: Arial, sans-serif; color: {item_text_color}; font-size: 11px;">Data Acquisition Date: {greg_str}</span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(date_box_html))

        m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
        folium.LayerControl(position='topright', collapsed=True).add_to(m)
        
        st_folium(m, width=1100, height=600)

    except Exception as map_render_err:
        st.error("⚠️ خطا در پردازش و رندر نقشه:")
        st.exception(map_render_err)
else:
    if st.session_state.role != 'admin':
        st.warning("⚠️ هنوز هیچ دیتایی توسط مدیر سیستم پردازش و ذخیره نشده است.")
    else:
        st.info("👈 برای شروع، از منوی تنظیمات کناری، فرآیند پردازش را اجرا کنید.")