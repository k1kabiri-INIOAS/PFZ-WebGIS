# File Name: app.py
# Description: Streamlit GUI main application file with INCOIS-style vector fronts rendering.

import os
import tempfile
import zipfile
import streamlit as st
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from modules.fetcher import fetch_near_realtime_data
from modules.processor import process_pfz_pipeline

# تنظیم بک‌اند متپلوت‌لیب برای محیط بدون گرافیک سرور
plt.switch_backend('Agg')

st.set_page_config(page_title="PFZ Management System", layout="wide")

st.title("🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ)")
st.markdown("سامانه یکپارچه اقیانوس‌شناسی WebGIS با قابلیت استخراج خطوط جبهه (INCOIS Style).")

st.sidebar.header("تنظیمات پردازش و مدل")

uploaded_shapefile_zip = st.sidebar.file_uploader(
    "آپلود فایل فشرده شیپ‌فایل منطقه (.zip)", 
    type="zip",
    help="لطفاً فایل‌های شیپ‌فایل خود (.shp, .shx, .dbf, .prj) را در یک فایل فشرده (ZIP) قرار داده و آپلود کنید."
)

output_dir = "Data_Processed"

st.sidebar.subheader("وزن‌دهی پارامترها")
sst_weight = st.sidebar.slider("وزن جبهه‌های حرارتی SST", 0.0, 1.0, 0.5, 0.1)
chl_weight = st.sidebar.slider("وزن کلروفیل-آ (Chlorophyll-a)", 0.0, 1.0, 0.5, 0.1)

# Session State Initialization
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "nc_out" not in st.session_state:
    st.session_state.nc_out = None
if "tif_out" not in st.session_state:
    st.session_state.tif_out = None
if "fronts_geojson" not in st.session_state:
    st.session_state.fronts_geojson = None
if "gdf" not in st.session_state:
    st.session_state.gdf = None
if "minx" not in st.session_state:
    st.session_state.minx = None
if "miny" not in st.session_state:
    st.session_state.miny = None
if "maxx" not in st.session_state:
    st.session_state.maxx = None
if "maxy" not in st.session_state:
    st.session_state.maxy = None

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if uploaded_shapefile_zip is None:
        st.error("لطفاً فایل فشرده شیپ‌فایل منطقه (.zip) را در سایدبار آپلود کنید.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_shapefile_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            shp_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.shp'):
                        shp_files.append(os.path.join(root, file))
            
            if not shp_files:
                st.error("فایل با پسوند .shp در داخل فایل فشرده یافت نشد.")
            else:
                shapefile_path = shp_files[0]
                
                with st.spinner("در حال اتصال به سرور و دریافت داده‌ها..."):
                    gdf = gpd.read_file(shapefile_path)
                    # تبدیل مختصات به WGS84 برای سازگاری کامل با فولیوم
                    if gdf.crs is not None and gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    minx, miny, maxx, maxy = gdf.total_bounds
                    
                    os.makedirs(output_dir, exist_ok=True)
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                    
                    if sst_nc_path and chl_nc_path:
                        st.success(f"داده‌های تاریخ {latest_date} با موفقیت دریافت شدند.")
                        
                        with st.spinner("در حال محاسبه مدل و استخراج خطوط جبهه..."):
                            nc_out, tif_out, fronts_geojson = process_pfz_pipeline(
                                shapefile_path, sst_nc_path, chl_nc_path, output_dir, sst_weight, chl_weight
                            )
                            
                            st.session_state.analysis_done = True
                            st.session_state.nc_out = nc_out
                            st.session_state.tif_out = tif_out
                            st.session_state.fronts_geojson = fronts_geojson
                            st.session_state.gdf = gdf
                            st.session_state.minx = minx
                            st.session_state.miny = miny
                            st.session_state.maxx = maxx
                            st.session_state.maxy = maxy
                            
                            st.success("تحلیل چندمتواره و استخراج جبهه‌ها با موفقیت به پایان رسید!")
                    else:
                        st.error("خطا در دریافت داده‌های ماهواره‌ای.")

# Render Results
if st.session_state.analysis_done and st.session_state.gdf is not None:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه (INCOIS Style)")
    
    center_lat = (st.session_state.miny + st.session_state.maxy) / 2
    center_lon = (st.session_state.minx + st.session_state.maxx) / 2
    
    # ساخت نقشه پایه
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=6, 
        tiles="OpenStreetMap"
    )
    
    # 1. افزودن محدوده مطالعاتی (Region Boundary)
    folium.GeoJson(
        st.session_state.gdf,
        name="Region Boundary",
        style_function=lambda x: {
            'color': '#0000FF', 
            'fillColor': 'transparent', 
            'weight': 2,
            'dashArray': '5, 5'
        }
    ).add_to(m)
    
    # 2. افزودن لایه حرارتی پس‌زمینه (Raster Overlay با اصلاح تنظیمات چاپ تصویر)
    if st.session_state.nc_out and os.path.exists(st.session_state.nc_out):
        try:
            ds_res = xr.open_dataset(st.session_state.nc_out)
            if "pfz_index" in ds_res:
                pfz_da = ds_res["pfz_index"]
                
                # تنظیم دقیق ابعاد پلات بدون حاشیه اضافی برای تطابق صددرصدی با مختصات
                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                pfz_da.plot.imshow(
                    ax=ax, 
                    cmap="coolwarm", 
                    alpha=0.5, 
                    vmin=0, 
                    vmax=1, 
                    add_colorbar=False
                )
                
                overlay_path = os.path.join(output_dir, "pfz_overlay.png")
                fig.savefig(overlay_path, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                folium.raster_layers.ImageOverlay(
                    image=overlay_path,
                    bounds=[[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]],
                    opacity=0.6,
                    name="Background Heatmap"
                ).add_to(m)
        except Exception as e:
            st.warning(f"امکان نمایش لایه پس‌زمینه حرارتی وجود ندارد: {e}")
    
    # 3. افزودن خطوط جبهه (INCOIS-Style Vector Contours)
    fronts_gdf = None
    if st.session_state.fronts_geojson and os.path.exists(st.session_state.fronts_geojson):
        try:
            fronts_gdf = gpd.read_file(st.session_state.fronts_geojson)
            if not fronts_gdf.empty:
                if fronts_gdf.crs is not None and fronts_gdf.crs != "EPSG:4326":
                    fronts_gdf = fronts_gdf.to_crs("EPSG:4326")
                
                folium.GeoJson(
                    fronts_gdf,
                    name="PFZ Front Lines (High Probability)",
                    style_function=lambda x: {
                        'color': '#FF0000', # خطوط قرمز تیره مشخص
                        'weight': 3.5,      # ضخامت مناسب
                        'opacity': 0.9
                    }
                ).add_to(m)
                st.success(f"✅ تعداد {len(fronts_gdf)} خط جبهه روی نقشه بارگذاری شد.")
            else:
                st.warning("⚠️ فایل جبهه‌های استخراج‌شده خالی است (عارضه‌ای یافت نشد).")
        except Exception as err:
            st.error(f"خطا در خواندن فایل GeoJSON جبهه‌ها: {err}")

    # تنظیم محدوده نقشه روی مختصات شیپ‌فایل
    m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)
    
    # نمایش جدول اطلاعات توصیفی عوارض جبهه (Attribute Table)
    if fronts_gdf is not None and not fronts_gdf.empty:
        st.subheader("📋 جدول اطلاعات عوارض خطوط جبهه استخراج‌شده")
        st.dataframe(fronts_gdf.drop(columns='geometry', errors='ignore'), use_container_width=True)

    # دکمه دانلود داده‌های برداری
    if st.session_state.fronts_geojson and os.path.exists(st.session_state.fronts_geojson):
        with open(st.session_state.fronts_geojson, "rb") as file:
            st.download_button(
                label="📥 دانلود خطوط جبهه نهایی (GeoJSON)",
                data=file,
                file_name="pfz_front_lines.geojson",
                mime="application/geo+json"
            )
else:
    st.info("💡 لطفاً ابتدا محدوده مورد نظر خود را از طریق فایل شیپ‌فایل در سایدبار بارگذاری کرده و دکمه‌ی تحلیل را بزنید.")