# File Name: app.py
# Description: Streamlit GUI main application file with INCOIS-style vector fronts rendering.
# test

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
                    minx, miny, maxx, maxy = gdf.total_bounds
                    
                    os.makedirs(output_dir, exist_ok=True)
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                    
                    if sst_nc_path and chl_nc_path:
                        st.success(f"داده‌های تاریخ {latest_date} با موفقیت دریافت شدند.")
                        
                        with st.spinner("در حال محاسبه مدل و استخراج خطوط جبهه..."):
                            # Unpack 3 variables now
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
if st.session_state.analysis_done:
    st.subheader("🗺️ نقشه تعاملی خطوط جبهه (INCOIS Style)")
    
    center_lat = (st.session_state.miny + st.session_state.maxy) / 2
    center_lon = (st.session_state.minx + st.session_state.maxx) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=6, 
        tiles="CartoDB positron"
    )
    
    # 1. Add region boundary polygon
    folium.GeoJson(
        st.session_state.gdf,
        name="Region Boundary",
        style_function=lambda x: {'color': 'blue', 'fillColor': 'transparent', 'weight': 1}
    ).add_to(m)
    
    # 2. Add faint background raster (Context)
    ds_res = xr.open_dataset(st.session_state.nc_out)
    pfz_da = ds_res["pfz_index"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    pfz_da.plot.imshow(ax=ax, cmap="coolwarm", alpha=0.3, vmin=0, vmax=1, add_colorbar=False)
    overlay_path = os.path.join(output_dir, "pfz_overlay.png")
    fig.savefig(overlay_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=100)
    plt.close(fig)
    
    folium.raster_layers.ImageOverlay(
        image=overlay_path,
        bounds=[[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]],
        opacity=0.4,
        name="Background Heatmap"
    ).add_to(m)
    
    # 3. Add INCOIS-Style Vector Contours (The Fronts)
    if st.session_state.fronts_geojson and os.path.exists(st.session_state.fronts_geojson):
        folium.GeoJson(
            st.session_state.fronts_geojson,
            name="PFZ Front Lines (High Probability)",
            style_function=lambda x: {
                'color': '#FF0000', # Solid Red lines
                'weight': 3.5,      # Thick lines like INCOIS
                'opacity': 0.9
            }
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)
    
    # Download Button for the Vector Data
    with open(st.session_state.fronts_geojson, "rb") as file:
        st.download_button(
            label="دانلود خطوط جبهه (GeoJSON)",
            data=file,
            file_name="pfz_front_lines.geojson",
            mime="application/geo+json"
        )