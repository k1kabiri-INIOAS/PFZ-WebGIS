# File Name: app.py
# Description: Streamlit GUI main application file for PFZ management system with Folium WebGIS integration.

import os
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
st.markdown("سامانه یکپارچه اقیانوس‌شناسی WebGIS برای شناسایی مناطق مستعد صید ماهیان در خلیج فارس، دریای عمان و دریای عرب.")

# Sidebar configuration inputs
st.sidebar.header("تنظیمات پردازش و مدل")
shapefile_path = st.sidebar.text_input("مسیر شیپ‌فایل منطقه:", r"F:\\PFZ\\Shape_Files\\SA.shp")
output_dir = st.sidebar.text_input("مسیر ذخیره خروجی‌ها:", r"F:\\PFZ\\Data\\Processed")

st.sidebar.subheader("وزن‌دهی پارامترها")
sst_weight = st.sidebar.slider("وزن جبهه‌های حرارتی SST", 0.0, 1.0, 0.5, 0.1)
chl_weight = st.sidebar.slider("وزن کلروفیل-آ (Chlorophyll-a)", 0.0, 1.0, 0.5, 0.1)

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if not os.path.exists(shapefile_path):
        st.error("فایل شیپ‌فایل یافت نشد لطفا مسیر را بررسی کنید.")
    else:
        with st.spinner("در حال اتصال به سرور و دریافت داده‌های به‌روز SST و کلروفیل..."):
            gdf = gpd.read_file(shapefile_path)
            minx, miny, maxx, maxy = gdf.total_bounds
            
            sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
            
            if sst_nc_path and chl_nc_path:
                st.success(f"داده‌های تاریخ {latest_date} با موفقیت دریافت شدند.")
                
                with st.spinner("در حال محاسبه مدل چندمتواره PFZ..."):
                    nc_out, tif_out = process_pfz_pipeline(
                        shapefile_path, sst_nc_path, chl_nc_path, output_dir, sst_weight, chl_weight
                    )
                    
                st.success("تحلیل چندمتواره با موفقیت به پایان رسید!")
                st.info(f"فایل نهایی در مسیر زیر ذخیره شد:\n`{tif_out}`")
                
                # WebGIS Folium Map Integration
                st.subheader("🗺️ نقشه تعاملی WebGIS مناطق مستعد صید (PFZ)")
                
                center_lat = (miny + maxy) / 2
                center_lon = (minx + maxx) / 2
                
                m = folium.Map(
                    location=[center_lat, center_lon], 
                    zoom_start=6, 
                    tiles="CartoDB positron"
                )
                
                # Add region boundary polygon to map
                folium.GeoJson(
                    gdf,
                    name="IHO Region Boundary",
                    style_function=lambda x: {'color': 'blue', 'fillColor': 'transparent', 'weight': 2}
                ).add_to(m)
                
                # Generate overlay image from pfz_index for Folium
                ds_res = xr.open_dataset(nc_out)
                pfz_da = ds_res["pfz_index"]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                pfz_da.plot.imshow(
                    ax=ax, 
                    cmap="jet", 
                    alpha=0.6, 
                    vmin=0, 
                    vmax=1,
                    add_colorbar=False
                )
                
                overlay_path = os.path.join(output_dir, "pfz_overlay.png")
                fig.savefig(overlay_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=150)
                plt.close(fig)
                
                # Add image overlay to Folium map
                folium.raster_layers.ImageOverlay(
                    image=overlay_path,
                    bounds=[[miny, minx], [maxy, maxx]],
                    opacity=0.7,
                    name="PFZ Index Overlay"
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                # Render map in Streamlit
                st_folium(m, width=1100, height=600)
                
            else:
                st.error("خطا در دریافت داده‌های ماهواره‌ای.")