# File Name: app.py
import os
import tempfile
import zipfile
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

# جلوگیری از چاپ هشدارهای متپلوت‌لیب در لاگ سرور
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

st.set_page_config(page_title="PFZ Management System", layout="wide")

st.title("🌊 سامانه هوشمند تشخیص مناطق مستعد صید (PFZ)")
st.markdown("سامانه یکپارچه اقیانوس‌شناسی WebGIS با قابلیت استخراج دینامیک خطوط جبهه.")

st.sidebar.header("تنظیمات پردازش و مدل")

uploaded_shapefile_zip = st.sidebar.file_uploader(
    "آپلود فایل فشرده شیپ‌فایل منطقه (.zip)", 
    type="zip",
    help="لطفاً فایل‌های شیپ‌فایل خود (.shp, .shx, .dbf, .prj) را در یک فایل فشرده (ZIP) قرار داده و آپلود کنید."
)

output_dir = "Data_Processed"

st.sidebar.subheader("پارامترهای مدل")
sst_weight = st.sidebar.slider("وزن جبهه‌های حرارتی SST", 0.0, 1.0, 0.5, 0.1)
chl_weight = st.sidebar.slider("وزن کلروفیل-آ (Chlorophyll-a)", 0.0, 1.0, 0.5, 0.1)

st.sidebar.subheader("تنظیمات استخراج عوارض")
pfz_threshold = st.sidebar.slider(
    "آستانه حساسیت جبهه‌ها (Threshold)", 
    0.1, 1.0, 0.45, 0.05, 
    help="اگر هیچ خطی روی نقشه ظاهر نمی‌شود، این مقدار را کاهش دهید تا جبهه‌های ضعیف‌تر نیز شناسایی شوند."
)

def generate_fronts_fallback(nc_path, output_geojson_path, user_threshold):
    try:
        ds = xr.open_dataset(nc_path)
        if "pfz_index" not in ds:
            return False
        
        da = ds["pfz_index"]
        lat_name = 'lat' if 'lat' in da.dims else ('latitude' if 'latitude' in da.dims else da.dims[0])
        lon_name = 'lon' if 'lon' in da.dims else ('longitude' if 'longitude' in da.dims else da.dims[1])
        
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        data = da.values
        
        if data.ndim == 3:
            data = data[0, :, :]
            
        valid_mask = ~np.isnan(data)
        if not valid_mask.any():
            st.warning("⚠️ داده‌های محاسباتی تماماً خالی (NaN) هستند.")
            return False
            
        raw_max = float(np.nanmax(data))
        
        # نرم‌سازی ماتریس
        data_filled = np.nan_to_num(data, nan=0.0)
        data_smoothed = ndimage.gaussian_filter(data_filled, sigma=1.0)
        data_smoothed[~valid_mask] = np.nan
        
        # محاسبه ماکزیمم جدید پس از فیلتر گوسی
        valid_smoothed = data_smoothed[valid_mask]
        smooth_max = float(np.nanmax(valid_smoothed))
        
        st.info(f"📊 **حداکثر شاخص خام:** {raw_max:.3f} | **حداکثر پس از نرم‌سازی:** {smooth_max:.3f}")
        
        # آداپته کردن آستانه کاربر با واقعیتِ ماتریس نرم‌شده
        active_threshold = user_threshold
        if active_threshold >= smooth_max:
            active_threshold = smooth_max * 0.85  # تنظیم روی ۸۵ درصد ماکزیمم جدید
            st.warning(f"⚠️ به دلیل اعمال فیلتر نرم‌ساز، آستانه شما به طور خودکار به {active_threshold:.3f} تعدیل شد.")

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = []
            for collection in cs.collections:
                for path in collection.get_paths():
                    verts = path.vertices
                    if len(verts) > 1:
                        extracted.append(LineString(verts))
            plt.close(fig)
            return extracted

        # تلاش اول برای رسم خط
        lines = extract_lines(active_threshold)
        
        # سیستم پشتیبان بر اساس درصدهای ماکزیمم نرم‌شده
        if not lines:
            fallback_percents = [0.70, 0.50, 0.30, 0.10]
            st.warning("⚠️ در آستانه اولیه خط ممتدی یافت نشد. در حال اسکن اعماق ماتریس...")
            for pct in fallback_percents:
                test_t = smooth_max * pct
                if test_t <= 0: continue
                lines = extract_lines(test_t)
                if lines:
                    st.success(f"✅ جبهه‌ها با موفقیت در آستانه جایگزین ({test_t:.3f}) پیدا شدند!")
                    active_threshold = test_t
                    break
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Threshold'] = active_threshold
            # افزودن طول برای فیلتر کردن خطوط بسیار کوتاه (نویزها)
            gdf_fronts = gdf_fronts.to_crs("EPSG:3857") # تبدیل به سیستم متریک موقت
            gdf_fronts['Length_km'] = gdf_fronts.geometry.length / 1000
            gdf_fronts = gdf_fronts.to_crs("EPSG:4326") # بازگشت به WGS84
            
            # حذف خطوطی که از 1 کیلومتر کوتاه‌تر هستند (نویزهای نقطه‌ای)
            gdf_fronts = gdf_fronts[gdf_fronts['Length_km'] > 1.0]
            
            if not gdf_fronts.empty:
                os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
                gdf_fronts.to_file(output_geojson_path, driver="GeoJSON")
                return True
            else:
                st.warning("⚠️ جبهه‌ها شناسایی شدند اما طول آن‌ها برای تشکیل یک زون صیادی پیوسته بسیار کوتاه بود.")
                return False
        else:
            st.error("⚠️ ماتریس داده کاملاً یکنواخت است و هیچ گرادیان (تغییرات) معناداری برای رسم خط در آن وجود ندارد.")
            return False
            
    except Exception as ex:
        st.error(f"خطا در پردازش ماتریس: {ex}")
    return False

# Session State Initialization
for key in ["analysis_done", "nc_out", "tif_out", "fronts_geojson", "gdf", "minx", "miny", "maxx", "maxy"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

if st.sidebar.button("دریافت داده‌های به‌روز و اجرای تحلیل"):
    if uploaded_shapefile_zip is None:
        st.error("لطفاً فایل فشرده شیپ‌فایل منطقه (.zip) را در سایدبار آپلود کنید.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_shapefile_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            shp_files = [os.path.join(r, f) for r, d, files in os.walk(tmpdir) for f in files if f.endswith('.shp')]
            
            if not shp_files:
                st.error("فایل با پسوند .shp در داخل فایل فشرده یافت نشد.")
            else:
                shapefile_path = shp_files[0]
                
                with st.spinner("در حال اتصال به سرور و دریافت داده‌ها..."):
                    gdf = gpd.read_file(shapefile_path)
                    if gdf.crs is not None and gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    minx, miny, maxx, maxy = gdf.total_bounds
                    
                    os.makedirs(output_dir, exist_ok=True)
                    sst_nc_path, chl_nc_path, latest_date = fetch_near_realtime_data(minx, miny, maxx, maxy, output_dir)
                    
                    if sst_nc_path and chl_nc_path:
                        st.success(f"✅ داده‌های تاریخ {latest_date} با موفقیت دریافت شدند.")
                        
                        with st.spinner("در حال محاسبه مدل و استخراج خطوط جبهه..."):
                            nc_out, tif_out, fronts_geojson = process_pfz_pipeline(
                                shapefile_path, sst_nc_path, chl_nc_path, output_dir, sst_weight, chl_weight
                            )
                            
                            target_geojson = os.path.join(output_dir, "pfz_fronts.geojson")
                            fallback_success = generate_fronts_fallback(nc_out, target_geojson, user_threshold=pfz_threshold)
                            
                            if fallback_success:
                                fronts_geojson = target_geojson
                            else:
                                fronts_geojson = None
                            
                            st.session_state.analysis_done = True
                            st.session_state.nc_out = nc_out
                            st.session_state.fronts_geojson = fronts_geojson
                            st.session_state.gdf = gdf
                            st.session_state.minx, st.session_state.miny, st.session_state.maxx, st.session_state.maxy = minx, miny, maxx, maxy
                    else:
                        st.error("خطا در دریافت داده‌های ماهواره‌ای.")

# Render Results
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
                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                pfz_da.plot.imshow(ax=ax, cmap="jet", alpha=0.5, add_colorbar=False)
                
                overlay_path = os.path.join(output_dir, "pfz_overlay.png")
                fig.savefig(overlay_path, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                folium.raster_layers.ImageOverlay(
                    image=overlay_path,
                    bounds=[[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]],
                    opacity=0.6,
                    name="PFZ Index Heatmap"
                ).add_to(m)
        except Exception:
            pass

    fronts_gdf = None
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
        except Exception:
            pass

    m.fit_bounds([[st.session_state.miny, st.session_state.minx], [st.session_state.maxy, st.session_state.maxx]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)
    
    if fronts_gdf is not None and not fronts_gdf.empty:
        st.subheader("📋 جدول اطلاعات عوارض خطوط جبهه استخراج‌شده")
        display_gdf = fronts_gdf.drop(columns='geometry', errors='ignore')
        st.dataframe(display_gdf, use_container_width=True)
        
        with open(st.session_state.fronts_geojson, "rb") as file:
            st.download_button("📥 دانلود خطوط جبهه نهایی (GeoJSON)", data=file, file_name="pfz_front_lines.geojson", mime="application/geo+json")