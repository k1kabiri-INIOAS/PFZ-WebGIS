import os
import tempfile
import zipfile
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# تنظیمات صفحه استریم‌لیت
st.set_page_config(
    page_title="سامانه پایش جبهه‌ها و مناطق حاشیه‌ای اقیانوسی (PFZ)",
    layout="wide"
)

# اطمینان از وجود پوشه خروجی
os.makedirs("outputs", exist_ok=True)

st.title("🌊 سامانه پایش و تحلیل پویایی جبهه‌های اقیانوسی")
st.markdown("سامانه وب‌جی‌آیس تخصصی برای نمایش، تحلیل و پایش جبهه‌های اقیانوسی و محدوده‌های مطالعاتی.")

# نوار کناری (Sidebar) برای بارگذاری داده‌های مکانی
st.sidebar.header("⚙️ داده‌های ورودی مکانی")
uploaded_file = st.sidebar.file_uploader(
    "بارگذاری محدوده یا فایل برداری (Shapefile فشرده .zip یا GeoJSON)",
    type=["zip", "geojson"]
)

geojson_output_path = "outputs/pfz_fronts.geojson"
gdf = None

# پردازش فایل آپلود شده
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")]
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    # ذخیره به عنوان خروجی پیش‌فرض سامانه
                    gdf.to_file(geojson_output_path, driver="GeoJSON")
                    st.sidebar.success(f"✅ محدوده بارگذاری شد (تعداد عوارض: {len(gdf)})")
                else:
                    st.sidebar.error("❌ فایل .shp درون فایل فشرده یافت نشد.")
        
        elif uploaded_file.name.endswith(".geojson"):
            gdf = gpd.read_file(uploaded_file)
            gdf.to_file(geojson_output_path, driver="GeoJSON")
            st.sidebar.success("✅ فایل GeoJSON با موفقیت بارگذاری شد!")
    except Exception as e:
        st.sidebar.error(f"❌ خطا در خواندن فایل برداری: {e}")

# اگر فایلی آپلود نشده باشد اما فایل خروجی قبلی روی دیسک موجود باشد، آن را بارگذاری می‌کنیم
if gdf is None and os.path.exists(geojson_output_path):
    try:
        gdf = gpd.read_file(geojson_output_path)
    except Exception:
        pass

# بخش اصلی: نمایش نقشه تعاملی
st.subheader("🗺️ نقشه تعاملی مناطق و جبهه‌های اقیانوسی")

# ایجاد نقشه پایه فولیوم (متمرکز روی خلیج فارس / منطقه جنوب ایران به صورت پیش‌فرض)
m = folium.Map(location=[26.5, 54.0], zoom_start=6, tiles="CartoDB positron")

if gdf is not None and not gdf.empty:
    # اطمینان از سیستم مختصات جغرافیایی WGS84
    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # افزودن لایه برداری به نقشه
    folium.GeoJson(
        gdf,
        name="محدوده / جبهه",
        style_function=lambda x: {
            'color': '#1f77b4',
            'weight': 2.5,
            'fillColor': '#ff7f0e',
            'fillOpacity': 0.2
        }
    ).add_to(m)
    
    # تنظیم خودکار زوم نقشه روی مختصات لایه
    bounds = gdf.total_bounds  # [xmin, ymin, xmax, ymax]
    if len(bounds) == 4 and not pd.isna(bounds).any():
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

folium.LayerControl().add_to(m)

# رندر نقشه در استریم‌لیت با استفاده از streamlit-folium
st_data = st_folium(m, width=1250, height=520)

st.divider()

# بخش مدیریت فایل، دانلود و جدول اطلاعات توصیفی
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 دانلود خروجی")
    if os.path.exists(geojson_output_path):
        with open(geojson_output_path, "rb") as file:
            st.download_button(
                label="دانلود فایل نهایی (GeoJSON)",
                data=file,
                file_name="pfz_fronts.geojson",
                mime="application/json"
            )
    else:
        st.info("هنوز فایلی برای دانلود موجود نیست.")

with col2:
    st.subheader("📊 اطلاعات آماری و عوارض")
    if gdf is not None and not gdf.empty:
        st.write(f"تعداد کل عوارض ثبت‌شده: **{len(gdf)}**")
    else:
        st.warning("⚠️ هیچ لایه مکانی فعال یا بارگذاری نشده است. لطفاً یک فایل Shapefile یا GeoJSON از منوی کناری بارگذاری کنید.")

# نمایش جدول اطلاعات توصیفی در صورت وجود داده
if gdf is not None and not gdf.empty:
    st.subheader("📋 جدول اطلاعات توصیفی (Attribute Table)")
    st.dataframe(gdf.drop(columns='geometry', errors='ignore'), use_container_width=True)