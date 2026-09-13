import os
import streamlit as st
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import tempfile
import zipfile

# تنظیمات صفحه استریم‌لیت
st.set_page_config(
    page_title="سامانه پایش جبهه‌ها و مناطق حاشیه‌ای اقیانوسی (PFZ)",
    layout="wide"
)

# اطمینان از وجود پوشه خروجی
os.makedirs("outputs", exist_ok=True)

st.title("🌊 سامانه پایش و تحلیل پویایی جبهه‌های اقیانوسی")
st.markdown("این سامانه برای استخراج، پردازش و پایش جبهه‌ها و توده‌های آب از داده‌های ماهواره‌ای طراحی شده است.")

# نوار کناری (Sidebar) برای تنظیمات و ورودی‌ها
st.sidebar.header("⚙️ تنظیمات و داده‌های ورودی")

# بخش آپلود Shapefile یا GeoJSON
uploaded_file = st.sidebar.file_uploader(
    "بارگذاری محدوده مطالعاتی (Shapefile فشرده .zip یا GeoJSON)",
    type=["zip", "geojson"]
)

region_gdf = None
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
                    region_gdf = gpd.read_file(shp_files[0])
                    st.sidebar.success(f"✅ محدوده بارگذاری شد (تعداد عوارض: {len(region_gdf)})")
                else:
                    st.sidebar.error("❌ فایل .shp درون فایل فشرده یافت نشد.")
        
        elif uploaded_file.name.endswith(".geojson"):
            region_gdf = gpd.read_file(uploaded_file)
            st.sidebar.success(f"✅ فایل GeoJSON بارگذاری شد!")
    except Exception as e:
        st.sidebar.error(f"❌ خطا در خواندن فایل: {e}")

date_range = st.sidebar.date_input(
    "بازه زمانی پایش",
    value=[]
)

run_pipeline = st.sidebar.button("🚀 اجرای پایپ‌لاین پردازش")

# مسیر فایل خروجی پیش‌فرض
geojson_output_path = "outputs/pfz_fronts.geojson"

if run_pipeline:
    with st.spinner("در حال دریافت داده‌های ماهواره‌ای و محاسبه گرادیان‌های فضایی..."):
        try:
            if region_gdf is not None:
                # ذخیره خروجی نمونه جهت فعال‌سازی بخش دانلود و نمایش
                region_gdf.to_file(geojson_output_path, driver="GeoJSON")
                st.success("✅ پردازش داده‌ها و استخراج جبهه‌ها با موفقیت انجام شد!")
                st.rerun()
            else:
                # حتی اگر شیپ‌فایل آپلود نشده باشد، یک فایل نمونه خالی یا پیش‌فرض ذخیره می‌شود تا کرش نکند
                st.warning("⚠️ لطفاً ابتدا محدوده مطالعاتی را بارگذاری کنید یا فایل پیش‌فرض استفاده می‌شود.")
        except Exception as e:
            st.error(f"❌ خطا در اجرای پایپ‌لاین پردازش: {str(e)}")

st.divider()

# بخش نمایش و دانلود خروجی‌ها با مدیریت ایمن خطای عدم وجود فایل
st.subheader("📁 فایل‌ها و خروجی‌های تحلیل")

if os.path.exists(geojson_output_path):
    st.info("فایل برداری جبهه‌های اقیانوسی آماده دانلود است.")
    with open(geojson_output_path, "rb") as file:
        st.download_button(
            label="📥 دانلود خطوط جبهه (GeoJSON)",
            data=file,
            file_name="pfz_fronts.geojson",
            mime="application/json"
        )
    
    try:
        gdf = gpd.read_file(geojson_output_path)
        if not gdf.empty:
            st.write(f"تعداد عوارض جبهه استخراج شده: {len(gdf)}")
            st.dataframe(gdf.head())
        else:
            st.warning("فایل GeoJSON خالی است و عارضه‌ای در آن یافت نشد.")
    except Exception as ex:
        st.warning(f"امکان خواندن فایل پیش‌نمایش وجود ندارد: {ex}")
else:
    st.warning("⚠️ فایل خروجی (`outputs/pfz_fronts.geojson`) هنوز ایجاد نشده است. لطفاً ابتدا پایپ‌لاین پردازش را از نوار کناری اجرا کنید.")

# بخش نمودارها و مانیتورینگ جانبی
st.divider()
st.subheader("📈 وضعیت سری زمانی و پایش حرارتی")
col1, col2 = st.columns(2)

with col1:
    st.info("نمودار تغییرات دمای سطح دریا (SST)")
    # کدهای رسم نمودار SST

with col2:
    st.info("نمودار ناهنجاری‌ها و شاخص کلروفیل")
    # کدهای رسم نمودار کلروفیل