import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# 1. تنظیمات اولیه صفحه
# ==========================================
st.set_page_config(
    page_title="سامانه شناساگر مناطق مستعد صید (PFZ)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 سامانه پایش و استخراج جبهه‌های حرارتی اقیانوسی (PFZ)")
st.caption("پردازش داده‌های سنجش از دور دریایی (SST/CHL) و بارگذاری نقشه‌های تعاملی")

# ==========================================
# 2. مدیریت Session State
# ==========================================
if "ds" not in st.session_state:
    st.session_state.ds = None
if "fronts_df" not in st.session_state:
    st.session_state.fronts_df = None

# ==========================================
# 3. تابع اصلی استخراج جبهه‌ها (Fallback/Core)
# ==========================================
def generate_fronts_fallback(ds, var_name="sst", threshold=None):
    """
    استخراج خطوط کانتور جبهه‌های حرارتی با رفع خطاهای Matplotlib 3.8+ و ابعاد مکانی
    """
    try:
        # 3.1. استانداردسازی نام ابعاد مکانی
        rename_dict = {}
        if 'longitude' in ds.dims:
            rename_dict['longitude'] = 'lon'
        if 'latitude' in ds.dims:
            rename_dict['latitude'] = 'lat'
        if rename_dict:
            ds = ds.rename(rename_dict)
            
        # تنظیم ابعاد فضایی برای rioxarray در صورت نیاز
        if hasattr(ds, 'rio'):
            x_name = 'lon' if 'lon' in ds.dims else ('x' if 'x' in ds.dims else None)
            y_name = 'lat' if 'lat' in ds.dims else ('y' if 'y' in ds.dims else None)
            if x_name and y_name:
                ds = ds.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=True)

        # استخراج آرایه دو بعدی
        data_array = ds[var_name]
        if 'time' in data_array.dims:
            data_array = data_array.isel(time=0)
            
        lons = data_array['lon'].values if 'lon' in data_array.dims else data_array['x'].values
        lats = data_array['lat'].values if 'lat' in data_array.dims else data_array['y'].values
        Z = data_array.values

        # 3.2. بررسی سلامت داده‌ها
        z_min = np.nanmin(Z)
        z_max = np.nanmax(Z)
        
        if np.isnan(z_min) or np.isnan(z_max) or z_min == z_max:
            st.warning("⚠️ داده‌های ورودی یکنواخت یا تماماً NaN هستند؛ امکان استخراج جبهه وجود ندارد.")
            return None

        # تعیین آستانه پویا در صورت عدم تعیین توسط کاربر
        if threshold is None or threshold < z_min or threshold > z_max:
            threshold = float(z_min + 0.5 * (z_max - z_min))

        # 3.3. استخراج خطوط کانتور (سازگار با Matplotlib 3.8+)
        X, Y = np.meshgrid(lons, lats)
        
        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, Z, levels=[threshold])
        
        lines = []
        # استفاده از get_paths() به جای cs.collections
        for path in cs.get_paths():
            for polygon in path.to_polygons():
                if len(polygon) > 0:
                    lines.append(polygon)
                    
        plt.close(fig)

        if not lines:
            st.warning(f"⚠️ هیچ خط کانتوری در آستانه {threshold:.2f} پیدا نشد.")
            return None

        # 3.4. تبدیل خطوط به DataFrame
        front_points = []
        for line_idx, line in enumerate(lines):
            for pt in line:
                front_points.append({
                    'line_id': line_idx,
                    'lon': pt[0],
                    'lat': pt[1],
                    'threshold_val': threshold
                })
                
        df_fronts = pd.DataFrame(front_points)

        # 3.5. ذخیره‌سازی ایمن خروجی در پوشه موقت (/tmp)
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, "extracted_fronts.csv")
        df_fronts.to_csv(output_path, index=False)
        
        return df_fronts

    except Exception as e:
        st.error(f"❌ خطا در پردازش جبهه‌ها: {str(e)}")
        return None

# ==========================================
# 4. نوار کناری (Sidebar) - دریافت ورودی‌ها
# ==========================================
st.sidebar.header("⚙️ تنظیمات ورودی")

uploaded_file = st.sidebar.file_uploader(
    "فایل NetCDF را بارگذاری کنید (.nc یا .nc4)", 
    type=["nc", "nc4"]
)

if uploaded_file is not None:
    try:
        # ذخیره موقت فایل آپلود شده برای خواندن توسط xarray
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        st.session_state.ds = xr.open_dataset(tmp_path)
        st.sidebar.success("فایل با موفقیت بارگذاری شد.")
    except Exception as e:
        st.sidebar.error(f"خطا در باز کردن فایل NetCDF: {e}")

# ==========================================
# 5. پنل اصلی پردازش و نمایش
# ==========================================
if st.session_state.ds is not None:
    ds = st.session_state.ds
    
    # انتخاب متغیر
    var_options = list(ds.data_vars.keys())
    selected_var = st.selectbox("متغیر مورد نظر جهت تحلیل را انتخاب کنید:", var_options)
    
    # محاسبه دامنه تغییرات متغیر انتخاب شده
    var_data = ds[selected_var].values
    min_val = float(np.nanmin(var_data))
    max_val = float(np.nanmax(var_data))
    default_thresh = float(min_val + 0.5 * (max_val - min_val))
    
    st.info(f"📊 محدوده متغیر {selected_var}: حداقل = {min_val:.2f} | حداکثر = {max_val:.2f}")

    # انتخاب مقدار آستانه
    threshold_val = st.slider(
        "مقدار آستانه (Threshold) برای کانتورگیری:",
        min_value=min_val,
        max_value=max_val,
        value=default_thresh,
        step=0.1
    )

    # دکمه اجرای استخراج جبهه
    if st.button("🚀 پردازش و استخراج جبهه‌های حرارتی", type="primary"):
        with st.spinner("در حال استخراج جبهه‌ها و تولید خروجی..."):
            df_result = generate_fronts_fallback(ds, var_name=selected_var, threshold=threshold_val)
            st.session_state.fronts_df = df_result

    # نمایش نتائج
    if st.session_state.fronts_df is not None:
        df_fronts = st.session_state.fronts_df
        st.subheader("📌 جبهه‌های استخراج‌شده")
        
        col1, col2 = st.subplots([2, 1])
        
        with col1:
            # نمایش پیش‌نمایش نقشه نقطه به نقطه بر اساس Lat/Lon
            st.map(df_fronts[['lat', 'lon']])
            
        with col2:
            st.write(f"**تعداد نقاط جبهه:** {len(df_fronts)}")
            st.write(f"**تعداد خطوط تفکیک‌شده:** {df_fronts['line_id'].nunique()}")
            
            # ارائه لینک دانلود فایل CSV
            csv_data = df_fronts.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 دانلود نقشه جبهه‌ها (CSV)",
                data=csv_data,
                file_name="pfz_fronts_output.csv",
                mime="text/csv"
            )

else:
    st.info("👋 برای شروع، لطفاً یک فایل NetCDF حاوی داده‌های اقیانوسی را از نوار کناری بارگذاری کنید.")