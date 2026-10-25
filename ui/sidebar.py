# ui/sidebar.py
import os
import zipfile
import streamlit as st

from config import DEFAULT_SHAPES_PATH, DEFAULT_REGION_DEFAULTS, OUTPUT_DIR
from auth import log_user_activity, get_user_logs, get_all_users
from utils.logging_helpers import record_error


def render_sidebar_user():
    st.sidebar.markdown(f"### 👤 سلام **{st.session_state.username}**")
    role_label = (
        "مدیر سیستم (Admin)"
        if st.session_state.role == "admin"
        else "کاربر عادی (User)"
    )
    st.sidebar.caption(f"🛡️ سطح دسترسی: **{role_label}**")
    if st.sidebar.button("🚪 خروج (Logout)", use_container_width=True):
        log_user_activity(st.session_state.username, "LOGOUT")
        st.session_state.logged_in = False
        st.rerun()
    st.sidebar.markdown("---")


def render_admin_controls():
    """تنظیمات مناطق و آپلود shapefile. برمی‌گرداند region_configs dict."""
    region_configs = {}
    if st.session_state.role != "admin":
        return region_configs

    st.sidebar.header("⚙️ تنظیمات پردازش و مدل")
    uploaded_shapefile_zip = st.sidebar.file_uploader(
        "آپلود فایل شیپ‌فایل مناطق (.zip) - اختیاری", type="zip"
    )

    extract_path = os.path.join(OUTPUT_DIR, "extracted_shapes")
    os.makedirs(extract_path, exist_ok=True)

    zip_to_extract = None
    if uploaded_shapefile_zip:
        zip_to_extract = uploaded_shapefile_zip
    elif os.path.exists(DEFAULT_SHAPES_PATH):
        zip_to_extract = DEFAULT_SHAPES_PATH

    if zip_to_extract is not None:
        try:
            with zipfile.ZipFile(zip_to_extract, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            shp_files = [
                os.path.join(r, f)
                for r, d, files in os.walk(extract_path)
                for f in files
                if f.endswith(".shp") and not f.startswith("._")
            ]

            if shp_files:
                st.sidebar.subheader("📌 تنظیمات اختصاصی هر منطقه")
                for shp_path in sorted(shp_files):
                    reg_name = (
                        os.path.splitext(os.path.basename(shp_path))[0]
                        .replace("_", " ")
                        .title()
                    )
                    def_sst = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get(
                        "sst_w", 0.60
                    )
                    def_thresh = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get(
                        "thresh", 0.50
                    )

                    with st.sidebar.expander(f"منطقه: {reg_name}", expanded=True):
                        sst_w = st.slider(
                            f"وزن SST ({reg_name})",
                            0.0,
                            1.0,
                            def_sst,
                            0.05,
                            key=f"sst_{reg_name}",
                        )
                        chl_w = round(1.0 - sst_w, 2)
                        st.caption(f"وزن کلروفیل-آ: **{chl_w}**")
                        thresh = st.slider(
                            f"آستانه حساسیت ({reg_name})",
                            0.1,
                            1.0,
                            def_thresh,
                            0.05,
                            key=f"thresh_{reg_name}",
                        )
                        region_configs[reg_name] = {
                            "shp_path": shp_path,
                            "sst_weight": sst_w,
                            "chl_weight": chl_w,
                            "threshold": thresh,
                        }
        except Exception as ex:
            record_error("خطا در استخراج شیپ‌فایل", ex)
            st.sidebar.error("خطا در خواندن فایل shapefile. لطفاً فایل معتبر آپلود کنید.")

    return region_configs


def render_admin_panel():
    if st.session_state.role != "admin":
        return
    with st.expander("📊 گزارش ورود و خروج کاربران و اطلاعات ثبت‌نامی", expanded=False):
        tab_log1, tab_log2 = st.tabs(
            ["📝 لاگ فعالیت‌ها", "👥 لیست کاربران ثبت‌نام‌شده"]
        )
        with tab_log1:
            logs_df = get_user_logs(50)
            if not logs_df.empty:
                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("هنوز گزارشی ثبت نشده است.")
        with tab_log2:
            users_df = get_all_users()
            if not users_df.empty:
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("کاربری ثبت نشده است.")
