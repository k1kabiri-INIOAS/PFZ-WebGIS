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


def _normalize_weights(weights: dict):
    """weights: {name: raw_value} → normalized dict summing to 1."""
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def render_admin_controls():
    """
    تنظیمات مناطق، عوامل مدل و وزن‌ها.
    برمی‌گرداند: (region_configs, model_options)
    """
    region_configs = {}
    model_options = {
        "use_sst": True,
        "use_chl": True,
        "use_current": False,
        "use_ssh": False,
        "sst_weight": 0.40,
        "chl_weight": 0.30,
        "current_weight": 0.20,
        "ssh_weight": 0.10,
    }

    if st.session_state.role != "admin":
        return region_configs, model_options

    st.sidebar.header("⚙️ تنظیمات پردازش و مدل")

    # ---- عوامل مدل (سراسری) ----
    st.sidebar.subheader("🧬 عوامل مدل PFZ")
    st.sidebar.caption(
        "وزن‌ها مستقل‌اند و فقط روی عوامل فعال نرمال می‌شوند (جمع = ۱)."
    )

    use_sst = st.sidebar.checkbox("☑ SST (گرادیان حرارتی)", value=True, key="use_sst")
    use_chl = st.sidebar.checkbox("☑ Chlorophyll-a", value=True, key="use_chl")
    use_current = st.sidebar.checkbox(
        "☐ جریان سطحی ژئوستروفیک (Currents)",
        value=False,
        key="use_current",
        help="سرعت جریان از ارتفاع‌سنجی ماهواره‌ای — در صورت شکست دانلود، تحلیل با بقیه عوامل ادامه می‌یابد.",
    )
    use_ssh = st.sidebar.checkbox(
        "☐ گرادیان SSH / SLA",
        value=False,
        key="use_ssh",
        help="گرادیان ارتفاع سطح دریا — مرتبط با جبهه و ادی. در صورت شکست دانلود، نادیده گرفته می‌شود.",
    )

    # حداقل یک عامل پایه لازم است
    if not use_sst and not use_chl:
        st.sidebar.warning("حداقل یکی از SST یا Chlorophyll باید فعال باشد.")
        use_sst = True

    sst_w = st.sidebar.slider(
        "وزن SST",
        0.0,
        1.0,
        0.40,
        0.05,
        key="w_sst_global",
        disabled=not use_sst,
    )
    chl_w = st.sidebar.slider(
        "وزن Chlorophyll-a",
        0.0,
        1.0,
        0.30,
        0.05,
        key="w_chl_global",
        disabled=not use_chl,
    )
    cur_w = st.sidebar.slider(
        "وزن جریان (Current speed)",
        0.0,
        1.0,
        0.20,
        0.05,
        key="w_cur_global",
        disabled=not use_current,
    )
    ssh_w = st.sidebar.slider(
        "وزن گرادیان SSH",
        0.0,
        1.0,
        0.10,
        0.05,
        key="w_ssh_global",
        disabled=not use_ssh,
    )

    active_w = {}
    if use_sst:
        active_w["SST"] = sst_w
    if use_chl:
        active_w["Chl"] = chl_w
    if use_current:
        active_w["Current"] = cur_w
    if use_ssh:
        active_w["SSH"] = ssh_w

    norm_w = _normalize_weights(active_w)
    norm_txt = " · ".join([f"{k}={v:.0%}" for k, v in norm_w.items()])
    st.sidebar.info(f"وزن نرمال‌شده: {norm_txt}")

    model_options = {
        "use_sst": use_sst,
        "use_chl": use_chl,
        "use_current": use_current,
        "use_ssh": use_ssh,
        "sst_weight": float(sst_w) if use_sst else 0.0,
        "chl_weight": float(chl_w) if use_chl else 0.0,
        "current_weight": float(cur_w) if use_current else 0.0,
        "ssh_weight": float(ssh_w) if use_ssh else 0.0,
    }

    st.sidebar.markdown("---")
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
                st.sidebar.subheader("📌 آستانه حساسیت هر منطقه")
                for shp_path in sorted(shp_files):
                    reg_name = (
                        os.path.splitext(os.path.basename(shp_path))[0]
                        .replace("_", " ")
                        .title()
                    )
                    def_thresh = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get(
                        "thresh", 0.50
                    )
                    with st.sidebar.expander(f"منطقه: {reg_name}", expanded=False):
                        thresh = st.slider(
                            f"آستانه جبهه ({reg_name})",
                            0.1,
                            1.0,
                            def_thresh,
                            0.05,
                            key=f"thresh_{reg_name}",
                        )
                        region_configs = region_configs if False else {}  # noqa — set below
                    # re-get in loop properly
                region_configs = {}
                for shp_path in sorted(shp_files):
                    reg_name = (
                        os.path.splitext(os.path.basename(shp_path))[0]
                        .replace("_", " ")
                        .title()
                    )
                    def_thresh = DEFAULT_REGION_DEFAULTS.get(reg_name, {}).get(
                        "thresh", 0.50
                    )
                    thresh = st.session_state.get(f"thresh_{reg_name}", def_thresh)
                    # ensure widget exists
                    if f"thresh_{reg_name}" not in st.session_state:
                        with st.sidebar.expander(f"منطقه: {reg_name}", expanded=False):
                            thresh = st.slider(
                                f"آستانه جبهه ({reg_name})",
                                0.1,
                                1.0,
                                def_thresh,
                                0.05,
                                key=f"thresh_{reg_name}",
                            )
                    else:
                        # widget already created in first loop attempt — recreate cleanly
                        pass
                    region_configs[reg_name] = {
                        "shp_path": shp_path,
                        "sst_weight": model_options_weight(model_options := model_options if False else None),
                        "threshold": st.session_state.get(f"thresh_{reg_name}", def_thresh),
                    }
        except Exception as ex:
            record_error("خطا در استخراج شیپ‌فایل", ex)
            st.sidebar.error("خطا در خواندن فایل shapefile. لطفاً فایل معتبر آپلود کنید.")

    return region_configs, model_options


def model_options_weight(_):
    return 0.0


def _normalize_weights(weights: dict):
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


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
