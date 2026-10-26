# app.py — سامانه مدیریت PFZ (نسخه بازآرایی‌شده + مدل چندعاملی)
# ساختار ماژولار: auth / map / modules / ui / utils

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import warnings
import pandas as pd
import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium

from config import OUTPUT_DIR, MAP_HEIGHT_DESKTOP
from auth import init_db
from utils import (
    init_session_defaults,
    load_shared_state,
    save_shared_state,
    record_error,
    log_process,
)
from modules.fetcher import fetch_near_realtime_data
from modules.processor import process_pfz_pipeline
from modules.fronts import generate_fronts_fallback
from map import build_pfz_map
from ui import (
    apply_theme,
    show_logo,
    hide_deploy_for_users,
    render_login_page,
    render_sidebar_user,
    render_admin_controls,
    render_admin_panel,
    render_front_stats,
    render_download_section,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="سامانه مدیریت PFZ 🐟",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- مقداردهی اولیه ---
init_db()
init_session_defaults()

# --- تم و لوگو ---
st.sidebar.markdown("---")
map_theme = st.sidebar.radio(
    "🎨 تم صفحه (App Theme)", ["☀️ روز (Light)", "🌙 شب (Dark)"], index=0
)
is_dark = "شب" in map_theme
apply_theme(is_dark)
show_logo()
hide_deploy_for_users()

# --- صفحه ورود ---
if not st.session_state.logged_in:
    render_login_page()

# --- بدنه اصلی ---
load_shared_state()
render_sidebar_user()
region_configs, model_options = render_admin_controls()

st.markdown(
    '<div class="main-title">🌊 سامانه هوشمند تشخیص مناطق مستعد صیادی (نسخه آزمایشی) (PFZ, Beta version) 🐟</div>',
    unsafe_allow_html=True,
)

render_admin_panel()

# --- دکمه اجرای تحلیل (فقط ادمین) ---
if st.session_state.role == "admin":
    if st.sidebar.button(
        "🚀 دریافت داده‌های به‌روز و اجرای تحلیل", use_container_width=True
    ):
        if not region_configs:
            st.error(
                "تنظیمات مناطق بارگذاری نشده است. لطفاً shapefile پیش‌فرض یا فایل آپلود‌شده را بررسی کنید."
            )
        else:
            st.session_state.process_logs = []
            # فقط پوشهٔ خروجی هر منطقه پاک می‌شود (نتیجهٔ قبلی روی صفحه می‌ماند تا اجرای جدید تمام شود)
            # shared_state و login دست نخورده می‌مانند تا کرش/logout نشود
            import shutil
            for _reg in list(region_configs.keys()):
                _d = os.path.join(OUTPUT_DIR, _reg.replace(" ", "_"))
                if os.path.isdir(_d):
                    try:
                        shutil.rmtree(_d, ignore_errors=True)
                    except Exception:
                        pass

            with st.status(
                "🚀 شروع فرآیند پردازش داده‌های مکانی چندمنطقه‌ای...", expanded=True
            ) as status:
                try:
                    log_process("info", "در حال خواندن شیپ‌فایل‌های منطقه‌ای...", status)
                    all_gdfs = []
                    for reg_name, cfg in region_configs.items():
                        temp_gdf = gpd.read_file(cfg["shp_path"]).to_crs("EPSG:4326")
                        temp_gdf["Region"] = reg_name
                        all_gdfs.append(temp_gdf)

                    combined_region_gdf = gpd.GeoDataFrame(
                        pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326"
                    )
                    minx, miny, maxx, maxy = combined_region_gdf.total_bounds

                    factors_txt = ["SST", "Chl"]
                    if model_options.get("use_current", False):
                        factors_txt.append("Current")
                    if model_options.get("use_ssh", False):
                        factors_txt.append("SSH")
                    log_process(
                        "info",
                        f"در حال دریافت داده از ERDDAP: {', '.join(factors_txt)} ...",
                        status,
                    )

                    fetch_result = fetch_near_realtime_data(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        OUTPUT_DIR,
                        fetch_current=model_options.get("use_current", False),
                        fetch_ssh=model_options.get("use_ssh", False),
                    )

                    sst_nc_path = fetch_result["sst_nc_path"]
                    chl_nc_path = fetch_result["chl_nc_path"]
                    current_nc_path = fetch_result.get("current_nc_path")
                    ssh_nc_path = fetch_result.get("ssh_nc_path")
                    latest_date = fetch_result["latest_date"]
                    fetched = fetch_result.get("fetched", {})

                    # گزارش شفاف وضعیت هر لایه
                    if fetched.get("sst"):
                        log_process("success", "✓ SST دانلود شد.", status)
                    else:
                        log_process("error", "✗ SST از سرور نیامد (fallback مصنوعی).", status)

                    if fetched.get("chl"):
                        log_process("success", "✓ Chlorophyll دانلود شد.", status)
                    else:
                        log_process("error", "✗ Chlorophyll از سرور نیامد (fallback مصنوعی).", status)

                    if model_options.get("use_current"):
                        if fetched.get("current") and current_nc_path:
                            log_process("success", "✓ جریان (Currents) دانلود شد.", status)
                        else:
                            log_process(
                                "error",
                                "✗ دانلود جریان ناموفق بود — ادامه بدون Current.",
                                status,
                            )
                    if model_options.get("use_ssh"):
                        if fetched.get("ssh") and ssh_nc_path:
                            log_process("success", "✓ SSH / SLA دانلود شد.", status)
                        else:
                            log_process(
                                "error",
                                "✗ دانلود SSH ناموفق بود — ادامه بدون SSH.",
                                status,
                            )

                    if sst_nc_path and chl_nc_path:
                        used = ["SST", "Chl"]
                        if model_options.get("use_current") and fetched.get("current") and current_nc_path:
                            used.append("Current")
                        if model_options.get("use_ssh") and fetched.get("ssh") and ssh_nc_path:
                            used.append("SSH")
                        log_process(
                            "success",
                            f"عوامل فعال در این اجرا: {' + '.join(used)}",
                            status,
                        )
                        all_front_gdfs, nc_out_list = [], []

                        for reg_name, cfg in region_configs.items():
                            _use_cur = bool(model_options.get("use_current")) and bool(current_nc_path)
                            _use_ssh = bool(model_options.get("use_ssh")) and bool(ssh_nc_path)
                            _extra_txt = []
                            if _use_cur:
                                _extra_txt.append("Current")
                            if _use_ssh:
                                _extra_txt.append("SSH")
                            _extra_msg = (" + " + " + ".join(_extra_txt)) if _extra_txt else " (فقط SST+Chl)"
                            log_process(
                                "info",
                                f"در حال پردازش **{reg_name}**{_extra_msg}...",
                                status,
                            )
                            try:
                                # وزن‌های منطقه‌ای (مثل قبل) + سهم اختیاری جریان/SSH
                                _sst_w = float(cfg["sst_weight"])
                                _chl_w = float(cfg["chl_weight"])
                                _cur_w = float(model_options.get("current_weight", 0.0)) if _use_cur else 0.0
                                _ssh_w = float(model_options.get("ssh_weight", 0.0)) if _use_ssh else 0.0
                                # اگر عوامل اختیاری روشن باشند، سهم‌شان از SST/Chl کم می‌شود
                                _extra = _cur_w + _ssh_w
                                if _extra > 0 and _extra < 1.0:
                                    _scale = 1.0 - _extra
                                    _sst_w *= _scale
                                    _chl_w *= _scale
                                nc_out, _, _ = process_pfz_pipeline(
                                    shapefile_path=cfg["shp_path"],
                                    sst_nc_path=sst_nc_path,
                                    chl_nc_path=chl_nc_path,
                                    output_dir=os.path.join(
                                        OUTPUT_DIR, reg_name.replace(" ", "_")
                                    ),
                                    sst_weight=_sst_w,
                                    chl_weight=_chl_w,
                                    current_nc_path=current_nc_path,
                                    current_weight=_cur_w,
                                    ssh_nc_path=ssh_nc_path,
                                    ssh_weight=_ssh_w,
                                    use_sst=True,
                                    use_chl=True,
                                    use_current=_use_cur,
                                    use_ssh=_use_ssh,
                                )
                            except Exception as proc_ex:
                                record_error(f"خطا در مدل منطقه {reg_name}", proc_ex)
                                nc_out = None

                            if nc_out and os.path.exists(nc_out):
                                nc_out_list.append(
                                    (reg_name, nc_out, cfg["shp_path"])
                                )
                                reg_fronts_gdf = generate_fronts_fallback(
                                    nc_out,
                                    cfg["threshold"],
                                    reg_name,
                                    sst_nc_path=sst_nc_path,
                                    region_shp_path=cfg["shp_path"],
                                )
                                if reg_fronts_gdf is not None:
                                    all_front_gdfs.append(reg_fronts_gdf)
                                    log_process(
                                        "success",
                                        f"جبهه‌های منطقه {reg_name} استخراج گردید.",
                                        status,
                                    )

                        st.session_state.combined_fronts_gdf = (
                            pd.concat(all_front_gdfs, ignore_index=True)
                            if all_front_gdfs
                            else None
                        )
                        st.session_state.combined_region_gdf = combined_region_gdf
                        st.session_state.nc_out_list = nc_out_list
                        st.session_state.sst_nc_path = sst_nc_path
                        st.session_state.chl_nc_path = chl_nc_path
                        st.session_state.current_nc_path = current_nc_path
                        st.session_state.ssh_nc_path = ssh_nc_path
                        st.session_state.latest_date = latest_date
                        (
                            st.session_state.minx,
                            st.session_state.miny,
                            st.session_state.maxx,
                            st.session_state.maxy,
                        ) = (minx, miny, maxx, maxy)
                        st.session_state.analysis_done = True
                        st.session_state.model_options_used = model_options

                        save_shared_state()
                        status.update(
                            label="تمام مراحل پردازش با موفقیت به پایان رسید!",
                            state="complete",
                        )
                        st.balloons()
                    else:
                        log_process(
                            "error",
                            "فایل‌های SST یا CHL دریافت نشدند.",
                            status,
                        )
                        status.update(label="پردازش متوقف شد", state="error")

                except Exception as global_ex:
                    record_error("خطای کلی در جریان اجرای برنامه", global_ex)
                    status.update(
                        label="اجرای برنامه با خطا متوقف شد",
                        state="error",
                    )
                    st.error(
                        "خطای غیرمنتظره رخ داد. جزئیات زیر را ببینید (نشست شما قطع نمی‌شود)."
                    )
                    st.exception(global_ex)
                    # هرگز logged_in را تغییر نده

    if st.session_state.process_logs:
        with st.expander("📝 گزارش مختصر مراحل پردازش", expanded=False):
            for msg_type, text in st.session_state.process_logs:
                if msg_type == "success":
                    st.success(text)
                elif msg_type == "error":
                    st.error(text)
                else:
                    st.info(text)

# --- نقشه + آمار + دانلود ---
if st.session_state.analysis_done and st.session_state.combined_region_gdf is not None:
    st.markdown(
        '<h3 style="text-align: right; color: #1E3A8A; font-weight: bold; margin-top: 1rem;">🗺️ نقشه تعاملی خطوط جبهه و لایه‌های پایه</h3>',
        unsafe_allow_html=True,
    )

    render_front_stats()
    render_download_section()

    try:
        m = build_pfz_map(is_dark=is_dark)
        if m is not None:
            st_folium(
                m,
                width="100%",
                height=MAP_HEIGHT_DESKTOP,
                returned_objects=[],
                use_container_width=True,
            )
        else:
            st.warning("نقشه قابل ساخت نبود. وضعیت تحلیل را بررسی کنید.")
    except Exception as map_render_err:
        st.error("⚠️ خطا در پردازش و رندر نقشه:")
        st.exception(map_render_err)
        if st.session_state.error_logs:
            with st.expander("جزئیات خطاهای ثبت‌شده"):
                for err in st.session_state.error_logs[-10:]:
                    st.code(err)
else:
    if st.session_state.role != "admin":
        st.warning(
            "⚠️ هنوز هیچ دیتایی توسط مدیر سیستم پردازش و ذخیره نشده است. پس از اجرای تحلیل توسط ادمین، نقشه در دسترس خواهد بود."
        )
    else:
        st.info(
            "👈 برای شروع، از منوی تنظیمات کناری پارامترها را تنظیم کرده و دکمه **دریافت داده‌های به‌روز و اجرای تحلیل** را بزنید."
        )

st.markdown("---")
st.caption(
    "سامانه PFZ · INIOAS · مدل چندعاملی (SST / Chl / Current / SSH) · NOAA ERDDAP"
)
