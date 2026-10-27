# utils/state.py — ذخیره و بازیابی وضعیت تحلیل
import os
import json
import streamlit as st
import geopandas as gpd

from config import STATE_FILE, FRONTS_FILE, REGIONS_FILE
from .logging_helpers import record_error


def init_session_defaults():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
    if "error_logs" not in st.session_state:
        st.session_state.error_logs = []
    if "process_logs" not in st.session_state:
        st.session_state.process_logs = []
    if "current_nc_path" not in st.session_state:
        st.session_state.current_nc_path = None
    if "ssh_nc_path" not in st.session_state:
        st.session_state.ssh_nc_path = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    for key in [
        "nc_out_list",
        "sst_nc_path",
        "chl_nc_path",
        "combined_fronts_gdf",
        "combined_region_gdf",
        "minx",
        "miny",
        "maxx",
        "maxy",
        "latest_date",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None


def save_shared_state():
    if st.session_state.combined_fronts_gdf is not None:
        st.session_state.combined_fronts_gdf.to_file(FRONTS_FILE, driver="GeoJSON")
    if st.session_state.combined_region_gdf is not None:
        st.session_state.combined_region_gdf.to_file(REGIONS_FILE, driver="GeoJSON")

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "latest_date": st.session_state.latest_date,
                "minx": st.session_state.minx,
                "miny": st.session_state.miny,
                "maxx": st.session_state.maxx,
                "maxy": st.session_state.maxy,
                "nc_out_list": st.session_state.nc_out_list,
                "sst_nc_path": st.session_state.sst_nc_path,
                "chl_nc_path": st.session_state.chl_nc_path,
            },
            f,
            ensure_ascii=False,
        )


def load_shared_state():
    if os.path.exists(STATE_FILE) and os.path.exists(REGIONS_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            st.session_state.latest_date = state.get("latest_date")
            st.session_state.minx = state.get("minx")
            st.session_state.miny = state.get("miny")
            st.session_state.maxx = state.get("maxx")
            st.session_state.maxy = state.get("maxy")
            st.session_state.nc_out_list = state.get("nc_out_list")
            st.session_state.sst_nc_path = state.get("sst_nc_path")
            st.session_state.chl_nc_path = state.get("chl_nc_path")

            st.session_state.combined_region_gdf = gpd.read_file(REGIONS_FILE)
            if os.path.exists(FRONTS_FILE):
                st.session_state.combined_fronts_gdf = gpd.read_file(FRONTS_FILE)
            else:
                st.session_state.combined_fronts_gdf = None

            st.session_state.analysis_done = True
        except Exception as e:
            record_error("خطا در بارگذاری آخرین وضعیت نقشه", e)
