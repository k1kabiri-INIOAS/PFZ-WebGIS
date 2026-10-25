# ui/styles.py
import os
import base64
import streamlit as st

from config import LOGO_FILENAME


def apply_theme(is_dark: bool):
    dark_css = """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
        .stApp, [data-testid="stSidebar"], .stMarkdown {
            background-color: #121212 !important;
            color: #E0E0E0 !important;
            direction: rtl;
            text-align: right;
        }
        p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input {
            font-family: 'Vazirmatn', sans-serif;
        }
        .main-title {
            font-size: 1.75rem !important;
            color: #64B5F6 !important;
            font-weight: bold;
            margin-bottom: 0.75rem;
            text-align: right !important;
        }
        .inioas-logo-topleft {
            position: fixed; top: 12px; left: 20px; z-index: 999999;
            width: 90px; border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.9); padding: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        }
        /* موبایل */
        @media (max-width: 768px) {
            .main-title { font-size: 1.25rem !important; }
            .inioas-logo-topleft { width: 60px; top: 8px; left: 8px; }
            [data-testid="stSidebar"] { min-width: 85vw !important; }
        }
        /* بهبود دکمه‌ها و پیام‌ها */
        .stAlert { direction: rtl; text-align: right; }
        div[data-testid="stMetricValue"] { direction: ltr; }
        </style>
    """
    light_css = """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
        .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
        p, h1, h2, h3, h4, h5, h6, span, div, label, li, button, input {
            font-family: 'Vazirmatn', sans-serif;
        }
        .main-title {
            font-size: 1.75rem !important;
            color: #1E3A8A !important;
            font-weight: bold;
            margin-bottom: 0.75rem;
            text-align: right !important;
        }
        .stMarkdown, .stSelectbox, .stSlider { text-align: right; }
        .inioas-logo-topleft {
            position: fixed; top: 12px; left: 20px; z-index: 999999;
            width: 90px; border-radius: 8px;
            background-color: #FFFFFF; padding: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        @media (max-width: 768px) {
            .main-title { font-size: 1.25rem !important; }
            .inioas-logo-topleft { width: 60px; top: 8px; left: 8px; }
            [data-testid="stSidebar"] { min-width: 85vw !important; }
        }
        .stAlert { direction: rtl; text-align: right; }
        div[data-testid="stMetricValue"] { direction: ltr; }
        </style>
    """
    st.markdown(dark_css if is_dark else light_css, unsafe_allow_html=True)


def show_logo():
    if not os.path.exists(LOGO_FILENAME):
        return
    try:
        with open(LOGO_FILENAME, "rb") as logo_file:
            encoded_logo = base64.b64encode(logo_file.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/jpeg;base64,{encoded_logo}" class="inioas-logo-topleft" alt="INIOAS Logo">',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def hide_deploy_for_users():
    if st.session_state.get("logged_in") and st.session_state.get("role") != "admin":
        st.markdown(
            """
            <style>
            .stDeployButton {display: none !important;}
            [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
            header [data-testid="baseButton-header"] {display: none !important;}
            </style>
            """,
            unsafe_allow_html=True,
        )
