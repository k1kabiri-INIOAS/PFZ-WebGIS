# utils/logging_helpers.py
import logging
import sys
import streamlit as st

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def record_error(msg, exc=None):
    full_msg = f"{msg}"
    if exc is not None:
        full_msg = f"{msg}: {exc}"
    logging.warning(f"[PFZ-LOG] {full_msg}")
    if "error_logs" in st.session_state:
        st.session_state.error_logs.append(full_msg)


def log_process(msg_type, msg_text, status_obj=None):
    if "process_logs" in st.session_state:
        st.session_state.process_logs.append((msg_type, msg_text))
    if status_obj:
        status_obj.write(msg_text)
