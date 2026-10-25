# ui/login.py
import streamlit as st
from auth import authenticate_user, login_or_register_email_user, log_user_activity


def render_login_page():
    st.markdown(
        '<div class="main-title">🌊 ورود به سامانه هوشمند مناطق مستعد صیادی (PFZ)</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔐 ورود به سیستم", "📧 ورود با ایمیل و مشخصات"])

        with tab1:
            with st.form("login_form"):
                login_user = st.text_input("👤 نام کاربری")
                login_pass = st.text_input("🔑 رمز عبور", type="password")
                submit_login = st.form_submit_button(
                    "ورود به سامانه", use_container_width=True
                )

                if submit_login:
                    if not login_user.strip() or not login_pass:
                        st.error("لطفاً نام کاربری و رمز عبور را وارد کنید.")
                    else:
                        role = authenticate_user(login_user.strip(), login_pass)
                        if role:
                            st.session_state.logged_in = True
                            st.session_state.username = login_user.strip()
                            st.session_state.role = role
                            log_user_activity(login_user.strip(), "LOGIN")
                            st.success(f"خوش آمدید {login_user.strip()}!")
                            st.rerun()
                        else:
                            st.error("نام کاربری یا رمز عبور اشتباه است.")

        with tab2:
            st.markdown("### ورود سریع با ایمیل و اطلاعات شخصی")
            with st.form("email_login_form"):
                email_input = st.text_input("📧 آدرس ایمیل (Email Address)")
                first_name_input = st.text_input("👤 نام *")
                last_name_input = st.text_input("👤 نام خانوادگی *")
                phone_input = st.text_input("📞 شماره تلفن (اختیاری)")
                org_input = st.text_input("🏢 نام سازمان (اختیاری)")
                submit_email_login = st.form_submit_button(
                    "تایید و ورود به سامانه", use_container_width=True
                )

                if submit_email_login:
                    if not email_input or "@" not in email_input:
                        st.error("لطفا یک آدرس ایمیل معتبر وارد کنید.")
                    elif not first_name_input.strip() or not last_name_input.strip():
                        st.error("لطفا فیلدهای اجباری (نام و نام خانوادگی) را پر کنید.")
                    else:
                        role = login_or_register_email_user(
                            email_input.strip(),
                            first_name_input.strip(),
                            last_name_input.strip(),
                            phone_input.strip(),
                            org_input.strip(),
                        )
                        st.session_state.logged_in = True
                        st.session_state.username = email_input.strip()
                        st.session_state.role = role
                        log_user_activity(
                            email_input.strip(),
                            f"LOGIN (Email: {first_name_input} {last_name_input} - Org: {org_input})",
                        )
                        st.success(f"ورود موفق با ایمیل: {email_input.strip()}")
                        st.rerun()
    st.stop()
