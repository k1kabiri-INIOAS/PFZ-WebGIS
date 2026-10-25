# ui/analytics.py — آمار جبهه‌ها + دانلود خروجی
import io
import streamlit as st
import pandas as pd


def render_front_stats():
    """نمایش متریک‌ها و جدول خلاصه جبهه‌ها."""
    gdf = st.session_state.get("combined_fronts_gdf")
    if gdf is None or gdf.empty:
        return

    st.markdown("### 📈 خلاصه تحلیلی جبهه‌های صیادی")

    total_segments = len(gdf)
    total_length = float(gdf["Length_km"].sum()) if "Length_km" in gdf.columns else 0.0
    n_regions = gdf["Region"].nunique() if "Region" in gdf.columns else 1

    c1, c2, c3 = st.columns(3)
    c1.metric("تعداد قطعات جبهه", f"{total_segments}")
    c2.metric("مجموع طول جبهه (کیلومتر)", f"{total_length:.1f}")
    c3.metric("تعداد مناطق", f"{n_regions}")

    if "Region" in gdf.columns and "Length_km" in gdf.columns:
        agg_map = {
            "تعداد_قطعه": ("geometry", "count"),
            "مجموع_طول_کیلومتر": ("Length_km", "sum"),
            "میانگین_طول": ("Length_km", "mean"),
        }
        if "Threshold" in gdf.columns:
            agg_map["آستانه"] = ("Threshold", "first")
        summary = (
            gdf.groupby("Region", dropna=False)
            .agg(**agg_map)
            .reset_index()
            .rename(columns={"Region": "منطقه"})
        )
        if "آستانه" in summary.columns:
            summary["آستانه"] = pd.to_numeric(summary["آستانه"], errors="coerce").round(3)
        summary["مجموع_طول_کیلومتر"] = summary["مجموع_طول_کیلومتر"].round(2)
        summary["میانگین_طول"] = summary["میانگین_طول"].round(2)
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_download_section():
    """دکمه‌های دانلود GeoJSON جبهه‌ها و مناطق."""
    fronts = st.session_state.get("combined_fronts_gdf")
    regions = st.session_state.get("combined_region_gdf")

    if fronts is None and regions is None:
        return

    st.markdown("### 📥 دانلود خروجی‌ها")
    col_a, col_b, col_c = st.columns(3)

    if fronts is not None and not fronts.empty:
        try:
            geojson_str = fronts.to_json()
            col_a.download_button(
                label="🚩 دانلود جبهه‌ها (GeoJSON)",
                data=geojson_str,
                file_name="pfz_fronts.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
            # CSV خلاصه
            csv_buf = io.StringIO()
            export_cols = [c for c in ["Region", "Threshold", "Length_km"] if c in fronts.columns]
            if export_cols:
                fronts[export_cols].to_csv(csv_buf, index=False)
                col_b.download_button(
                    label="📊 دانلود جدول جبهه‌ها (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="pfz_fronts_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"خطا در آماده‌سازی دانلود جبهه‌ها: {e}")

    if regions is not None and not regions.empty and st.session_state.get("role") == "admin":
        try:
            geojson_reg = regions.to_json()
            col_c.download_button(
                label="📌 دانلود مرز مناطق (GeoJSON)",
                data=geojson_reg,
                file_name="pfz_regions.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"خطا در آماده‌سازی دانلود مناطق: {e}")
