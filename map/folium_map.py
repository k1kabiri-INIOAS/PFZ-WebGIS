# map/folium_map.py — ساخت نقشه Folium و کنترل‌های سفارشی
import os
import folium
from folium.plugins import Fullscreen
from branca.element import MacroElement
from jinja2 import Template
import geopandas as gpd
import xarray as xr
import streamlit as st

from config import OFFSET_LON_DEG, OFFSET_LAT_DEG, ISLANDS_ZIP, OUTPUT_DIR
from modules.fronts import mask_pfz_by_fronts, mask_land_from_pfz
from modules.rendering import (
    apply_visual_offset_to_bounds,
    image_to_base64,
    render_pixel_perfect_heatmap,
    load_and_crop_dataset,
)
from utils.dates import parse_date_formats
from utils.logging_helpers import record_error


class CustomMapFeatures(MacroElement):
    _template = Template(
        """
    {% macro script(this, kwargs) %}
    var map = {{ this._parent.get_name() }};
    let useDDM = false;
    let lastLatLng = null;
    let currentMarker = null;

    function toDDM(deg, isLat) {
      const absolute = Math.abs(deg);
      const degrees = Math.floor(absolute);
      const decimalMinutes = ((absolute - degrees) * 60).toFixed(3);
      const direction = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
      return `${degrees}° ${decimalMinutes}' ${direction}`;
    }

    function updateCoordDisplay(latlng) {
      const displayElement = document.getElementById('coord-text');
      if (!displayElement || !latlng) return;
      if (useDDM) {
        displayElement.innerHTML = `${toDDM(latlng.lat, true)} | ${toDDM(latlng.lng, false)}`;
      } else {
        displayElement.innerHTML = `Lat: ${latlng.lat.toFixed(5)} | Lng: ${latlng.lng.toFixed(5)}`;
      }
    }

    const coordControl = L.control({ position: 'bottomright' });
    coordControl.onAdd = function (map) {
      const div = L.DomUtil.create('div', 'coord-box');
      div.style.padding = '8px 12px';
      div.style.background = 'rgba(255, 255, 255, 0.95)';
      div.style.border = '2px solid #2B5B84';
      div.style.borderRadius = '8px';
      div.style.fontSize = '13px';
      div.style.direction = 'ltr';
      div.style.fontFamily = 'monospace';
      div.style.display = 'flex';
      div.style.alignItems = 'center';
      div.style.gap = '10px';
      div.style.boxShadow = '0 2px 6px rgba(0,0,0,0.3)';
      div.style.zIndex = '1000';

      div.innerHTML = `
        <span id="coord-text" style="font-weight: bold; color: #333;">Lat: -- | Lng: --</span>
        <button id="toggle-coord-btn" title="تغییر فرمت نمایش" style="
          cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: bold;
          border: 1px solid #2B5B84; background: #e8f4fc; border-radius: 4px; color: #1E3A8A;">DDM</button>
      `;
      L.DomEvent.disableClickPropagation(div);
      setTimeout(function() {
        const btn = document.getElementById('toggle-coord-btn');
        if (btn) {
          btn.addEventListener('click', function() {
            useDDM = !useDDM;
            btn.innerText = useDDM ? 'DEC' : 'DDM';
            if (lastLatLng) updateCoordDisplay(lastLatLng);
          });
        }
      }, 100);
      return div;
    };
    coordControl.addTo(map);

    map.on('mousemove', function (e) {
      lastLatLng = e.latlng;
      updateCoordDisplay(e.latlng);
    });

    map.on('click', function (e) {
      const latlng = e.latlng;
      if (currentMarker) {
        currentMarker.setLatLng(latlng);
      } else {
        currentMarker = L.marker(latlng).addTo(map);
      }

      const latDDM = toDDM(latlng.lat, true);
      const lngDDM = toDDM(latlng.lng, false);
      const copyText = latDDM + '  |  ' + lngDDM;
      const latFixed = latlng.lat.toFixed(5);
      const lngFixed = latlng.lng.toFixed(5);

      const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${latFixed},${lngFixed}`;
      const openSeaMapUrl = `https://map.openseamap.org/?zoom=13&lat=${latFixed}&lon=${lngFixed}`;
      const windyUrl = `https://www.windy.com/?${latFixed},${lngFixed},11`;

      const popupHtml = `
        <div style="direction:ltr; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; color:#1E3A8A; min-width:230px; padding: 2px;">
          <div style="margin-bottom:6px;">${latDDM}<br>${lngDDM}</div>
          <input type="text" id="coord-input-box" value="${copyText}" readonly style="width: 100%; text-align: center; font-family: monospace; font-size: 11px; padding: 4px; margin-bottom: 6px; border: 1px solid #007bff; border-radius: 4px; background: #f0f4f8; color: #333;" />
          <button id="popup-copy-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #007bff; color: white; border-radius: 4px; width: 100%; font-weight:bold; margin-bottom: 5px;">📋 کپی کُد مختصات (Copy)</button>
          <button id="popup-share-btn" style="cursor: pointer; padding: 5px 8px; font-size: 11px; border: none; background: #6c757d; color: white; border-radius: 4px; font-weight: bold; text-align: center; width: 100%; margin-bottom: 5px;">🔀 اشتراک‌گذاری مختصات (Share With)</button>
          <a href="${gmapsUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #4285F4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">📍 باز کردن در گوگل مپ</a>
          <a href="${openSeaMapUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #007791; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 4px;">🌐 باز کردن در OpenSeaMap</a>
          <a href="${windyUrl}" target="_blank" style="display: block; padding: 5px 8px; font-size: 11px; background: #1B65B4; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center;">🌊 باز کردن در Windy</a>
        </div>
      `;
      currentMarker.bindPopup(popupHtml).openPopup();
      lastLatLng = latlng;
      updateCoordDisplay(latlng);

      setTimeout(() => {
        const copyBtn = document.getElementById('popup-copy-btn');
        const shareBtn = document.getElementById('popup-share-btn');
        const inputBox = document.getElementById('coord-input-box');
        if (copyBtn && inputBox) {
          L.DomEvent.disableClickPropagation(copyBtn);
          L.DomEvent.disableClickPropagation(inputBox);
          copyBtn.addEventListener('click', function() {
            inputBox.select();
            try {
              document.execCommand('copy');
              copyBtn.innerText = 'کپی شد! (Copied)';
              copyBtn.style.background = '#17a2b8';
              setTimeout(() => { copyBtn.innerText = '📋 کپی کُد مختصات (Copy)'; copyBtn.style.background = '#007bff'; }, 2000);
            } catch (err) {
              copyBtn.innerText = 'متن انتخاب شد (Ctrl+C)';
            }
          });
        }
        if (shareBtn) {
          L.DomEvent.disableClickPropagation(shareBtn);
          shareBtn.addEventListener('click', () => {
            if (navigator.share) {
              navigator.share({ title: 'مختصات نقطه صیادی PFZ', text: copyText, url: gmapsUrl }).catch(() => {});
            } else {
              window.open(gmapsUrl, '_blank');
            }
          });
        }
      }, 150);
    });
    {% endmacro %}
    """
    )

    def __init__(self):
        super().__init__()


def build_pfz_map(is_dark=False):
    """ساخت نقشه کامل PFZ بر اساس session_state. برمی‌گرداند folium.Map یا None."""
    if not st.session_state.analysis_done or st.session_state.combined_region_gdf is None:
        return None

    user_is_admin = st.session_state.get("role") == "admin"
    miny, maxy = st.session_state.miny, st.session_state.maxy
    minx, maxx = st.session_state.minx, st.session_state.maxx

    m = folium.Map(
        location=[
            (miny + maxy) / 2 + OFFSET_LAT_DEG,
            (minx + maxx) / 2 + OFFSET_LON_DEG,
        ],
        zoom_start=6,
        tiles=None,
    )

    folium.TileLayer("OpenStreetMap", name="نقشه خیابانی (OSM)", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="تصاویر ماهواره‌ای گوگل (Google Satellite)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="نقشه ترکیبی گوگل (Google Hybrid)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    CustomMapFeatures().add_to(m)
    Fullscreen(
        position="topright",
        title="حالت تمام‌صفحه (Fullscreen)",
        title_cancel="خروج از حالت تمام‌صفحه",
        force_separate_button=True,
    ).add_to(m)

    # لایه‌های رستری PFZ
    if st.session_state.nc_out_list:
        land_gdf = None
        try:
            if os.path.exists(ISLANDS_ZIP):
                land_gdf = gpd.read_file(f"zip://{ISLANDS_ZIP}").to_crs("EPSG:4326")
        except Exception as e:
            record_error("خطا در خواندن فایل Islands.zip", e)

        for reg_name, nc_out, reg_shp_path in st.session_state.nc_out_list:
            if not nc_out or not os.path.exists(nc_out):
                continue
            try:
                with xr.open_dataset(nc_out) as ds_pfz:
                    var_key = (
                        "pfz_index"
                        if "pfz_index" in ds_pfz
                        else list(ds_pfz.data_vars.keys())[0]
                    )
                    da_pfz = ds_pfz[var_key].load()

                if land_gdf is not None:
                    da_pfz = mask_land_from_pfz(da_pfz, land_gdf)

                # لایه کامل PFZ — فقط ادمین
                if user_is_admin:
                    img_path, bounds = render_pixel_perfect_heatmap(
                        da_pfz, "PFZ_Full", reg_name, "jet", OUTPUT_DIR
                    )
                    if img_path and bounds and os.path.exists(img_path):
                        shifted = apply_visual_offset_to_bounds(
                            bounds, OFFSET_LON_DEG, OFFSET_LAT_DEG
                        )
                        encoded = image_to_base64(img_path)
                        if encoded:
                            folium.raster_layers.ImageOverlay(
                                image=encoded,
                                bounds=shifted,
                                opacity=0.70,
                                name=f"🌊 پهنه‌بندی کل منطقه - PFZ Index ({reg_name})",
                                show=False,
                            ).add_to(m)

                # لایه PFZ اطراف جبهه‌ها — همه
                if st.session_state.combined_fronts_gdf is not None:
                    da_masked = mask_pfz_by_fronts(
                        da_pfz, st.session_state.combined_fronts_gdf
                    )
                    if da_masked is not None:
                        img_path, bounds = render_pixel_perfect_heatmap(
                            da_masked, "PFZ_NearFront", reg_name, "jet", OUTPUT_DIR
                        )
                        if img_path and bounds and os.path.exists(img_path):
                            shifted = apply_visual_offset_to_bounds(
                                bounds, OFFSET_LON_DEG, OFFSET_LAT_DEG
                            )
                            encoded = image_to_base64(img_path)
                            if encoded:
                                folium.raster_layers.ImageOverlay(
                                    image=encoded,
                                    bounds=shifted,
                                    opacity=0.75,
                                    name=f"🎯 پتانسیل صید نزدیک جبهه ({reg_name})",
                                    show=True,
                                ).add_to(m)

                # SST — فقط ادمین
                if user_is_admin and st.session_state.sst_nc_path:
                    try:
                        da_sst = load_and_crop_dataset(
                            st.session_state.sst_nc_path, reg_shp_path
                        )
                        if da_sst is not None:
                            img_path, bounds = render_pixel_perfect_heatmap(
                                da_sst, "SST", reg_name, "RdYlBu_r", OUTPUT_DIR
                            )
                            if img_path and bounds and os.path.exists(img_path):
                                shifted = apply_visual_offset_to_bounds(
                                    bounds, OFFSET_LON_DEG, OFFSET_LAT_DEG
                                )
                                encoded = image_to_base64(img_path)
                                if encoded:
                                    folium.raster_layers.ImageOverlay(
                                        image=encoded,
                                        bounds=shifted,
                                        opacity=0.65,
                                        name=f"🌡️ دمای سطح دریا - SST ({reg_name})",
                                        show=False,
                                    ).add_to(m)
                    except Exception as sst_ex:
                        record_error("خطا لایه SST", sst_ex)

                # Chlorophyll — فقط ادمین
                if user_is_admin and st.session_state.chl_nc_path:
                    try:
                        da_chl = load_and_crop_dataset(
                            st.session_state.chl_nc_path, reg_shp_path
                        )
                        if da_chl is not None:
                            img_path, bounds = render_pixel_perfect_heatmap(
                                da_chl, "Chlorophyll-a", reg_name, "YlGn", OUTPUT_DIR
                            )
                            if img_path and bounds and os.path.exists(img_path):
                                shifted = apply_visual_offset_to_bounds(
                                    bounds, OFFSET_LON_DEG, OFFSET_LAT_DEG
                                )
                                encoded = image_to_base64(img_path)
                                if encoded:
                                    folium.raster_layers.ImageOverlay(
                                        image=encoded,
                                        bounds=shifted,
                                        opacity=0.65,
                                        name=f"🌱 غلظت کلروفیل - Chlorophyll-a ({reg_name})",
                                        show=False,
                                    ).add_to(m)
                    except Exception as chl_ex:
                        record_error("خطا لایه Chl", chl_ex)

            except Exception as reg_ex:
                record_error(f"خطا در لایه منطقه {reg_name}", reg_ex)

    # خطوط جبهه — همه
    if (
        st.session_state.combined_fronts_gdf is not None
        and not st.session_state.combined_fronts_gdf.empty
    ):
        fronts_fg = folium.FeatureGroup(
            name="🚩 خطوط جبهه صیادی (Front Lines)", show=True
        )
        folium.GeoJson(
            st.session_state.combined_fronts_gdf,
            style_function=lambda x: {
                "color": "#000000",
                "weight": 3.5,
                "opacity": 0.95,
            },
        ).add_to(fronts_fg)
        fronts_fg.add_to(m)

    # مرز مناطق — فقط ادمین
    if user_is_admin and st.session_state.combined_region_gdf is not None:
        regions_fg = folium.FeatureGroup(name="📌 محدوده مناطق (Regions)", show=False)
        folium.GeoJson(
            st.session_state.combined_region_gdf,
            style_function=lambda x: {
                "color": "#007BFF",
                "fillColor": "transparent",
                "weight": 2,
                "dashArray": "5, 5",
            },
        ).add_to(regions_fg)
        regions_fg.add_to(m)

    # Legend
    bg_color = "rgba(20, 20, 20, 0.90)" if is_dark else "rgba(255, 255, 255, 0.95)"
    text_color = "#FFFFFF" if is_dark else "#1E3A8A"
    item_text_color = "#DDDDDD" if is_dark else "#333333"
    border_color = "#444444" if is_dark else "#2B5B84"

    legend_html = f"""
    <div style="position: fixed;
                bottom: 85px; left: 20px; width: 185px;
                z-index:9999; font-size:11px; background-color: {bg_color};
                border: 2px solid {border_color}; border-radius: 8px;
                padding: 8px; font-weight: bold; color: {text_color};
                direction: rtl; font-family: 'Vazirmatn', sans-serif;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4);">
        <div style="text-align: center; border-bottom: 1px solid #777; padding-bottom: 4px; margin-bottom: 6px; font-size: 12px; font-weight: bold;">
            🎯 پتانسیل صید (PFZ)
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
            <span style="width: 14px; height: 14px; background: #800000; display: inline-block; border-radius: 3px;"></span>
            <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">عالی (بسیار بالا)</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
            <span style="width: 14px; height: 14px; background: #FF0000; display: inline-block; border-radius: 3px;"></span>
            <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">خوب (بالا)</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
            <span style="width: 14px; height: 14px; background: #00FF00; display: inline-block; border-radius: 3px;"></span>
            <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">متوسط</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
            <span style="width: 14px; height: 14px; background: #00FFFF; display: inline-block; border-radius: 3px;"></span>
            <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">کم</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="width: 14px; height: 14px; background: #000080; display: inline-block; border-radius: 3px;"></span>
            <span style="flex-grow: 1; margin-right: 8px; color: {item_text_color};">بسیار کم / ناچیز</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
    if greg_str and jalali_str:
        persian_digits = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
        jalali_str_fa = jalali_str.translate(persian_digits)
        date_box_html = f"""
            <div style="position: fixed;
                        bottom: 25px; left: 20px; width: 250px; height: 50px;
                        z-index:9999; font-size:12px; background-color: {bg_color};
                        border: 2px solid {border_color}; border-radius: 6px;
                        padding: 4px; font-weight: bold; text-align: center; color: {text_color}; line-height: 1.4;
                        direction: rtl; font-family: 'Vazirmatn', sans-serif;">
                تاریخ اخذ داده: {jalali_str_fa}<br>
                <span style="font-family: Arial, sans-serif; color: {item_text_color}; font-size: 11px;">Data Acquisition Date: {greg_str}</span>
            </div>
        """
        m.get_root().html.add_child(folium.Element(date_box_html))

    m.fit_bounds(
        [
            [miny + OFFSET_LAT_DEG, minx + OFFSET_LON_DEG],
            [maxy + OFFSET_LAT_DEG, maxx + OFFSET_LON_DEG],
        ]
    )
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m
