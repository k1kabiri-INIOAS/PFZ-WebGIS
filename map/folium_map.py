# map/folium_map.py — نقشه Folium با گرافیک حرفه‌ای دریایی
import os
import folium
from folium.plugins import Fullscreen, MiniMap, MousePosition
from branca.element import MacroElement, Element
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

# رنگ یکسان خطوط جبهه در همه مناطق
DEFAULT_FRONT_COLOR = "#FF1744"


class CustomMapFeatures(MacroElement):
    """کنترل مختصات + مارکر کلیک با پاپ‌آپ."""

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
      div.style.cssText = 'padding:8px 12px;background:rgba(15,23,42,0.92);border:1px solid #38bdf8;border-radius:10px;font-size:12px;direction:ltr;font-family:ui-monospace,monospace;display:flex;align-items:center;gap:10px;box-shadow:0 4px 14px rgba(0,0,0,0.35);z-index:1000;color:#e2e8f0;';
      div.innerHTML = `
        <span id="coord-text" style="font-weight:600;color:#f8fafc;">Lat: -- | Lng: --</span>
        <button id="toggle-coord-btn" title="تغییر فرمت" style="cursor:pointer;padding:3px 8px;font-size:11px;font-weight:700;border:1px solid #38bdf8;background:#0ea5e9;border-radius:6px;color:#fff;">DDM</button>
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
        currentMarker = L.marker(latlng, {
          icon: L.divIcon({
            className: 'pfz-click-marker',
            html: '<div style="width:14px;height:14px;border-radius:50%;background:#38bdf8;border:2px solid #fff;box-shadow:0 0 8px #0ea5e9;"></div>',
            iconSize: [14, 14],
            iconAnchor: [7, 7]
          })
        }).addTo(map);
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
        <div style="direction:ltr;text-align:center;font-family:system-ui,sans-serif;font-size:12px;color:#0f172a;min-width:230px;padding:4px;">
          <div style="margin-bottom:8px;font-weight:700;font-size:13px;color:#0369a1;">${latDDM}<br>${lngDDM}</div>
          <input type="text" id="coord-input-box" value="${copyText}" readonly
            style="width:100%;text-align:center;font-family:monospace;font-size:11px;padding:6px;margin-bottom:8px;border:1px solid #0ea5e9;border-radius:6px;background:#f0f9ff;color:#0f172a;" />
          <button id="popup-copy-btn" style="cursor:pointer;padding:7px 8px;font-size:11px;border:none;background:linear-gradient(135deg,#0284c7,#0ea5e9);color:white;border-radius:6px;width:100%;font-weight:700;margin-bottom:5px;">📋 کپی مختصات</button>
          <button id="popup-share-btn" style="cursor:pointer;padding:7px 8px;font-size:11px;border:none;background:#64748b;color:white;border-radius:6px;width:100%;font-weight:700;margin-bottom:5px;">🔀 اشتراک‌گذاری</button>
          <a href="${gmapsUrl}" target="_blank" style="display:block;padding:7px 8px;font-size:11px;background:#4285F4;color:white;text-decoration:none;border-radius:6px;font-weight:700;text-align:center;margin-bottom:4px;">📍 Google Maps</a>
          <a href="${openSeaMapUrl}" target="_blank" style="display:block;padding:7px 8px;font-size:11px;background:#0e7490;color:white;text-decoration:none;border-radius:6px;font-weight:700;text-align:center;margin-bottom:4px;">🌐 OpenSeaMap</a>
          <a href="${windyUrl}" target="_blank" style="display:block;padding:7px 8px;font-size:11px;background:#1e40af;color:white;text-decoration:none;border-radius:6px;font-weight:700;text-align:center;">🌊 Windy</a>
        </div>
      `;
      currentMarker.bindPopup(popupHtml, {maxWidth: 280}).openPopup();
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
              copyBtn.innerText = '✓ کپی شد';
              copyBtn.style.background = '#059669';
              setTimeout(() => { copyBtn.innerText = '📋 کپی مختصات'; copyBtn.style.background = 'linear-gradient(135deg,#0284c7,#0ea5e9)'; }, 2000);
            } catch (err) {
              copyBtn.innerText = 'Ctrl+C';
            }
          });
        }
        if (shareBtn) {
          L.DomEvent.disableClickPropagation(shareBtn);
          shareBtn.addEventListener('click', () => {
            if (navigator.share) {
              navigator.share({ title: 'مختصات PFZ', text: copyText, url: gmapsUrl }).catch(() => {});
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


def _front_style_factory(color):
    """استایل خطوط جبهه با هاله درخشان (دو لایه)."""

    def core_style(feature):
        return {
            "color": color,
            "weight": 2.8,
            "opacity": 0.95,
            "lineCap": "round",
            "lineJoin": "round",
        }

    def halo_style(feature):
        return {
            "color": color,
            "weight": 7,
            "opacity": 0.28,
            "lineCap": "round",
            "lineJoin": "round",
        }

    return halo_style, core_style


def _add_gradient_legend(m, is_dark: bool):
    """Legend گرادیانی افقی برای PFZ."""
    bg = "rgba(15, 23, 42, 0.92)" if is_dark else "rgba(255, 255, 255, 0.95)"
    fg = "#f1f5f9" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#475569"
    border = "#38bdf8" if is_dark else "#0284c7"

    # گرادیان مطابق colormap جدید
    gradient = "linear-gradient(90deg, #0B1D36 0%, #0D3B66 12%, #1A6B9A 25%, #1B9E77 40%, #5CB85C 55%, #E6C229 70%, #F4A261 85%, #9B2226 100%)"

    html = f"""
    <div style="position:fixed; bottom:90px; left:18px; z-index:9999;
                background:{bg}; border:1.5px solid {border}; border-radius:12px;
                padding:10px 12px; width:220px;
                box-shadow:0 8px 24px rgba(0,0,0,0.25);
                font-family:'Vazirmatn', system-ui, sans-serif; direction:rtl; color:{fg};">
      <div style="text-align:center; font-weight:800; font-size:12px; margin-bottom:8px; letter-spacing:0.3px;">
        🎯 پتانسیل صید (PFZ)
      </div>
      <div style="height:14px; border-radius:7px; background:{gradient};
                  border:1px solid rgba(255,255,255,0.2); box-shadow:inset 0 1px 2px rgba(0,0,0,0.2);"></div>
      <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:10px; color:{muted}; font-weight:600; direction:ltr;">
        <span>بسیار کم</span>
        <span>متوسط</span>
        <span>عالی</span>
      </div>
      <div style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(148,163,184,0.35); font-size:10px; color:{muted};">
        <span style="display:inline-block;width:18px;height:3px;background:#FF1744;border-radius:2px;vertical-align:middle;margin-left:6px;box-shadow:0 0 6px #FF1744;"></span>
        خطوط جبهه صیادی
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(html))


def _add_date_badge(m, is_dark: bool):
    greg_str, jalali_str = parse_date_formats(st.session_state.latest_date)
    if not greg_str:
        return
    bg = "rgba(15, 23, 42, 0.92)" if is_dark else "rgba(255, 255, 255, 0.95)"
    fg = "#f1f5f9" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#64748b"
    border = "#38bdf8" if is_dark else "#0284c7"
    persian_digits = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    jalali_fa = (jalali_str or "").translate(persian_digits)

    html = f"""
    <div style="position:fixed; bottom:28px; left:18px; z-index:9999;
                background:{bg}; border:1.5px solid {border}; border-radius:10px;
                padding:6px 12px; min-width:200px; text-align:center;
                box-shadow:0 4px 16px rgba(0,0,0,0.2);
                font-family:'Vazirmatn', system-ui, sans-serif; direction:rtl; color:{fg};">
      <div style="font-size:11px; font-weight:700;">📅 تاریخ داده: {jalali_fa}</div>
      <div style="font-size:10px; color:{muted}; font-family:ui-monospace,monospace; margin-top:2px;">{greg_str}</div>
    </div>
    """
    m.get_root().html.add_child(Element(html))


def _da_from_current_nc(nc_path, shp_path):
    """سرعت جریان |V| از فایل ugos/vgos یا u_current/v_current."""
    if not nc_path or not os.path.exists(nc_path):
        return None
    try:
        with xr.open_dataset(nc_path) as ds:
            names = {v.lower(): v for v in ds.data_vars}
            u_var = next((names[k] for k in ("ugos", "u_current", "uo", "u") if k in names), None)
            v_var = next((names[k] for k in ("vgos", "v_current", "vo", "v") if k in names), None)
            if u_var is None or v_var is None:
                return None
            u = ds[u_var].squeeze()
            v = ds[v_var].squeeze()
            while u.ndim > 2:
                u = u.isel({u.dims[0]: 0})
            while v.ndim > 2:
                v = v.isel({v.dims[0]: 0})
            speed = (u.astype("float64") ** 2 + v.astype("float64") ** 2) ** 0.5
            speed.name = "current_speed"
            # crop via existing helper by writing temp — or reuse load path
            tmp = os.path.join(OUTPUT_DIR, "_tmp_current_speed.nc")
            speed.to_netcdf(tmp)
            da = load_and_crop_dataset(tmp, shp_path)
            try:
                os.remove(tmp)
            except Exception:
                pass
            return da
    except Exception as e:
        record_error("خطا بارگذاری جریان برای نقشه", e)
        return None


def _da_from_ssh_nc(nc_path, shp_path):
    """گرادیان |∇SLA| از فایل SSH."""
    if not nc_path or not os.path.exists(nc_path):
        return None
    try:
        with xr.open_dataset(nc_path) as ds:
            ssh_var = next(
                (
                    v
                    for v in ds.data_vars
                    if any(k in v.lower() for k in ("sla", "ssh", "adt", "height", "anomaly"))
                ),
                None,
            )
            if ssh_var is None:
                return None
            da = ds[ssh_var].squeeze()
            while da.ndim > 2:
                da = da.isel({da.dims[0]: 0})
            vals = da.astype("float64").values
            import numpy as np
            finite = np.isfinite(vals)
            if not finite.any():
                return None
            fill = float(np.nanmean(vals[finite]))
            filled = np.where(finite, vals, fill)
            dy, dx = np.gradient(filled)
            grad = np.sqrt(dx ** 2 + dy ** 2)
            grad_da = xr.DataArray(grad, coords=da.coords, dims=da.dims, name="ssh_grad")
            tmp = os.path.join(OUTPUT_DIR, "_tmp_ssh_grad.nc")
            grad_da.to_netcdf(tmp)
            out = load_and_crop_dataset(tmp, shp_path)
            try:
                os.remove(tmp)
            except Exception:
                pass
            return out
    except Exception as e:
        record_error("خطا بارگذاری SSH برای نقشه", e)
        return None



def build_pfz_map(is_dark=False):
    """ساخت نقشه کامل PFZ با گرافیک ارتقایافته."""
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
        control_scale=True,
        prefer_canvas=True,
    )

    # ----- لایه‌های پایه -----
    # پیش‌فرض: Esri Ocean / یا Carto dark بر اساس تم
    # basemapهای رایگان بدون API Key
    if is_dark:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri — Dark Gray Canvas",
            name="🌑 نقشه تیره (Esri Dark)",
            show=True,
            max_zoom=16,
        ).add_to(m)
        # لایه برچسب تیره (اختیاری)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri",
            name="🏷️ برچسب تیره",
            overlay=True,
            control=True,
            show=True,
            max_zoom=16,
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri — Ocean Basemap",
            name="🌊 بستر اقیانوس (Esri Ocean)",
            show=True,
            max_zoom=13,
        ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri — World Imagery",
        name="🛰️ تصاویر ماهواره‌ای (Esri)",
        show=False,
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="🛰️ گوگل ماهواره",
        show=False,
        max_zoom=20,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="🗺️ گوگل ترکیبی",
        show=False,
        max_zoom=20,
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="🗺️ خیابانی (OSM)",
        show=False,
    ).add_to(m)

    # OpenSeaMap overlay (علامت‌های دریایی)
    folium.TileLayer(
        tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        attr="OpenSeaMap",
        name="⚓ علائم دریایی (OpenSeaMap)",
        overlay=True,
        control=True,
        show=False,
        opacity=0.85,
    ).add_to(m)

    CustomMapFeatures().add_to(m)

    Fullscreen(
        position="topright",
        title="تمام‌صفحه",
        title_cancel="خروج",
        force_separate_button=True,
    ).add_to(m)

    # MiniMap با کاشی رایگان OSM (بدون API Key)
    minimap_tiles = folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap",
        name="minimap_base",
        control=False,
    )
    MiniMap(
        tile_layer=minimap_tiles,
        position="bottomright",
        width=140,
        height=140,
        zoom_level_offset=-5,
        toggle_display=True,
    ).add_to(m)

    # ----- لایه‌های رستری -----
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

                # PFZ کامل — ادمین
                if user_is_admin:
                    img_path, bounds = render_pixel_perfect_heatmap(
                        da_pfz, "PFZ_Full", reg_name, "pfz", OUTPUT_DIR, alpha=0.72
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
                                opacity=1.0,
                                name=f"🌊 پهنه‌بندی کامل PFZ ({reg_name})",
                                show=False,
                                interactive=False,
                            ).add_to(m)

                # PFZ نزدیک جبهه — همه (لایه اصلی قابل‌رؤیت)
                if st.session_state.combined_fronts_gdf is not None:
                    da_masked = mask_pfz_by_fronts(
                        da_pfz, st.session_state.combined_fronts_gdf
                    )
                    if da_masked is not None:
                        img_path, bounds = render_pixel_perfect_heatmap(
                            da_masked,
                            "PFZ_NearFront",
                            reg_name,
                            "pfz",
                            OUTPUT_DIR,
                            alpha=0.88,
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
                                    opacity=1.0,
                                    name=f"🎯 پتانسیل نزدیک جبهه ({reg_name})",
                                    show=True,
                                    interactive=False,
                                ).add_to(m)

                if user_is_admin and st.session_state.sst_nc_path:
                    try:
                        da_sst = load_and_crop_dataset(
                            st.session_state.sst_nc_path, reg_shp_path
                        )
                        if da_sst is not None:
                            img_path, bounds = render_pixel_perfect_heatmap(
                                da_sst, "SST", reg_name, "sst", OUTPUT_DIR, alpha=0.75
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
                                        opacity=1.0,
                                        name=f"🌡️ SST ({reg_name})",
                                        show=False,
                                    ).add_to(m)
                    except Exception as sst_ex:
                        record_error("خطا لایه SST", sst_ex)

                if user_is_admin and st.session_state.chl_nc_path:
                    try:
                        da_chl = load_and_crop_dataset(
                            st.session_state.chl_nc_path, reg_shp_path
                        )
                        if da_chl is not None:
                            img_path, bounds = render_pixel_perfect_heatmap(
                                da_chl, "Chlorophyll-a", reg_name, "chl", OUTPUT_DIR, alpha=0.75
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
                                        opacity=1.0,
                                        name=f"🌱 کلروفیل ({reg_name})",
                                        show=False,
                                    ).add_to(m)
                    except Exception as chl_ex:
                        record_error("خطا لایه Chl", chl_ex)

            except Exception as reg_ex:
                record_error(f"خطا در لایه منطقه {reg_name}", reg_ex)

    # ----- خطوط جبهه با هاله درخشان (یک رنگ برای همه مناطق) -----
    fronts = st.session_state.combined_fronts_gdf
    if fronts is not None and not fronts.empty:
        fronts_fg = folium.FeatureGroup(
            name="🚩 خطوط جبهه صیادی", show=True
        )
        halo_style, core_style = _front_style_factory(DEFAULT_FRONT_COLOR)
        # هاله
        folium.GeoJson(fronts, style_function=halo_style).add_to(fronts_fg)
        # خط اصلی + tooltip
        tooltip_fields = [c for c in ["Region", "Length_km", "Threshold"] if c in fronts.columns]
        tooltip = None
        if tooltip_fields:
            tooltip = folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=["منطقه", "طول (km)", "آستانه"][: len(tooltip_fields)],
                localize=True,
                style=(
                    "background-color:rgba(15,23,42,0.9);color:#f8fafc;"
                    "font-family:Vazirmatn,sans-serif;font-size:12px;padding:6px;border-radius:6px;"
                ),
            )
        folium.GeoJson(
            fronts,
            style_function=core_style,
            tooltip=tooltip,
        ).add_to(fronts_fg)
        fronts_fg.add_to(m)

    # ----- مرز مناطق (ادمین) -----
    if user_is_admin and st.session_state.combined_region_gdf is not None:
        regions_fg = folium.FeatureGroup(name="📌 محدوده مناطق", show=False)
        folium.GeoJson(
            st.session_state.combined_region_gdf,
            style_function=lambda x: {
                "color": "#38bdf8",
                "fillColor": "#0ea5e9",
                "fillOpacity": 0.04,
                "weight": 2,
                "dashArray": "8, 6",
                "opacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["Region"] if "Region" in st.session_state.combined_region_gdf.columns else [],
                aliases=["منطقه"],
            )
            if "Region" in st.session_state.combined_region_gdf.columns
            else None,
        ).add_to(regions_fg)
        regions_fg.add_to(m)

    _add_gradient_legend(m, is_dark)
    _add_date_badge(m, is_dark)

    m.fit_bounds(
        [
            [miny + OFFSET_LAT_DEG, minx + OFFSET_LON_DEG],
            [maxy + OFFSET_LAT_DEG, maxx + OFFSET_LON_DEG],
        ]
    )
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m
