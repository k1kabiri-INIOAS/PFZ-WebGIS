def generate_fronts_fallback(nc_path, output_geojson_path, user_threshold):
    try:
        if not nc_path or not os.path.exists(nc_path):
            record_error(f"فایل NetCDF وجود ندارد: {nc_path}")
            return False

        with xr.open_dataset(nc_path) as ds:
            if "pfz_index" not in ds:
                record_error("متغیر 'pfz_index' در فایل NetCDF یافت نشد.")
                return False
            
            da = ds["pfz_index"]
            lat_name = next((d for d in da.dims if d.lower() in ['lat', 'latitude', 'y']), None)
            lon_name = next((d for d in da.dims if d.lower() in ['lon', 'longitude', 'x']), None)
            
            if not lat_name or not lon_name:
                record_error("ابعاد مکانی (lat/lon) به درستی در فایل NetCDF یافت نشد.")
                return False
                
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            
            if da.ndim > 2:
                non_spatial_dims = [d for d in da.dims if d not in [lat_name, lon_name]]
                for d in non_spatial_dims:
                    da = da.isel({d: 0})
            
            data = da.values.copy()
            
        valid_mask = ~np.isnan(data) & (data > 0)
        if not valid_mask.any():
            record_error("ماتریس محاسباتی pfz_index کلاً از داده‌های NaN تشکیل شده است.")
            return False

        # ۱. پر کردن مقادیر NaN با میانگین داده‌های معتبر (جلوگیری از افت شدید گرادیان در مرز)
        mean_val = float(np.nanmean(data[valid_mask]))
        data_filled = np.where(valid_mask, data, mean_val)

        # ۲. اعمال فیلتر گوسی روی داده‌های یکنواخت‌شده
        data_smoothed = ndimage.gaussian_filter(data_filled, sigma=1.0).astype(float)

        # ۳. حذف حداقل ۳ پیکسل از تمامی مرزهای خشکی و لبه‌های تصویر (Mask Erosion)
        eroded_mask = ndimage.binary_erosion(valid_mask, structure=np.ones((3, 3)), iterations=3)
        
        # تمام پیکسل‌های ۳ پیکسل نزدیک به مرز/خشکی کاملاً NaN می‌شوند
        data_smoothed[~eroded_mask] = np.nan

        valid_smoothed = data_smoothed[eroded_mask]
        if len(valid_smoothed) == 0:
            record_error("پس از حذف مرزها، داده معتبری باقی نماند.")
            return False

        smooth_max = float(np.nanmax(valid_smoothed))
        active_threshold = user_threshold
        if active_threshold >= smooth_max:
            active_threshold = smooth_max * 0.85

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        def extract_lines(t_val):
            fig, ax = plt.subplots()
            cs = ax.contour(lon_grid, lat_grid, data_smoothed, levels=[t_val])
            extracted = []
            
            for segs in cs.allsegs:
                for poly in segs:
                    # حداقل ۳ نقطه برای رسم خط و عدم قرارگیری در حاشیه
                    if len(poly) > 2:
                        extracted.append(LineString(poly))
            
            plt.close(fig)
            return extracted

        lines = extract_lines(active_threshold)
        
        if not lines:
            fallback_percents = [0.70, 0.50, 0.30]
            for pct in fallback_percents:
                test_t = smooth_max * pct
                if test_t <= 0: continue
                lines = extract_lines(test_t)
                if lines:
                    active_threshold = test_t
                    break
        
        if lines:
            gdf_fronts = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
            gdf_fronts['Threshold'] = active_threshold
            gdf_fronts = gdf_fronts.to_crs("EPSG:3857")
            gdf_fronts['Length_km'] = gdf_fronts.geometry.length / 1000
            gdf_fronts = gdf_fronts.to_crs("EPSG:4326")
            
            # فیلتر خطوط کوتاه کمتر از ۱.۵ کیلومتر
            gdf_fronts = gdf_fronts[gdf_fronts['Length_km'] > 1.5]
            
            if not gdf_fronts.empty:
                os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
                gdf_fronts.to_file(output_geojson_path, driver="GeoJSON")
                return True
            else:
                record_error("جبهه‌ای خارج از محدوده مرزی یا با طول بیشتر از ۱.۵ کیلومتر یافت نشد.")
                return False
        else:
            record_error("هیچ خط کانتوری در آستانه‌های مختلف پیدا نشد.")
            return False
            
    except Exception as ex:
        record_error("خطای غیرمنتظره در generate_fronts_fallback", ex)
    return False