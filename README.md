# سامانه مدیریت PFZ (Potential Fishing Zones)

نسخه بازآرایی‌شده اپلیکیشن Streamlit WebGIS برای شناسایی مناطق مستعد صیادی در خلیج فارس، دریای عمان و دریای عرب.

## ساختار پروژه

```
PFZ/
├── app.py                 # نقطه ورود — orchestration سبک
├── config.py              # تنظیمات مرکزی
├── auth/                  # کاربران، نقش‌ها، لاگ
│   ├── db.py
│   └── __init__.py
├── modules/
│   ├── fetcher.py         # دریافت SST/CHL از ERDDAP
│   ├── processor.py       # pipeline محاسبه شاخص PFZ
│   ├── fronts.py          # استخراج و ماسک جبهه‌ها
│   └── rendering.py       # heatmap و برش داده
├── map/
│   └── folium_map.py      # نقشه Folium + کنترل مختصات
├── ui/
│   ├── styles.py          # تم روز/شب، لوگو، CSS موبایل
│   ├── login.py
│   ├── sidebar.py
│   └── analytics.py       # آمار جبهه + دانلود GeoJSON/CSV
├── utils/
│   ├── dates.py           # تبدیل تاریخ شمسی
│   ├── state.py           # ذخیره/بارگذاری وضعیت
│   └── logging_helpers.py
├── default_shapes.zip
├── Islands.zip
├── requirements.txt
└── Shape_Files/
```

## قابلیت‌های جدید این نسخه

1. **Refactor ماژولار** — جداسازی auth، map، processing و UI از `app.py`
2. **دانلود خروجی** — GeoJSON جبهه‌ها، CSV خلاصه، GeoJSON مناطق (ادمین)
3. **آمار تحلیلی** — تعداد قطعات، مجموع طول جبهه، تفکیک منطقه‌ای
4. **بهبود UX** — پیام خطای واضح‌تر، CSS واکنش‌گرا برای موبایل، balloons پس از موفقیت

## اجرا

```bash
pip install -r requirements.txt
streamlit run app.py
```

## نقش‌ها

| نقش | دسترسی |
|-----|--------|
| **admin** | اجرای تحلیل، لایه SST/CHL/PFZ کامل، مدیریت کاربران، دانلود مناطق |
| **user** | مشاهده نقشه جبهه‌ها و PFZ نزدیک جبهه، دانلود GeoJSON/CSV جبهه‌ها |

## دیپلوی Streamlit Cloud

همان `app.py` و `requirements.txt` را در ریشه ریپو نگه دارید. فایل‌های `default_shapes.zip` و `Islands.zip` باید در ریشه باشند.
