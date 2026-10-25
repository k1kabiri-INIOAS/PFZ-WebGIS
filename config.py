# config.py — تنظیمات مرکزی سامانه PFZ
import os
import tempfile

# شیفت دیداری لایه‌های رستری روی نقشه (درجه)
OFFSET_LON_DEG = 0.00
OFFSET_LAT_DEG = -0.1

# مسیرها
DB_PATH = "users.db"
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "Data_Processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_FILE = os.path.join(OUTPUT_DIR, "app_state.json")
FRONTS_FILE = os.path.join(OUTPUT_DIR, "latest_fronts.geojson")
REGIONS_FILE = os.path.join(OUTPUT_DIR, "latest_regions.geojson")

DEFAULT_SHAPES_PATH = "default_shapes.zip"
ISLANDS_ZIP = "Islands.zip"
LOGO_FILENAME = "INIOAS Logo Color-P.jpg"

# پیش‌فرض وزن و آستانه هر منطقه
DEFAULT_REGION_DEFAULTS = {
    "Persian Gulf": {"sst_w": 0.70, "thresh": 0.40},
}

# نقشه
MAP_DEFAULT_ZOOM = 6
MAP_WIDTH = None          # None = responsive
MAP_HEIGHT_DESKTOP = 620
MAP_HEIGHT_MOBILE = 420

# ادمین پیش‌فرض (فقط در صورت نبودن در DB ساخته می‌شود)
DEFAULT_ADMIN_USER = "k1_kabiri"
DEFAULT_ADMIN_PASS = "Keivan@010976"
DEFAULT_ADMIN_FIRST = "کیسان"
DEFAULT_ADMIN_LAST = "کبیری"
