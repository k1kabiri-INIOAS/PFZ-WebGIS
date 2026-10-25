# auth/db.py — مدیریت کاربران، نقش‌ها و لاگ
import sqlite3
import hashlib
import datetime
import pandas as pd

from config import (
    DB_PATH,
    DEFAULT_ADMIN_USER,
    DEFAULT_ADMIN_PASS,
    DEFAULT_ADMIN_FIRST,
    DEFAULT_ADMIN_LAST,
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT,
        oauth_provider TEXT,
        first_name TEXT,
        last_name TEXT,
        phone TEXT,
        organization TEXT
    )"""
    )

    for col, col_type in [
        ("first_name", "TEXT"),
        ("last_name", "TEXT"),
        ("phone", "TEXT"),
        ("organization", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    c.execute(
        """CREATE TABLE IF NOT EXISTS user_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        timestamp TEXT
    )"""
    )

    c.execute("SELECT * FROM users WHERE username=?", (DEFAULT_ADMIN_USER,))
    if not c.fetchone():
        hashed_pw = hashlib.sha256(DEFAULT_ADMIN_PASS.encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password, role, oauth_provider, first_name, last_name) VALUES (?, ?, ?, ?, ?, ?)",
            (
                DEFAULT_ADMIN_USER,
                hashed_pw,
                "admin",
                "local",
                DEFAULT_ADMIN_FIRST,
                DEFAULT_ADMIN_LAST,
            ),
        )

    conn.commit()
    conn.close()


def log_user_activity(username, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)",
        (username, action, dt_str),
    )
    conn.commit()
    conn.close()


def create_user(
    username,
    password,
    role="user",
    provider="local",
    first_name="",
    last_name="",
    phone="",
    organization="",
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest() if password else ""
        c.execute(
            """INSERT INTO users (username, password, role, oauth_provider, first_name, last_name, phone, organization)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (username, hashed_pw, role, provider, first_name, last_name, phone, organization),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        c.execute(
            """UPDATE users SET first_name=?, last_name=?, phone=?, organization=? WHERE username=?""",
            (first_name, last_name, phone, organization, username),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hashlib.sha256(password.encode()).hexdigest():
        return user[1]
    return None


def login_or_register_email_user(email, first_name, last_name, phone, organization):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=?", (email,))
    user = c.fetchone()
    conn.close()
    if user:
        create_user(
            email,
            "",
            role=user[0],
            provider="email",
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            organization=organization,
        )
        return user[0]
    create_user(
        email,
        "",
        role="user",
        provider="email",
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        organization=organization,
    )
    return "user"


def get_user_logs(limit=50):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"""SELECT username AS 'نام کاربری/ایمیل', action AS 'عملیات', timestamp AS 'زمان'
                FROM user_logs ORDER BY id DESC LIMIT {int(limit)}""",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_all_users():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """SELECT username AS 'ایمیل/نام‌کاربری', first_name AS 'نام', last_name AS 'نام خانوادگی',
                      phone AS 'تلفن', organization AS 'سازمان', role AS 'نقش' FROM users""",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()
