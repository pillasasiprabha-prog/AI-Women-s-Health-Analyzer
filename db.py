import sqlite3

def get_connection():
    return sqlite3.connect("womens_health_v2.db")

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    );
    """)

    conn.commit()
    conn.close()
