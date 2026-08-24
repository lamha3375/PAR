import sqlite3

DB_NAME = "backend/person_attributes.db"

def get_connection():
    """Hàm tạo kết nối đến Database"""
    return sqlite3.connect(DB_NAME)

def init_db():
    """Khởi tạo bảng với cấu trúc chuẩn JSON"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            crop_image_path TEXT,
            attributes TEXT,
            confidence_scores TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Tự động chạy khởi tạo khi import
init_db()