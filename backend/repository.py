import sqlite3
import json
import pandas as pd
import os

# Trỏ đúng vào file .db nằm cùng thư mục backend
DB_PATH = os.path.join(os.path.dirname(__file__), 'person_attributes.db')

def insert_recognition_log(track_id, crop_image_path, attributes_dict):
    """Hàm này để Linh gọi khi chạy xong AI: Bơm JSON vào Database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ép kiểu dictionary của VAnh thành chuỗi JSON
    attributes_json = json.dumps(attributes_dict)
    
    cursor.execute('''
        INSERT INTO recognition_logs (Track_ID, Timestamp, Crop_Image_Path, attributes)
        VALUES (?, datetime('now', 'localtime'), ?, ?)
    ''', (track_id, crop_image_path, attributes_json))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """Hàm này để UI gọi: Lấy dữ liệu ra vẽ biểu đồ"""
    conn = sqlite3.connect(DB_PATH)
    # Lấy dữ liệu và tự động chuyển thành DataFrame của Pandas
    df = pd.read_sql_query("SELECT * FROM recognition_logs ORDER BY id DESC", conn)
    conn.close()
    return df