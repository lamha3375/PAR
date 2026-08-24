import sqlite3
import json
from datetime import datetime
from backend.database import get_connection

def insert_log(track_id, crop_image_path, attributes_dict, confidence_dict):
    """
    Lưu dữ liệu vào DB. 
    attributes_dict và confidence_dict sẽ được chuyển thành chuỗi JSON.
    """
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Chuyển Dictionary thành chuỗi JSON để lưu trữ
    attr_json = json.dumps(attributes_dict)
    conf_json = json.dumps(confidence_dict)
    
    cursor.execute('''
        INSERT INTO recognition_logs (track_id, timestamp, crop_image_path, attributes, confidence_scores)
        VALUES (?, ?, ?, ?, ?)
    ''', (track_id, timestamp, crop_image_path, attr_json, conf_json))
    
    conn.commit()
    conn.close()

def search_attributes(gender_filter, accessory_filter):
    """Lọc dữ liệu bằng cách tìm chuỗi trong cột JSON"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT track_id, timestamp, crop_image_path, attributes, confidence_scores FROM recognition_logs WHERE 1=1"
    params = []
    
    # Ép kiểu bộ lọc Tiếng Việt sang Tiếng Anh để khớp với dữ liệu JSON của AI
    if gender_filter != "Tất cả":
        gender_val = "Male" if gender_filter == "Nam" else "Female"
        query += " AND attributes LIKE ?"
        params.append(f'%"{gender_val}"%')
        
    if accessory_filter != "Tất cả":
        acc_val = "Backpack" if accessory_filter == "Balo" else ("Glasses" if accessory_filter == "Kính" else "Handbag")
        query += " AND attributes LIKE ?"
        params.append(f'%"{acc_val}"%')
        
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    return results