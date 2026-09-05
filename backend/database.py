import sqlite3
import os

# Trỏ đúng vào file .db nằm cùng thư mục backend
DB_PATH = os.path.join(os.path.dirname(__file__), 'person_attributes.db')

def init_db():
    """Tạo bảng cơ sở dữ liệu chuẩn UPAR"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng với 1 cột attributes duy nhất (kiểu TEXT) để chứa cục JSON
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Track_ID INTEGER,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Crop_Image_Path TEXT,
            attributes TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("1. Đã khởi tạo cấu trúc Database mới thành công!")

def insert_mock_data():
    """Bơm dữ liệu giả để Web có biểu đồ hiển thị ngay"""
    from repository import insert_recognition_log
    
    # Dữ liệu giả mô phỏng đúng chuẩn JSON 11 Head của VAnh
    mock_1 = {
        "age": {"prediction": "Young", "confidence": 0.95},
        "gender": {"prediction": "Female", "confidence": 0.99},
        "upper_color": {"prediction": "White", "confidence": 0.90},
        "bag": {"prediction": "Bag/Handbag", "confidence": 0.88}
    }
    mock_2 = {
        "age": {"prediction": "Adult", "confidence": 0.90},
        "gender": {"prediction": "Male", "confidence": 0.97},
        "upper_color": {"prediction": "Black", "confidence": 0.94},
        "bag": {"prediction": "Backpack", "confidence": 0.85}
    }
    
    insert_recognition_log(1, "images/crop_1.jpg", mock_1)
    insert_recognition_log(2, "images/crop_2.jpg", mock_2)
    print("2. Đã bơm Dữ liệu giả thành công! Hãy chạy web để xem thành quả.")

if __name__ == "__main__":
    init_db()
    insert_mock_data()