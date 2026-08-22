# Data Directory Instructions

Tập dữ liệu ảnh người đi bộ (Market-1501, PA-100K, PETA) có kích thước lớn và không được lưu trực tiếp trên Git repository.

## Hướng dẫn đặt dữ liệu ảnh local

1. Tải và giải nén các tập dữ liệu gốc vào thư mục local:
   * `Market-1501`
   * `PA-100K`
   * `PETA`

2. Đặt dữ liệu ảnh vào cấu trúc thư mục sau:
   ```text
   D:\AI DATASET\3 Datasets\
   ├── Market1501/
   ├── PA-100K/
   └── PETA/
   ```

3. Các tệp annotation chuẩn hóa (.pkl) của bộ dữ liệu hợp nhất `UPAR_UNIFIED` được đặt tại:
   ```text
   D:\AI DATASET\UPAR_UNIFIED\annotations\
   ├── unified_annotations.pkl
   ├── train.pkl
   ├── val.pkl
   └── test.pkl
   ```
