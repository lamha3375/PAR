import streamlit as st
import pandas as pd
import json
import plotly.express as px
import sys
import os

# Thêm đường dẫn để import được thư mục backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.repository import get_all_logs

# Cấu hình trang
st.set_page_config(page_title="Hệ thống UPAR", layout="wide")

st.title("HỆ THỐNG NHẬN DIỆN THUỘC TÍNH NGƯỜI")
st.write("Hệ thống tự động phát hiện, theo vết và nhận diện thuộc tính người từ luồng Video/Webcam.")

# ==========================================
# 1. FORM CHỌN NGUỒN DỮ LIỆU & XỬ LÝ
# ==========================================
st.sidebar.header("📺 Chọn Nguồn Dữ Liệu")
input_source = st.sidebar.radio("Định dạng đầu vào:", ["Mở Webcam (Live)", "Upload File Video"])

start_processing = False
if input_source == "Upload File Video":
    video_file = st.sidebar.file_uploader("Tải lên Video (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])
    if st.sidebar.button("▶️ Chạy Xử Lý Video", type="primary"):
        if video_file: 
            start_processing = True
        else: 
            st.sidebar.warning("Vui lòng tải video lên trước!")
else:
    if st.sidebar.button("▶️ Bật Webcam", type="primary"):
        start_processing = True
        
st.sidebar.markdown("---")

# ==========================================
# 2. BỘ LỌC TÌM KIẾM & NGƯỠNG (THRESHOLD)
# ==========================================
st.sidebar.header("🔍 Lọc & Tìm Kiếm")
# Thanh chỉnh Threshold Scale (theo chuẩn tài liệu)
threshold = st.sidebar.slider("Ngưỡng tin cậy (Scale Threshold)", 0.0, 1.0, 0.6)

filter_gender = st.sidebar.selectbox("Giới tính", ["Tất cả", "Male", "Female"])
filter_age = st.sidebar.selectbox("Độ tuổi", ["Tất cả", "Young", "Adult", "Old"])
filter_color = st.sidebar.selectbox("Màu áo", ["Tất cả", "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"])
filter_bag = st.sidebar.selectbox("Phụ kiện (Balo/Túi)", ["Tất cả", "Backpack", "Bag/Handbag", "No Bag"])
filter_glasses = st.sidebar.selectbox("Kính", ["Tất cả", "Normal Glasses", "Sunglasses", "No Glasses"])

# ==========================================
# 3. KHU VỰC CHIẾU VIDEO / WEBCAM
# ==========================================
st.subheader("🔴 Luồng Theo Dõi Trực Tiếp")
video_placeholder = st.empty() 

if not start_processing:
    video_placeholder.info("Màn hình đang tắt. Vui lòng chọn nguồn dữ liệu và bấm Bật ở thanh bên trái.")
else:
    video_placeholder.warning("Đang kết nối luồng Pipeline YOLOv8 & ByteTrack từ nhóm... (Chờ tích hợp code của Linh)")

st.markdown("---")

# ==========================================
# 4. KHU VỰC BIỂU ĐỒ & BẢNG DỮ LIỆU
# ==========================================
st.subheader("📊 Báo Cáo Thống Kê Lịch Sử")
df = get_all_logs()

if not df.empty:
    # Hàm nhỏ để trích xuất dữ liệu từ chuỗi JSON
    def extract_attr(json_str, head_name):
        try:
            return json.loads(json_str).get(head_name, {}).get('prediction', 'Unknown')
        except: 
            return 'Unknown'

    # Tạo các cột dữ liệu ảo để phục vụ lọc và vẽ biểu đồ
    df['Giới tính'] = df['attributes'].apply(lambda x: extract_attr(x, 'gender'))
    df['Độ tuổi'] = df['attributes'].apply(lambda x: extract_attr(x, 'age'))
    df['Màu áo'] = df['attributes'].apply(lambda x: extract_attr(x, 'upper_color'))
    df['Phụ kiện'] = df['attributes'].apply(lambda x: extract_attr(x, 'bag'))
    df['Kính'] = df['attributes'].apply(lambda x: extract_attr(x, 'glasses'))

    # Xử lý Logic lọc dữ liệu từ thanh Sidebar
    if filter_gender != "Tất cả": df = df[df['Giới tính'] == filter_gender]
    if filter_age != "Tất cả": df = df[df['Độ tuổi'] == filter_age]
    if filter_color != "Tất cả": df = df[df['Màu áo'] == filter_color]
    if filter_bag != "Tất cả": df = df[df['Phụ kiện'] == filter_bag]
    if filter_glasses != "Tất cả": df = df[df['Kính'] == filter_glasses]

    if df.empty:
        st.warning("Không có đối tượng nào khớp với bộ lọc!")
    else:
        # Vẽ 3 biểu đồ
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_gender = px.pie(df, names='Giới tính', title="Phân bố Giới tính", hole=0.4, color_discrete_sequence=['#3498db', '#e74c3c'])
            st.plotly_chart(fig_gender, use_container_width=True)
        with col2:
            fig_age = px.bar(df['Độ tuổi'].value_counts().reset_index(), x='Độ tuổi', y='count', title="Phân bố Độ tuổi")
            st.plotly_chart(fig_age, use_container_width=True)
        with col3:
            fig_bag = px.pie(df, names='Phụ kiện', title="Thống kê Phụ kiện", hole=0.3)
            st.plotly_chart(fig_bag, use_container_width=True)

        # Hiển thị bảng chi tiết
        st.subheader("🗄️ Bảng Dữ Liệu Đối Tượng")
        st.dataframe(df[['Track_ID', 'Timestamp', 'Giới tính', 'Độ tuổi', 'Màu áo', 'Phụ kiện', 'Kính', 'Crop_Image_Path', 'attributes']])
else:
    st.info("Cơ sở dữ liệu trống. Hệ thống đang chờ ghi nhận dữ liệu từ luồng Camera/Video...")