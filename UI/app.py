import streamlit as st
import pandas as pd
import plotly.express as px
import json
import sys
import os

# Kết nối với backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import repository as repo

# Cấu hình trang - Phải để ở dòng đầu tiên
st.set_page_config(page_title="PAR Dashboard | Nhận diện Thuộc tính", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS GIÚP GIAO DIỆN CHUYÊN NGHIỆP HƠN ---
st.markdown("""
    <style>
        /* Ẩn menu mặc định và footer của Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    
        
        /* Chỉnh style cho nút bấm chính */
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            height: 45px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #34495e;
            border-color: #34495e;
            color: #f1c40f;
        }
        
        /* Chỉnh style cho các con số Metric (KPI) */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# KHU VỰC SIDEBAR (THANH ĐIỀU KHIỂN BÊN TRÁI)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3263/3263390.png", width=60) # Icon minh họa
    st.title("Bảng Cài đặt")
    st.markdown("---")
    
    st.subheader("1. Cấu hình Đầu vào")
    source_option = st.radio("Nguồn Dữ liệu:", ("📁 Upload Video", "📷 Live Webcam"))
    
    if source_option == "📁 Upload Video":
        st.file_uploader("Kéo thả file video vào đây", type=["mp4", "avi"])
        
    scale_threshold = st.slider("Hệ số Scale Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    st.markdown("---")
    st.subheader("2. Bộ lọc Dữ liệu")
    filter_gender = st.selectbox("Giới tính:", ["Tất cả", "Nam", "Nữ"])
    filter_accessory = st.selectbox("Phụ kiện:", ["Tất cả", "Balo", "Kính", "Túi xách"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Nút bấm tìm kiếm đặt ở sidebar
    btn_search = st.button("🔍 Lọc & Truy vấn Dữ liệu", use_container_width=True)
    
    st.markdown("---")
    st.button("▶️ Khởi động AI Pipeline", use_container_width=True, type="primary")


# ==========================================
# KHU VỰC MAIN BODY (HIỂN THỊ CHÍNH)
# ==========================================
st.title("HỆ THỐNG NHẬN DIỆN THUỘC TÍNH NGƯỜI")
st.markdown("Tiến trình: **Đang chờ tín hiệu...** | Tốc độ Pipeline: **0 FPS**")

# Xử lý Logic khi bấm nút Lọc (Lấy dữ liệu mặc định nếu chưa bấm)
results = repo.search_attributes(filter_gender, filter_accessory) if btn_search else repo.search_attributes("Tất cả", "Tất cả")

# Tiền xử lý dữ liệu để hiển thị
formatted_data = []
total_male = 0
total_female = 0

if results:
    for row in results:
        track_id, timestamp, crop_img, attr_json, conf_json = row
        attrs = json.loads(attr_json) 
        
        gender_str = "Nam" if attrs.get("gender") == "Male" else "Nữ"
        if gender_str == "Nam": total_male += 1
        else: total_female += 1
            
        formatted_data.append({
            "Track_ID": f"#{track_id:04d}", # Format ID cho đẹp (VD: #0012)
            "Thời điểm ghi nhận": timestamp,
            "Giới tính": gender_str,
            "Phụ kiện": "Kính" if attrs.get("glasses") else ("Balo" if attrs.get("bag") == "Backpack" else "Không có")
        })
df = pd.DataFrame(formatted_data)

# --- HIỂN THỊ KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Tổng số phát hiện", value=len(formatted_data))
with col2:
    st.metric(label="Số lượng Nam", value=total_male)
with col3:
    st.metric(label="Số lượng Nữ", value=total_female)
with col4:
    st.metric(label="Tình trạng Hệ thống", value="Sẵn sàng 🟢")

st.markdown("<br>", unsafe_allow_html=True)

# --- HIỂN THỊ DỮ LIỆU BẰNG TABS ---
tab1, tab2, tab3 = st.tabs(["📺 Màn hình Giám sát (Live)", "📋 Dữ liệu Chi tiết", "📊 Báo cáo Thống kê"])

with tab1:
    st.info("Khu vực này sẽ hiển thị luồng Video trực tiếp khi module pipeline.py được tích hợp.")
    # Khung giả lập video để UI không bị trống
    st.markdown("""
        <div style='background-color: #1e1e1e; height: 400px; border-radius: 10px; display: flex; align-items: center; justify-content: center; border: 2px dashed #555;'>
            <h3 style='color: #666;'>[ No Signal ]</h3>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    if not df.empty:
        # Sử dụng dataframe với giao diện sáng sủa
        st.dataframe(df, use_container_width=True, height=350, hide_index=True)
    else:
        st.warning("Không có dữ liệu nào khớp với bộ lọc hiện tại.")

with tab3:
    if not df.empty and len(df) > 0:
        c1, c2 = st.columns([1, 1])
        with c1:
            gender_count = df['Giới tính'].value_counts().reset_index()
            gender_count.columns = ['Giới tính', 'Số lượng']
            fig_gender = px.pie(gender_count, values='Số lượng', names='Giới tính', 
                                title='Tỷ lệ Nam/Nữ', hole=0.4, # Tạo dáng Donut chart (có lỗ ở giữa)
                                color_discrete_sequence=['#3498db', '#e74c3c'])
            st.plotly_chart(fig_gender, use_container_width=True)
            
        with c2:
            acc_count = df['Phụ kiện'].value_counts().reset_index()
            acc_count.columns = ['Phụ kiện', 'Số lượng']
            fig_acc = px.bar(acc_count, x='Phụ kiện', y='Số lượng', 
                             title='Thống kê Phụ kiện mang theo', text_auto=True,
                             color='Phụ kiện', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Cần có dữ liệu để vẽ biểu đồ.")