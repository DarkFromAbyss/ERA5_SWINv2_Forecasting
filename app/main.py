import streamlit as st
from components.sidebar import render_common_sidebar

# Cấu hình trang chủ
st.set_page_config(
    page_title="Weather AI Home", page_icon="⛈️", layout="wide"
)

# Hiển thị sidebar thông tin dùng chung
render_common_sidebar()

# Giao diện chính của trang chủ chào mừng
st.title("⛈️ Hệ Thống Dự Đoán Thời Tiết Bằng Mô Hình Swin v2")
st.markdown("---")

st.markdown("""
### 👋 Chào mừng bạn đến với Hệ thống dự báo thời tiết thông minh
Hệ thống tận dụng sức mạnh của kiến trúc mạng **Swin Transformer v2** phối hợp cùng dữ liệu mô phỏng khí hậu toàn cầu **ERA5** từ ECMWF.

#### 👈 Hướng dẫn sử dụng:
Vui lòng sử dụng menu điều hướng tự động ở **Thanh bên trái (Sidebar)** để chuyển mạch giữa các phân hệ:
1. **🛠️ Config:** Thiết lập tham số hệ thống, vùng tọa độ địa lý, định dạng tệp.
2. **📥 Download:** Kết nối API vận hành tải dữ liệu tự động.
3. **📊 Visualization:** Trực quan hóa bản đồ nhiệt độ và các luồng gió động tương tác.
4. **🏋️ Training:** Huấn luyện mạng Nơ-ron PyTorch.
5. **🔮 Inference:** Đưa ra dự báo mô hình cho tương lai.
""")