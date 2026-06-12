import os
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import yaml
import leafmap.foliumap as leafmap  # Sử dụng Folium backend tương thích tốt nhất với Streamlit
import matplotlib.pyplot as plt

# Xác định đường dẫn gốc và file cấu hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config():
    """Đọc file cấu hình config.yaml từ thư mục config."""
    if not os.path.exists(CONFIG_PATH):
        st.error(f"❌ Không tìm thấy file cấu hình tại: {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="⏳ Đang kết nối cơ sở dữ liệu mảng Zarr tổng hợp...")
def load_merged_data(processed_dir_cfg):
    """Mở file dữ liệu .zarr tổng hợp bằng cơ chế lazy-loading của Xarray."""
    zarr_path = os.path.join(BASE_DIR, processed_dir_cfg, "era5_merged.zarr")
    
    if not os.path.exists(zarr_path):
        st.error(f"❌ Không tìm thấy file dữ liệu tổng hợp tại: `{zarr_path}`. Hãy chạy trang DataLoader trước!")
        return None
        
    try:
        # Mở cấu trúc mảng nén Zarr
        ds = xr.open_dataset(zarr_path, engine="zarr")
        return ds
    except Exception as e:
        st.error(f"❌ Không thể đọc file Zarr. Chi tiết: {e}")
        return None


def main():
    st.set_page_config(page_title="Trực quan hóa dữ liệu ERA5", page_icon="📊", layout="wide")
    
    st.title("📊 Giao Diện Trực Quan Hóa Dữ Liệu Khí Tượng (ERA5)")
    st.markdown("Hệ thống bóc tách mảng không-thời gian từ tệp nén `.zarr` để cập nhật bản đồ nhiệt và trường vectơ gió.")

    cfg = load_config()
    if not cfg:
        st.stop()

    # Tải dữ liệu Zarr
    ds = load_merged_data(cfg["paths"].get("processed_data_dir", "data/processed/"))
    if ds is None:
        st.stop()

    # --- PHẦN 1: TRỤC THỜI GIAN (SLIDER) ---
    st.sidebar.header("⏱️ Trục Thời Gian Hệ Thống")
    
    # Lấy danh sách toàn bộ các mốc thời gian trong file Zarr
    time_steps = pd.to_datetime(ds['valid_time'].values)
    total_steps = len(time_steps)
    
    st.sidebar.info(f"📁 Tổng chuỗi thời gian: **{total_steps} mốc giờ**.")

    # Thanh trượt thời gian chính
    time_idx = st.select_slider(
        "🎚️ Kéo để duyệt và dịch chuyển mốc thời gian hiển thị bản đồ:",
        options=range(total_steps),
        format_func=lambda x: time_steps[x].strftime('%Y-%m-%d %H:%M'),
        key="time_slider"
    )
    
    selected_time = time_steps[time_idx]
    st.success(f"📌 Đang hiển thị dữ liệu tại mốc thời gian: **{selected_time.strftime('%d/%m/%Y lúc %H:%M')}**")

    # Trích xuất dữ liệu mảng (Slice) tại mốc thời gian được chọn và nạp vào RAM (.load())
    ds_slice = ds.isel(valid_time=time_idx).load() 

    # Đồng bộ hóa tên biến (hỗ trợ cả định dạng viết tắt và viết đầy đủ)
    t2m_key = "2m_temperature" if "2m_temperature" in ds_slice else "t2m"
    u10_key = "10m_u_component_of_wind" if "10m_u_component_of_wind" in ds_slice else "u10"
    v10_key = "10m_v_component_of_wind" if "10m_v_component_of_wind" in ds_slice else "v10"

    # --- PHẦN 2: THIẾT LẬP LAYOUT 2 CỘT BẢN ĐỒ ---
    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # CỘT 1: BẢN ĐỒ NHIỆT ĐỘ (HEATMAP / COLORMAP VIA MATPLOTLIB)
    # ---------------------------------------------------------
    with col1:
        st.subheader("🌡️ Bản đồ nhiệt độ bề mặt (2m Temperature)")
        
        if t2m_key in ds_slice:
            temp_data = ds_slice[t2m_key].values
            # Chuyển đổi tự động từ Kelvin sang độ C nếu cần
            if temp_data.max() > 150:
                temp_data = temp_data - 273.15
            
            lats = ds_slice['latitude'].values
            lons = ds_slice['longitude'].values

            # Khởi tạo biểu đồ đồ thị dạng nền tối của Streamlit
            fig, ax = plt.subplots(figsize=(8, 6.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            # Vẽ lưới màu sắc nhiệt độ liên tục
            im = ax.pcolormesh(
                lons, lats, temp_data, 
                cmap='turbo', 
                shading='auto'
            )
            
            # Cấu hình thanh Colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label('Nhiệt độ (°C)', color='white')
            cbar.ax.xaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.get_xticklabels(), color='white')

            ax.set_title(f"Nhiệt độ ngày {selected_time.strftime('%d/%m/%Y')}", color='white', fontsize=12)
            ax.set_xlabel('Kinh độ (Longitude)', color='white')
            ax.set_ylabel('Vĩ độ (Latitude)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ Không tìm thấy biến nhiệt độ (t2m/2m_temperature) trong tập dữ liệu.")

    # ---------------------------------------------------------
    # CỘT 2: BẢN ĐỒ GIÓ ĐỘNG (CORRECTED LEAFMAP CIRCLE MARKERS)
    # ---------------------------------------------------------
    with col2:
        st.subheader("💨 Bản đồ Trường Vector Gió (Leafmap Circle Markers)")

        if u10_key in ds_slice and v10_key in ds_slice:
            u_wind = ds_slice[u10_key].values
            v_wind = ds_slice[v10_key].values
            lats = ds_slice['latitude'].values
            lons = ds_slice['longitude'].values
            
            # Tính toán tốc độ gió tổng hợp
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)

            # Tính toán vị trí trung tâm bản đồ
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))
            
            m = leafmap.Map(
                center=[center_lat, center_lon], 
                zoom=6, 
                draw_control=False, 
                measure_control=False
            )
            
            # Sử dụng bản đồ nền tối để làm nổi bật các điểm dữ liệu khí tượng
            m.add_basemap("CartoDB.DarkMatter")

            # Tạo lưới tọa độ 2D
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Thuật toán Downsampling giảm mật độ điểm hiển thị tránh treo trình duyệt
            skip = max(1, int(len(lats) / 12))  
            
            sub_lat = lat_grid[::skip, ::skip].flatten()
            sub_lon = lon_grid[::skip, ::skip].flatten()
            sub_u = u_wind[::skip, ::skip].flatten()
            sub_v = v_wind[::skip, ::skip].flatten()
            sub_speed = wind_speed[::skip, ::skip].flatten()

            # --- KHỞI TẠO DATAFRAME THEO CHUẨN DOCUMENTATION LEAFMAP ---
            data_list = []
            for i in range(len(sub_lat)):
                if sub_speed[i] < 0.1:  # Lọc bỏ các vùng lặng gió
                    continue
                
                # Tạo chuỗi thông tin pop-up định dạng HTML hiển thị khi click
                popup_html = f"💨 <b>Tốc độ gió:</b> {sub_speed[i]:.2f} m/s <br> 🧭 <b>Thành phần:</b> U={sub_u[i]:.2f}, V={sub_v[i]:.2f}"
                
                # Tính toán bán kính (radius) tương ứng cho từng điểm tròn
                marker_radius = int(sub_speed[i] * 1.8) + 2

                data_list.append({
                    "lat": float(sub_lat[i]),
                    "lon": float(sub_lon[i]),
                    "popup": popup_html,
                    "radius": marker_radius}
                )

            if data_list:
                # Chuyển đổi danh sách thành DataFrame
                df_markers = pd.DataFrame(data_list)

                # Sử dụng hàm add_circle_markers chuẩn của Leafmap
                m.add_circle_markers_from_xy(
                    data=df_markers,
                    x = "lon",        # Tên cột vĩ độ trong DataFrame
                    y = "lat",         # Tên cột kinh độ trong DataFrame
                    radius="radius",       # Tên cột quy định bán kính động (tính năng nâng cao)
                    color="#00f2fe",       # Màu viền vòng tròn
                    fill_color="#4facfe",  # Màu nền vòng tròn
                    fill_opacity=0.6,
                    popup="popup"          # Tên cột chứa thông tin pop-up
                )

            # Đẩy bản đồ tương tác lên giao diện Streamlit
            m.to_streamlit(height=500)
            
            st.caption("ℹ️ *Chú giải:* Kích thước bán kính vòng tròn hiển thị tỷ lệ thuận với tốc độ gió (m/s) thực tế tại tọa độ đó.")
        else:
            st.warning("⚠️ Không tìm thấy thành phần gió u10/v10 trong tập dữ liệu.")

if __name__ == "__main__":
    main()