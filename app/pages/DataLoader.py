import os
import streamlit as st
import yaml
import cdsapi
import xarray as xr
import shutil
import warnings
import tempfile

# Tắt các cảnh báo hệ thống không cần thiết hiển thị ra giao diện terminal
warnings.filterwarnings('ignore', category=FutureWarning)

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config():
    """Đọc file cấu hình config.yaml từ thư mục config."""
    if not os.path.exists(CONFIG_PATH):
        st.error(f"❌ Không tìm thấy file cấu hình tại: {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_variables_to_long_names(short_vars):
    """Ánh xạ từ khóa viết tắt sang tên đầy đủ chuẩn mới của CDS API."""
    mapping = {
        "t2m": "2m_temperature",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "msl": "mean_sea_level_pressure",
        "q": "specific_humidity",
        "tp": "total_precipitation",
        "r": "relative_humidity",
    }
    return [mapping.get(v, v) for v in short_vars]


def sanitize_dataset(ds):
    """Xử lý dọn dẹp Dataset để loại bỏ lỗi Serialization và đồng bộ kiểu dữ liệu 'object'."""
    for var in ds.data_vars:
        if ds[var].dtype == object:
            ds[var] = ds[var].astype(str)
            
    for coord in ds.coords:
        if ds[coord].dtype == object:
            ds[coord] = ds[coord].astype(str)
            
    return ds


def download_and_convert_pipeline(cfg):
    """Pipeline xử lý dữ liệu tích hợp MỚI:
    Tải dữ liệu thô (.nc) độc lập theo từng BIẾN SỐ cho toàn bộ các năm -> 
    Gộp các file biến lại bằng combine='by_coords' -> Xuất sang .zarr tổng hợp duy nhất.
    """
    try:
        c = cdsapi.Client()
    except Exception as e:
        yield {
            "status": "error",
            "message": f"Không thể khởi tạo CDS API Client. Chi tiết: {e}",
        }
        return

    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    # 1. Chuẩn bị các tham số thời gian và không gian
    start_year = int(data_cfg["start_year"])
    end_year = int(data_cfg["end_year"])
    years = [str(y) for y in range(start_year, end_year + 1)]
    
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{t:02d}:00" for t in range(0, 24, int(data_cfg["time_step_hours"]))]
    
    crop = data_cfg["crop_region"]
    area_param = [crop["lat_max"], crop["lon_min"], crop["lat_min"], crop["lon_max"]]

    # Chuẩn hóa danh sách biến (Chuyển short_name sang long_name để tránh lỗi Ambiguous của MARS)
    short_variables = data_cfg["variables"]
    long_variables = map_variables_to_long_names(short_variables)
    total_vars = len(long_variables)

    # Thư mục đầu ra và khởi tạo thư mục tạm (temp_dir) an toàn
    processed_dir = os.path.join(BASE_DIR, paths_cfg.get("processed_data_dir", "data/processed/"))
    os.makedirs(processed_dir, exist_ok=True)
    
    final_zarr_path = os.path.join(processed_dir, "era5_merged.zarr")
    
    # Tạo một thư mục tạm thời trong data/raw/ để chứa các file biến .nc thô
    raw_dir = os.path.join(BASE_DIR, paths_cfg.get("raw_data_dir", "data/raw/"))
    os.makedirs(raw_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=raw_dir)

    downloaded_files = []

    # 2. VÒNG LẶP TẢI FILE THEO TỪNG BIẾN SỐ (Theo logic mới của bạn)
    for idx, var in enumerate(long_variables):
        # Tính toán tiến trình hiển thị UI
        progress_pct = int((idx / (total_vars + 1)) * 100)
        short_name = short_variables[idx]

        yield {
            "status": "downloading",
            "progress": progress_pct,
            "message": f"⏳ [Biến {short_name}] Đang yêu cầu tải dữ liệu chuỗi năm từ Copernicus...",
        }

        target_file = os.path.join(temp_dir, f"era5_{short_name}.nc")

        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'data_format': 'netcdf',  # Cập nhật từ 'format' sang 'data_format' tránh lỗi cảnh báo CDS
                    'variable': [var],
                    'year': years,
                    'month': months,
                    'day': days,
                    'time': times,
                    'area': area_param,
                },
                target_file
            )
            downloaded_files.append(target_file)
            
            yield {
                "status": "success_year", # Tận dụng trạng thái log thành công cũ
                "message": f"✅ Hoàn thành tải dữ liệu thô cho biến: `{short_name}`",
            }

        except Exception as e:
            # Dọn dẹp thư mục tạm trước khi thoát nếu lỗi
            shutil.rmtree(temp_dir, ignore_errors=True)
            yield {
                "status": "error",
                "message": f"❌ Lỗi khi tải dữ liệu biến {short_name}: {e}",
            }
            return

    # 3. GIAI ĐOẠN GỘP FILE THEO TỌA ĐỘ VÀ ĐỒNG BỘ CHUNK DỮ LIỆU
    merge_progress_pct = int((total_vars / (total_vars + 1)) * 100)
    yield {
        "status": "merging",
        "progress": merge_progress_pct,
        "message": f"🔀 Đang tiến hành kết hợp các biến (`combine='by_coords'`) và nén mảng thành một tập `.zarr` duy nhất...",
    }

    try:
        if os.path.exists(final_zarr_path):
            shutil.rmtree(final_zarr_path)

        # Mở và kết hợp đồng thời toàn bộ các file biến .nc tạm thời
        with xr.open_mfdataset(downloaded_files, combine='by_coords') as merged_dataset:
            
            # Khử lỗi dask array dtype=object để tránh SerializationWarning
            merged_dataset = sanitize_dataset(merged_dataset)
            
            # Sửa lỗi Uniform Chunk Sizes: Định nghĩa kích thước khối đồng đều an toàn cho cấu trúc Zarr
            target_chunks = {
                "valid_time": -1, 
                "latitude": "auto", 
                "longitude": "auto"
            }
            rechunked_dataset = merged_dataset.chunk(target_chunks)
            
            # Xuất ra thư mục .zarr duy nhất chuẩn V2 ổn định
            rechunked_dataset.to_zarr(final_zarr_path, mode='w', consolidated=True, zarr_version=2)

        yield {
            "status": "completed",
            "progress": 100,
            "message": f"🎉 Quá trình lưu và đồng bộ hoàn tất! Tập dữ liệu đã sẵn sàng tại: `{paths_cfg.get('processed_data_dir')}era5_merged.zarr`",
        }

    except Exception as e:
        yield {
            "status": "error",
            "message": f"❌ Lỗi trong quá trình gộp hoặc lưu file: {e}",
        }
    finally:
        # Xóa bỏ hoàn toàn thư mục chứa các file .nc tạm thời sau khi xử lý xong
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    st.set_page_config(page_title="Tải dữ liệu khí tượng", page_icon="📥", layout="wide")

    st.title("📥 Trình Tải & Tiền Xử Lý Dữ Liệu Tự Động (ERA5)")
    st.markdown(
        "Hệ thống tải dữ liệu bóc tách độc lập theo từng biến số và tự động gộp thành cấu trúc tập mảng `.zarr` duy nhất bằng phương pháp liên kết tọa độ địa lý."
    )

    cfg = load_config()
    if not cfg:
        st.stop()

    with st.expander("📊 Kiểm tra cấu hình hiện tại trước khi kích hoạt (config.yaml)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Tập Dữ Liệu:** `reanalysis-era5-single-levels`")
            st.markdown(f"**Loại Sản Phẩm:** `reanalysis`")
            st.markdown(f"**Định Dạng Gốc:** `NETCDF`")
        with col2:
            st.markdown(f"**Khoảng Thời Gian:** {cfg['data']['start_year']} ➔ {cfg['data']['end_year']}")
            st.markdown(f"**Tần Suất Lấy Mẫu:** {cfg['data']['time_step_hours']} tiếng / lần")
            st.markdown(f"**Biến Xử Lý:** `{', '.join(cfg['data']['variables'])}`")
        with col3:
            crop = cfg["data"]["crop_region"]
            st.markdown("**Giới Hạn Tọa Độ:**")
            st.caption(f"Vĩ độ (Lat): [{crop['lat_min']}, {crop['lat_max']}]")
            st.caption(f"Kinh độ (Lon): [{crop['lon_min']}, {crop['lon_max']}]")

    st.markdown("---")

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    start_button = st.button(
        "🚀 Khởi chạy tiến trình đồng bộ dữ liệu",
        type="primary",
        disabled=st.session_state.is_processing,
    )

    if start_button:
        st.session_state.is_processing = True

        progress_bar = st.progress(0)
        status_message = st.empty()

        st.markdown("#### 📝 Nhật ký ghi nhận hệ thống (Logs):")
        log_container = st.container()

        events = download_and_convert_pipeline(cfg)

        for event in events:
            if "progress" in event:
                progress_bar.progress(event["progress"])

            if event["status"] in ["downloading", "converting", "merging"]:
                status_message.info(event["message"])
            elif event["status"] == "success_year":
                log_container.success(event["message"])
            elif event["status"] == "error":
                status_message.error(event["message"])
                log_container.error(event["message"])
                st.session_state.is_processing = False
                st.stop()
            elif event["status"] == "completed":
                status_message.success(event["message"])
                st.balloons()

        st.session_state.is_processing = False


if __name__ == "__main__":
    main()