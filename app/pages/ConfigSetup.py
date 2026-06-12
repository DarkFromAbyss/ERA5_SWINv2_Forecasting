import os
import streamlit as st
import yaml
from components.sidebar import render_common_sidebar

# 1. Cấu hình trang bắt buộc ở đầu file
st.set_page_config(
    page_title="Cấu hình hệ thống SwinV2", page_icon="⚙️", layout="wide"
)

# 2. Gọi sidebar dùng chung để hiển thị các thành phần bổ sung bên dưới menu điều hướng
render_common_sidebar()

# 3. Phần Right Side - Giao diện chính của trang Config
st.title("🛠️ Cấu hình Hệ thống (config.yaml)")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config_data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi lưu file: {e}")
        return False


config = load_config()

if config:
    with st.form("config_form"):
       # ---------------------------------------------------------------------
        # 1. GLOBAL & PATH SETTINGS
        # ---------------------------------------------------------------------
        st.header("1. Cấu hình chung & Đường dẫn (Global & Paths)")
        col1, col2 = st.columns(2)
        with col1:
            config["global"]["project_name"] = st.text_input(
                "Tên dự án", config["global"].get("project_name", "")
            )
            config["global"]["device"] = st.selectbox(
                "Thiết bị tính toán (Device)",
                ["cuda", "cpu"],
                index=0 if config["global"].get("device") == "cuda" else 1,
            )
        with col2:
            config["global"]["seed"] = st.number_input(
                "Random Seed", value=int(config["global"].get("seed", 42))
            )
            config["global"]["verbose"] = st.checkbox(
                "Chế độ hiển thị chi tiết (Verbose)",
                value=bool(config["global"].get("verbose", True)),
            )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 2. DATA PROCESSING CONFIGURATION (ERA5)
        # ---------------------------------------------------------------------
        st.header("2. Cấu hình Dữ liệu ERA5")
        
        
        st.subheader("Thông tin định danh Tập dữ liệu trên Copernicus (CDS API)")
        col_api1, col_api2 = st.columns(2)
        with col_api1:
            config["data"]["dataset_name"] = st.text_input(
                "Tên tập dữ liệu (Dataset Name)",
                value=config["data"].get("dataset_name", "reanalysis-era5-single-levels"),
                help="Ví dụ: 'reanalysis-era5-single-levels' cho dữ liệu mặt đất hoặc 'reanalysis-era5-pressure-levels' cho dữ liệu tầng khí quyển."
            )
        with col_api2:
            config["data"]["product_type"] = st.text_input(
                "Loại sản phẩm (Product Type)",
                value=config["data"].get("product_type", "reanalysis"),
                help="Mặc định đối với dữ liệu ERA5 lịch sử phân tích lại là 'reanalysis'."
            )
        
        # --- PHẦN CẬP NHẬT: Chọn khoảng thời gian năm ---
        st.subheader("Khoảng thời gian thu thập dữ liệu")
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            config["data"]["start_year"] = st.number_input(
                "Năm bắt đầu (Start Year)",
                value=int(config["data"].get("start_year", 2020)),
                min_value=1940,  # ERA5 có dữ liệu từ khá sớm
                max_value=2026,
                step=1,
            )
        with col_y2:
            config["data"]["end_year"] = st.number_input(
                "Năm kết thúc (End Year)",
                value=int(config["data"].get("end_year", 2025)),
                min_value=1940,
                max_value=2026,
                step=1,
            )

        # Kiểm tra logic năm hợp lệ
        if config["data"]["start_year"] > config["data"]["end_year"]:
            st.error("Cảnh báo: Năm bắt đầu không được lớn hơn năm kết thúc!")
        # ------------------------------------------------

        st.subheader("Định dạng dữ liệu đầu vào (Format)")
        current_format = config["data"].get("format", "netcdf")
        format_options = ["netcdf", "grib"]
        config["data"]["format"] = st.selectbox(
            "Chọn định dạng file tải về từ CDS API",
            format_options,
            index=format_options.index(current_format) if current_format in format_options else 0,
            help="NetCDF (.nc) thường dễ xử lý hơn với xarray trên Python, GRIB (.grib) là định dạng chuẩn khí tượng quốc tế."
        )

        # Chọn các biến khí tượng
        all_available_vars = ["t2m", "u10", "v10", "msl", "q", "tp", "r"]
        current_vars = config["data"].get("variables", [])
        st.subheader("Các biến thời tiết sử dụng:")
        selected_vars = st.multiselect(
            "Chọn biến (t2m: Nhiệt độ, u10/v10: Gió, msl: Áp suất, q: Độ ẩm...)",
            all_available_vars,
            default=current_vars,
        )
        config["data"]["variables"] = selected_vars

        # Tham số lưới & Chuỗi thời gian
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            config["data"]["input_steps"] = st.number_input(
                "Số bước thời gian quá khứ (Input Steps)",
                value=int(config["data"].get("input_steps", 4)),
                min_value=1,
            )
            config["data"]["normalization"] = st.selectbox(
                "Phương pháp chuẩn hóa",
                ["z-score", "min-max"],
                index=0 if config["data"].get("normalization") == "z-score" else 1,
            )
        with col_d2:
            config["data"]["predict_steps"] = st.number_input(
                "Số bước dự đoán tương lai (Predict Steps)",
                value=int(config["data"].get("predict_steps", 1)),
                min_value=1,
            )
            config["data"]["val_split"] = st.slider(
                "Tỷ lệ tập Validation",
                0.05,
                0.3,
                value=float(config["data"].get("val_split", 0.1)),
            )
        with col_d3:
            config["data"]["time_step_hours"] = st.number_input(
                "Khoảng cách giữa các bước (Giờ)",
                value=int(config["data"].get("time_step_hours", 6)),
                min_value=1,
            )
            config["data"]["test_split"] = st.slider(
                "Tỷ lệ tập Test",
                0.05,
                0.3,
                value=float(config["data"].get("test_split", 0.1)),
            )

        # Phạm vi địa lý hiển thị cố định
        st.subheader("🌐 Phạm vi địa lý (Spatial Coverage)")
        crop = config["data"].get("crop_region", {})
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            crop["lat_max"] = st.number_input(
                "Vĩ độ Max (Bắc)",
                value=float(crop.get("lat_max", 26.0)),
                min_value=-90.0,
                max_value=90.0,
            )
        with col_c2:
            crop["lat_min"] = st.number_input(
                "Vĩ độ Min (Nam)",
                value=float(crop.get("lat_min", 8.0)),
                min_value=-90.0,
                max_value=90.0,
            )
        with col_c3:
            crop["lon_min"] = st.number_input(
                "Kinh độ Min (Tây)",
                value=float(crop.get("lon_min", 102.0)),
                min_value=-180.0,
                max_value=180.0,
            )
        with col_c4:
            crop["lon_max"] = st.number_input(
                "Kinh độ Max (Đông)",
                value=float(crop.get("lon_max", 110.0)),
                min_value=-180.0,
                max_value=180.0,
            )

        if crop["lat_min"] >= crop["lat_max"]:
            st.error("Lỗi: Vĩ độ Min phải nhỏ hơn Vĩ độ Max!")
        if crop["lon_min"] >= crop["lon_max"]:
            st.error("Lỗi: Kinh độ Min phải nhỏ hơn Kinh độ Max!")

        config["data"]["crop_region"] = crop
        st.markdown("---")

        # ---------------------------------------------------------------------
        # 3. MODEL ARCHITECTURE (SWIN TRANSFORMER V2)
        # ---------------------------------------------------------------------
        st.header("3. Kiến trúc mô hình Swin Transformer V2")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            img_h = st.number_input(
                "Kích thước lưới Grid - Chiều cao (H)",
                value=int(config["model"].get("img_size", [64, 64])[0]),
            )
            img_w = st.number_input(
                "Kích thước lưới Grid - Chiều rộng (W)",
                value=int(config["model"].get("img_size", [64, 64])[1]),
            )
            config["model"]["img_size"] = [img_h, img_w]

            config["model"]["patch_size"] = st.selectbox(
                "Kích thước Patch (Ví dụ: 4x4 vừng lưới)",
                [2, 4, 8],
                index=[2, 4, 8].index(config["model"].get("patch_size", 4)),
            )
            config["model"]["window_size"] = st.number_input(
                "Kích thước cửa sổ Attention (Window Size)",
                value=int(config["model"].get("window_size", 8)),
            )
        with col_m2:
            config["model"]["embed_dim"] = st.number_input(
                "Số chiều Embedding gốc (embed_dim)",
                value=int(config["model"].get("embed_dim", 128)),
            )
            config["model"]["in_chans"] = st.number_input(
                "Số lượng kênh đầu vào (in_chans)",
                value=int(config["model"].get("in_chans", 20)),
                help="Tính toán tự động: (Số biến mặt đất + số biến tầng cao) * input_steps",
            )
            config["model"]["out_chans"] = st.number_input(
                "Số lượng kênh đầu ra (out_chans)",
                value=int(config["model"].get("out_chans", 5)),
            )
            config["model"]["drop_rate"] = st.slider(
                "Dropout Rate",
                0.0,
                0.5,
                value=float(config["model"].get("drop_rate", 0.1)),
            )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 4. TRAINING HYPERPARAMETERS
        # ---------------------------------------------------------------------
        st.header("4. Tham số Huấn luyện (Training)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            config["train"]["batch_size"] = st.number_input(
                "Batch Size", value=int(config["train"].get("batch_size", 16))
            )
            config["train"]["epochs"] = st.number_input(
                "Số Epochs", value=int(config["train"].get("epochs", 50))
            )
            config["train"]["loss_function"] = st.text_input(
                "Hàm Loss chuyên dụng",
                config["train"].get("loss_function", "LatitudeWeightedRMSE"),
            )
        with col_t2:
            config["train"]["optimizer"]["lr"] = st.number_input(
                "Learning Rate (Tốc độ học)",
                value=float(config["train"]["optimizer"].get("lr", 0.0005)),
                format="%.6f",
            )
            config["train"]["optimizer"]["weight_decay"] = st.number_input(
                "Weight Decay",
                value=float(config["train"]["optimizer"].get("weight_decay", 0.05)),
                format="%.4f",
            )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # BUTTON SAVE
        # ---------------------------------------------------------------------
        submitted = st.form_submit_with_clicks = st.form_submit_button(
            "💾 Lưu Cấu Hình Khung Hệ Thống"
        )
        if submitted:
            success = save_config(config)
            if success:
                st.success(
                    "🎉 Cấu hình đã được cập nhật thành công vào file `config/config.yaml`!"
                )
                st.balloons()