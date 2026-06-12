import os
import streamlit as st
import yaml
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sử dụng timm (Torch Image Models) để khởi tạo backbone Swin Transformer V2 chuẩn chỉ
import timm

# Xác định đường dẫn gốc và file cấu hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config():
    """Đọc tệp cấu hình hệ thống config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        st.error(f"❌ Không tìm thấy tệp cấu hình tại: {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------
# 1. PHÂN ĐOẠN KHỞI TẠO BỘ DỮ LIỆU PYTORCH DATASET (WEATHER DATASET)
# --------------------------------------------------------------------
class ERA5SwinV2Dataset(Dataset):
    """Dataset xây dựng cấu trúc cặp dữ liệu [X (Quá khứ), y (Tương lai)] 
    để huấn luyện mô hình dự báo thời tiết chuỗi thời gian."""
    def __init__(self, zarr_path, feature_vars, target_vars, sequence_length=1):
        super().__init__()
        # Mở lazy-loading tập Zarr tổng hợp
        self.ds = xr.open_dataset(zarr_path, engine="zarr")
        self.seq_len = sequence_length
        
        # Trích xuất mảng numpy từ các biến đặc trưng đầu vào và đầu ra
        self.X_data = np.stack([self.ds[v].values for v in feature_vars], axis=1) # Shape: (Time, C, Lat, Lon)
        self.y_data = np.stack([self.ds[v].values for v in target_vars], axis=1)   # Shape: (Time, C, Lat, Lon)
        
        # Chuẩn hóa Min-Max scale cục bộ đơn giản để SwinV2 hội tụ nhanh hơn
        self.X_min, self.X_max = self.X_data.min(), self.X_data.max()
        self.X_data = (self.X_data - self.X_min) / (self.X_max - self.X_min + 1e-6)
        
        self.y_min, self.y_max = self.y_data.min(), self.y_data.max()
        self.y_data = (self.y_data - self.y_min) / (self.y_max - self.y_min + 1e-6)

    def __len__(self):
        return len(self.X_data) - self.seq_len

    def __getitem__(self, idx):
        # Lấy mốc thời gian t làm đầu vào để dự báo mốc thời gian t+1
        x = self.X_data[idx]          # Shape: (C_in, Lat, Lon)
        y = self.y_data[idx + self.seq_len] # Shape: (C_out, Lat, Lon)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# --------------------------------------------------------------------
# 2. KIẾN TRÚC MÔ HÌNH SWINV2 CHO BÀI TOÁN KHÍ TƯỢNG (DỰ BÁO DẠNG LƯỚI)
# --------------------------------------------------------------------
class SwinV2Forecaster(nn.Module):
    """Mô hình thích ứng Swin Transformer V2 nhận lưới khí tượng làm ảnh đầu vào 
    và giải nén (Upsample) trở lại mảng dự báo thời tiết cùng kích thước."""
    def __init__(self, in_channels, out_channels, img_size=(64, 64)):
        super().__init__()
        
        # Khởi tạo backbone SwinV2 Tiny làm bộ trích xuất đặc trưng (Feature Extractor)
        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256', 
            pretrained=False, 
            in_chans=in_channels,
            num_classes=0 # Loại bỏ tầng phân loại gốc (Dense Layer)
        )
        
        # Nhận diện kích thước chiều ẩn đầu ra của backbone tinh chỉnh
        num_features = self.backbone.num_features # Mặc định với dòng tiny là 768
        
        # Tầng giải mã không gian (Decoder Spatial Upsampling) chuyển đổi vector đặc trưng về lưới tọa độ (Lat, Lon)
        # Sử dụng nội suy PixelShuffle hoặc ConvTranspose2d thích ứng
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 256 * (img_size[0] // 4) * (img_size[1] // 4)),
            nn.GELU(),
            nn.Unflatten(1, (256, img_size[0] // 4, img_size[1] // 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # Khôi phục tỷ lệ x2
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Khôi phục tỷ lệ x4 (đạt gốc img_size)
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Đưa về số lượng biến đầu ra đầu ra mục tiêu
        )

    def forward(self, x):
        # x shape: (Batch, In_Channels, Lat, Lon)
        features = self.backbone(x) # Trích xuất cấu hình toàn cục (Batch, num_features)
        output = self.decoder(features) # Giải nén cấu trúc ảnh khí tượng (Batch, Out_Channels, Lat, Lon)
        return output


# --------------------------------------------------------------------
# 3. HÀM LOSS CHUYÊN DỤNG CHO DỮ LIỆU VĨ ĐỘ (LATITUDE WEIGHTED RMSE)
# --------------------------------------------------------------------
class LatitudeWeightedRMSE(nn.Module):
    """Hàm tổn thất tính trọng số theo vĩ độ. Do bề mặt Trái Đất co hẹp ở hai cực, 
    các điểm lưới vùng vĩ độ cao cần nhân với hệ số cos(vĩ độ) để tính toán chuẩn xác."""
    def __init__(self, lats):
        super().__init__()
        # Tính trọng số vĩ độ tính theo radian
        weights = np.cos(np.radians(lats))
        weights = weights / np.mean(weights)
        # Chuyển đổi tensor tương thích thiết bị phần cứng
        self.weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1) # Shape: (Lat, 1)

    def forward(self, pred, target):
        # pred, target shape: (Batch, Channel, Lat, Lon)
        self.weights = self.weights.to(pred.device)
        # Bình phương sai số nhân với ma trận trọng số vĩ độ
        squared_error = (pred - target) ** 2
        weighted_error = squared_error * self.weights.unsqueeze(0).unsqueeze(0)
        return torch.sqrt(torch.mean(weighted_error) + 1e-6)


# --------------------------------------------------------------------
# 4. GIAO DIỆN CHÍNH TRANG ĐIỀU KHIỂN TRAIN
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Huấn luyện mô hình SwinV2", page_icon="🏋️", layout="wide")
    st.title("🏋️ Trung Tâm Huấn Luyện Mô Hình Dự Báo (SwinV2 - ERA5)")
    st.markdown("Trình khởi chạy vòng lặp tối ưu hóa siêu tham số mảng Deep Learning sử dụng cấu trúc tăng tốc phần cứng PyTorch.")

    cfg = load_config()
    if not cfg:
        st.stop()

    # Kiểm tra thiết bị tăng tốc (CUDA Core / MPS / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.header("🖥️ Tài nguyên phần cứng")
    if device.type == "cuda":
        st.sidebar.success(f"🚀 Thiết bị: **GPU NVidia (CUDA Active)**\nDevice Name: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("⚠️ Thiết bị: **CPU Chậm** (Khuyến nghị bật CUDA để tăng tốc độ tính toán khối Swin)")

    # Kiểm tra file Zarr tổng hợp đầu vào
    zarr_dir_rel = cfg["paths"].get("processed_data_dir", "data/processed/")
    zarr_path = os.path.join(BASE_DIR, zarr_dir_rel, "era5_merged.zarr")

    if not os.path.exists(zarr_path):
        st.error(f"❌ Không tìm thấy cơ sở dữ liệu Zarr tại `{zarr_path}`. Hãy chạy tiền xử lý ở mục DataLoader trước khi kích hoạt huấn luyện!")
        st.stop()

    # --- KHU VỰC CẤU HÌNH SIÊU THAM SỐ NHANH ---
    st.markdown("### 🎛️ Thiết lập Siêu tham số nhanh (Hyperparameters)")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        epochs = st.number_input("Số chu kỳ (Epochs):", min_value=1, max_value=500, value=int(cfg.get("train", {}).get("epochs", 10)))
    with col_p2:
        batch_size = st.number_input("Kích thước lô (Batch Size):", min_value=1, max_value=256, value=int(cfg.get("train", {}).get("batch_size", 4)))
    with col_p3:
        lr = st.number_input("Tốc độ học (Learning Rate):", min_value=1e-6, max_value=1e-1, value=float(cfg.get("train", {}).get("optimizer", {}).get("lr", 0.0005)), format="%.6f")
    with col_p4:
        loss_selection = st.selectbox("Hàm tổn thất (Loss):", ["Latitude-Weighted RMSE", "Standard MSE"])

    # Đọc cấu hình các biến từ yaml
    feature_vars = cfg["data"]["variables"]
    target_vars = cfg["data"]["variables"] # Mặc định dự báo chính các biến này ở mốc tương lai

    # Trình kích hoạt huấn luyện
    if "training_active" not in st.session_state:
        st.session_state.training_active = False

    btn_train = st.button("🚀 Kích hoạt vòng lặp tối ưu SwinV2", type="primary", disabled=st.session_state.training_active)

    if btn_train:
        st.session_state.training_active = True
        
        # Khởi tạo Trình thu thập dữ liệu (Dataloader)
        status_txt = st.empty()
        status_txt.info("⏳ Đang phân mảnh và tải cơ sở dữ liệu mảng Zarr lên RAM...")
        
        try:
            dataset = ERA5SwinV2Dataset(zarr_path, feature_vars, target_vars, sequence_length=1)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            # Lấy thông tin kích thước không gian lưới từ tệp dữ liệu thực tế
            img_shape = (len(dataset.ds['latitude']), len(dataset.ds['longitude']))
            lats_array = dataset.ds['latitude'].values
            
            # Khởi tạo mạng nơ-ron mạng tích hợp
            model = SwinV2Forecaster(in_channels=len(feature_vars), out_channels=len(target_vars), img_size=img_shape)
            model = model.to(device)
            
            # Khởi tạo hàm loss và tối ưu hóa adamw chuẩn cấu trúc transformer
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            criterion = LatitudeWeightedRMSE(lats_array) if loss_selection == "Latitude-Weighted RMSE" else nn.MSELoss()
            
            status_txt.success("✅ Khởi tạo hạ tầng mạng thành công! Bắt đầu chu kỳ Gradient Descent...")
            
            # Khởi tạo các khung hiển thị đồ thị động thời gian thực (Real-time tracking)
            progress_bar = st.progress(0)
            col_metric1, col_metric2 = st.columns(2)
            metric_epoch = col_metric1.empty()
            metric_loss = col_metric2.empty()
            
            plot_spot = st.empty() # Khung render đồ thị động
            
            loss_history = []
            
            # --- VÒNG LẶP HUẤN LUYỆN CHÍNH (TRAINING LOOP) ---
            for epoch in range(epochs):
                model.train()
                epoch_losses = []
                
                for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                # Tính toán loss trung bình của epoch
                avg_epoch_loss = np.mean(epoch_losses)
                loss_history.append(avg_epoch_loss)
                
                # Cập nhật thông số hiển thị ra màn hình UI
                progress_bar.progress(int(((epoch + 1) / epochs) * 100))
                metric_epoch.metric("Chu kỳ hiện tại (Epoch)", f"{epoch + 1} / {epochs}")
                metric_loss.metric("Hàm tổn thất trung bình (Loss)", f"{avg_epoch_loss:.6f}")
                
                # Cập nhật đồ thị Loss Curve thời gian thực bằng Matplotlib
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                
                ax.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='#00f2fe', linewidth=2)
                ax.set_title("Biểu đồ suy giảm Hàm tổn thất (SwinV2 Loss Curve)", color='white')
                ax.set_xlabel("Epoch", color='white')
                ax.set_ylabel("Loss Value", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, linestyle='--', alpha=0.2)
                
                plot_spot.pyplot(fig)
                plt.close(fig)
                
                # Giảm tải luồng tránh treo UI Streamlit
                time.sleep(0.05)
                
            # Lưu trữ trọng số mô hình đã huấn luyện xong (Weight Checkpoint)
            model_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, "swinv2_era5_best.pt")
            
            torch.save(model.state_dict(), model_save_path)
            status_txt.success(f"🎉 Huấn luyện hoàn tất xuất sắc! File trọng số mô hình tối ưu đã lưu tại: `models/swinv2_era5_best.pt`")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Tiến trình huấn luyện gặp sự cố nghiêm trọng: {e}")
        finally:
            st.session_state.training_active = False


if __name__ == "__main__":
    main()