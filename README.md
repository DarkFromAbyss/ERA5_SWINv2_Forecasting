# ERA5_SWINv2_Forecasting

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 1. Giới thiệu đề tài & Mục đích Nghiên cứu
Dự báo thời tiết chính xác luôn là một thách thức lớn do tính chất hỗn loạn và phi tuyến tính của khí quyển. Các mô hình dự báo thời tiết số trị (NWP) truyền thống, mặc dù mạnh mẽ, nhưng lại đòi hỏi năng lực tính toán khổng lồ và thời gian chạy mô phỏng dài. 

Nghiên cứu khoa học này được thực hiện nhằm mục đích khám phá và đánh giá khả năng của các mô hình học sâu (Deep Learning), đặc biệt là kiến trúc **Swin Transformer V2 (Shifted Window Vision Transformer)**, trong việc học các biểu diễn không gian - thời gian từ dữ liệu khí tượng. Bằng cách tiếp cận theo hướng Data-driven, dự án kỳ vọng tạo ra một 파ipeline dự báo với độ trễ thấp hơn, tối ưu hóa quá trình tính toán mà vẫn đảm bảo được độ chính xác khi so sánh với các hệ thống dự báo truyền thống. Tọa độ được tập trung nghiên cứu (Vĩ độ `23.5` đến `8.5`, Kinh độ `102` đến `110`) bao quát trọn vẹn khu vực lãnh thổ và vùng biển Việt Nam, mang tính ứng dụng thực tiễn cao cho quốc gia.

## 2. Tập dữ liệu ERA5
Dự án sử dụng **ERA5** - bộ dữ liệu tái phân tích (reanalysis) thế hệ thứ 5 toàn cầu về khí hậu và thời tiết của Trung tâm Dự báo Thời tiết Trung hạn Châu Âu (ECMWF).

### Cấu hình Dữ liệu (Dựa trên `config.yaml`)
* **Giai đoạn thu thập:** 2023 - 2025.
* **Phạm vi không gian (Bounding Box):** `[23.5, 102, 8.5, 110]` (Khu vực Việt Nam).
* **Bước thời gian (Step Hours):** `6h`. Việc lấy mẫu mỗi 6 giờ là sự cân bằng tối ưu giữa việc giữ lại các biến động thời tiết quan trọng (như chu kỳ nhiệt độ ngày/đêm) và giới hạn về tài nguyên tính toán/lưu trữ, giúp mô hình học được chuỗi chu kỳ (Sequence) hiệu quả.

### Các biến Khí tượng (Variables)
Nghiên cứu tập trung vào 5 biến bề mặt (Single Level) có tính quyết định đến hình thái thời tiết:
1.  **Nhiệt độ 2m (2m_temperature - `t2m`):** Biến số cơ bản nhất phản ánh điều kiện nhiệt độ bề mặt, ảnh hưởng trực tiếp đến đời sống và bốc hơi nước.
2.  **Thành phần gió U ở 10m (10m_u_component_of_wind - `u10`):** Đại diện cho vận tốc gió theo trục Đông - Tây.
3.  **Thành phần gió V ở 10m (10m_v_component_of_wind - `v10`):** Đại diện cho vận tốc gió theo trục Bắc - Nam. *(Kết hợp U và V giúp mô hình nắm bắt được hướng gió, sự hình thành bão và hoàn lưu gió mùa).*
4.  **Áp suất bề mặt (surface_pressure - `sp`):** Chỉ số quan trọng để theo dõi các vùng áp thấp/áp cao, tiền đề để dự báo mưa và bão.
5.  **Tổng lượng mưa (total_precipitation - `tp`):** Biến số phức tạp nhất nhưng có giá trị cực cao trong việc cảnh báo thiên tai và lũ lụt.

## 3. Kiến trúc Mô hình: Swin Transformer V2
Dự án triển khai huấn luyện dựa trên kiến trúc **SwinV2**, một bản nâng cấp giải quyết được các vấn đề về phân giải cao (high-resolution) và tính ổn định khi mở rộng quy mô (scaling up) của Vision Transformer truyền thống. Nhờ cơ chế *Shifted Window Attention*, mô hình có khả năng nắm bắt được cả các hình thái thời tiết cục bộ (nhỏ) lẫn các dải nhiễu động diện rộng.

Dự án hỗ trợ 3 biến thể chính để người dùng có thể linh hoạt cấu hình tùy vào tài nguyên phần cứng (tham chiếu kích thước ảnh đầu vào là $256 \times 256$):

| Biến thể (Variant) | Số lượng tham số (Parameters) | Ứng dụng thực nghiệm |
| :--- | :--- | :--- |
| **SwinV2-Small** | ~ 50 Triệu | Môi trường hạn chế tài nguyên, chạy thử nghiệm nhanh (Baseline). |
| **SwinV2-Base** | ~ 88 Triệu | Cấu hình mặc định (`swinv2_base_window16_256`). Cân bằng tốt giữa độ chính xác và tốc độ huấn luyện. |
| **SwinV2-Large** | ~ 197 Triệu | Yêu cầu GPU VRAM lớn. Phù hợp cho việc fine-tune để tối đa hóa độ chính xác dự báo. |

## 4. Hướng dẫn Cài đặt & Sử dụng

### 4.1. Đăng ký tài khoản và API Key CDS
Để tải dữ liệu ERA5, bạn cần có tài khoản Copernicus Climate Data Store (CDS).
1.  Tạo tài khoản tại: [CDS Registration](https://cds.climate.copernicus.eu/user/register)
2.  Đăng nhập và truy cập trang Profile để lấy **API Key**.
3.  Tạo một file `.cdsapirc` ở thư mục home (ví dụ: `C:\Users\Username\.cdsapirc` trên Windows hoặc `~/.cdsapirc` trên Linux/Mac) với nội dung:
    ```text
    url: [https://cds.climate.copernicus.eu/api/v2](https://cds.climate.copernicus.eu/api/v2)
    key: <UID>:<API_KEY_CỦA_BẠN>
    ```
4.  **Quan trọng:** Trước khi chạy script tải dữ liệu, bạn phải truy cập vào trang [ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), cuộn xuống phần **Terms of use** và nhấn "Accept Terms".

### 4.2. Clone Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
```
