# 📈 House Price Prediction Project

Dự án xây dựng hệ thống trí tuệ nhân tạo để dự báo giá bất động sản dựa trên các đặc trưng của ngôi nhà (diện tích, vị trí, số phòng ngủ, 
tiện ích xung quanh...). Hệ thống bao gồm pipeline xử lý dữ liệu tự động, mô hình học máy và giao diện web để người dùng tra cứu giá nhà trực quan.

---

## 📂 Cấu trúc Thư mục (Project Structure)

```text
root/
├── api/                        # [BACKEND] FastAPI Application
│   ├── api_v1/                 # Các Routes xử lý yêu cầu (Predict, Auth,...)
│   ├── core/                   # Config, Security (JWT), Database Connection
│   ├── models/                 # Database Schemas (SQLAlchemy)
│   ├── repositories/           # Tầng tương tác DB (Repository Pattern)
│   ├── schemas/                # Data Validation (Pydantic models)
│   ├── services/               # Logic nghiệp vụ & Kết nối AI Model
│   └── main.py                 # File khởi chạy server Backend
│
├── src/                        # [AI PIPELINE] Mã nguồn xử lý lõi
│   ├── data_loader.py          # Script tải và nạp dữ liệu từ nguồn
│   ├── preprocessing.py        # Làm sạch, xử lý giá trị thiếu/nhiễu
│   ├── feature_engineering.py  # Trích xuất biến thời gian, biến categorical
│   ├── training.py             # Script huấn luyện & lưu mô hình
│   └── inference.py            # Logic dự đoán (Dùng để tích hợp vào API)
│
├── data/                       # [DATA] Quản lý dữ liệu
│   ├── raw/                    # Dữ liệu gốc chưa qua xử lý
│   └── processed/              # Dữ liệu sạch đã sẵn sàng để train
│
├── frontend/                   # [FRONTEND] Giao diện người dùng 
├── models/                     # [BINARIES] Lưu trữ file model (.pkl, .onnx)
├── notebooks/                  # [RESEARCH] Phân tích EDA và thử nghiệm Model
├── tests/                      # [TESTING] Unit tests cho Backend & ML logic
├── scripts/                    # Script tự động hóa (Setup, Docker, Deploy)
│
├── .gitignore                  # Khai báo các file không đưa lên Git
├── .env.example                # File chứa các biến môi trường mẫu
├── LICENSE                     # Quy định về bản quyền
└── requirements.txt            # Danh sách thư viện Python cần thiết