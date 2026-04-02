# 📈 House Price Prediction Project

Dự án xây dựng hệ thống trí tuệ nhân tạo để dự báo giá bất động sản dựa trên các đặc trưng của ngôi nhà (diện tích, vị trí, số phòng ngủ, tiện ích xung quanh...). Hệ thống bao gồm pipeline xử lý dữ liệu tự động, mô hình học máy (XGBoost, CatBoost) với giải thích AI (SHAP), và giao diện web (Streamlit) để người dùng tra cứu giá nhà trực quan.

---

## 🎯 Tính Năng (Features)

- ✅ **Dự đoán giá bất động sản** - Sử dụng ensemble của XGBoost và CatBoost
- ✅ **Giải thích AI (XAI)** - SHAP waterfall charts để hiểu các yếu tố ảnh hưởng
- ✅ **API FastAPI** - RESTful API với Swagger/Redoc documentation
- ✅ **Giao diện Streamlit** - Web app thân thiện người dùng
- ✅ **Docker containerized** - Sẵn sàng deploy
- ✅ **Dự đoán hàng loạt** - Upload CSV để dự đoán nhiều nhà cùng lúc

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```
house-price-prediction-project/
├── src/                              # Mã nguồn chính
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # Entry point API
│   │   ├── schemas.py                # Pydantic schemas
│   │   └── services.py               # Business logic
│   ├── frontend/                     # Streamlit frontend
│   │   ├── app.py                    # Main Streamlit app
│   │   ├── api_client.py             # API client
│   │   ├── config.py                 # Configuration
│   │   ├── ui_components.py          # UI components
│   │   ├── utils.py                  # Utilities
│   │   └── pages/                    # Streamlit pages
│   ├── core/                         # Core modules
│   │   ├── config.py                 # App configuration
│   │   └── utils.py                  # Utility functions
│   └── ml_pipeline/                  # ML pipeline
│       ├── data_loader.py            # Load data
│       ├── preprocessing.py          # Data preprocessing
│       ├── trainer.py                # Model training
│       ├── evaluation.py             # Model evaluation
│       └── explainability.py         # SHAP explanations
│
├── artifacts/                        # Trained models and artifacts
│   ├── models/                       # Saved model files
│   ├── scalers/                      # Feature scalers
│   └── explainers/                   # SHAP explainers
│
├── data/                             # Dataset
│   └── train-house-prices-advanced-regression-techniques.csv
│
├── notebooks/                        # Jupyter notebooks
├── configs/                          # Configuration files
├── logs/                             # Application logs
│
├── Dockerfile                        # Docker image definition
├── docker-compose.yml                # Docker Compose configuration
├── .dockerignore                     # Docker build exclusions
├── .env.docker                       # Docker environment variables
│
├── requirements.txt                  # Python dependencies
├── start_api.py                      # Start API server
├── run_pipeline.py                   # Run ML pipeline
│
├── test_api.py                       # API tests
├── test_explainer.py                 # Explainer tests
├── test_column_mapping.py            # Column mapping tests
│
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🚀 Cách Sử Dụng (How to Use)

### 📋 Yêu Cầu (Requirements)

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 1.29+ ([Install Docker Compose](https://docs.docker.com/compose/install/))

Hoặc chạy cục bộ:
- **Python** 3.11+
- **pip** hoặc **conda**

### 🐳 Quick Start với Docker (Recommended)

#### 1️⃣ Build và Chạy

```bash
# Clone repo (nếu chưa có)
cd house-price-prediction-project

# Build Docker images
docker-compose build

# Chạy ứng dụng
docker-compose up -d
```

#### 2️⃣ Truy Cập Ứng Dụng

| Service | URL | Mô Tả |
|---------|-----|--------|
| **Streamlit Frontend** | http://localhost:8501 | Giao diện web |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **API Redoc** | http://localhost:8000/redoc | ReDoc documentation |
| **API Health** | http://localhost:8000/health | Health check |

#### 3️⃣ Dừng Ứng Dụng

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

#### 4️⃣ Xem Logs

```bash
# View all logs
docker-compose logs -f

# View API logs
docker-compose logs -f api

# View Frontend logs
docker-compose logs -f frontend
```

### 💻 Chạy Cục Bộ (Local Development)

#### 1️⃣ Cài Đặt Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2️⃣ Chạy API Server

```bash
python start_api.py
```

API sẽ chạy tại: http://localhost:8000

#### 3️⃣ Chạy Streamlit Frontend

```bash
streamlit run src/frontend/app.py
```

Frontend sẽ chạy tại: http://localhost:8501

---

## 📊 API Endpoints

### Dự Đoán Đơn (Single Prediction)

```bash
POST /api/predict
Content-Type: application/json

{
  "MSSubClass": 20,
  "LotArea": 8450,
  "LotShape": "Regular",
  "LotConfig": "Inside",
  ...
}

Response:
{
  "predicted_price": 180,
  "model_name": "final_model_Cat",
  "prediction_id": "uuid"
}
```

### Dự Đoán Hàng Loạt (Batch Prediction)

```bash
POST /api/predict/batch
Content-Type: application/json

{
  "data": [
    { "MSSubClass": 20, "LotArea": 8450, ... },
    { "MSSubClass": 30, "LotArea": 9000, ... }
  ]
}

Response:
{
  "predictions": [180, 190, ...],
  "count": 2,
  "model_name": "final_model_Cat"
}
```

### Giải Thích Dự Đoán (Explanation)

```bash
POST /api/explain
Content-Type: application/json

{
  "MSSubClass": 20,
  "LotArea": 8450,
  ...
}

Response:
{
  "predicted_price": 180,
  "base_value": 175,
  "contributions": [
    {"feature": "LotArea", "contribution": 2.5},
    {"feature": "GrLivArea", "contribution": 1.8},
    ...
  ],
  "waterfall_data": {...}
}
```

### Health Check

```bash
GET /api/health

Response:
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "final_model_Cat",
  "version": "1.0.0"
}
```

---

## 🎮 Streamlit Frontend Guide

### Tab 1: 🏠 Dự Đoán Đơn (Single Predict)

- Nhập thông tin ngôi nhà vào form
- Chọn mô hình (XGBoost hoặc CatBoost)
- Xem kết quả dự đoán giá
- Xem biểu đồ SHAP giải thích các yếu tố ảnh hưởng

### Tab 2: 📊 Dự Đoán Hàng Loạt (Batch Predict)

- Upload file CSV với thông tin nhiều ngôi nhà
- Xem kết quả cho toàn bộ dữ liệu
- Download kết quả dưới dạng CSV

### Tab 3: 🔍 Giải Thích (Explain)

- Nhập thông tin ngôi nhà
- Xem waterfall chart (giải thích chi tiết)
- Xem bar plot (öl ảnh hưởng tính năng)
- Hiểu các yếu tố quyết định giá

---

## 🔧 Configuration

### Environment Variables

Xem file `.env.docker` để cấu hình:

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
API_URL=http://api:8000

# Python
PYTHONUNBUFFERED=1
```

### Docker Compose Configuration

File `docker-compose.yml` bao gồm:

- **API Service**: FastAPI backend trên port 8000
- **Frontend Service**: Streamlit app trên port 8501
- **Networks**: Internal network cho communication
- **Volumes**: Lưu trữ model, log, và data
- **Health Checks**: Tự động kiểm tra sức khỏe

---

## 🧪 Testing

### Run Tests

```bash
# Test API endpoints
python test_api.py

# Test explainer
python test_explainer.py

# Test column mapping
python test_column_mapping.py
```

### Docker Testing

```bash
# Run tests inside container
docker-compose exec api python test_api.py
```

---

## 📈 Model Information

- **Models Used**: XGBoost, CatBoost (Ensemble)
- **Features**: 243 processed features (79 original)
- **Training Data**: Advanced House Prices dataset
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Preprocessing**: StandardScaler + Feature Engineering

### Model Performance

- Trained on house price prediction dataset
- Cross-validation used for model selection
- Features: Area measurements, quality indicators, location, etc.

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8000 (Windows)
netstat -ano | findstr :8000

# Check port 8501 (Linux/Mac)
lsof -i :8501

# Kill process (adjust port/PID as needed)
taskkill /PID <PID> /F
```

### Container Won't Start

```bash
# Remove old containers
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache

# Run again
docker-compose up -d
```

### API Health Check Failing

```bash
# Check API logs
docker-compose logs -f api

# Manual health check
curl http://localhost:8000/health

# Restart API
docker-compose restart api
```

### Frontend Can't Connect to API

```bash
# Verify API is running
docker-compose ps

# Check network connectivity
docker-compose exec frontend curl http://api:8000/health

# Restart services
docker-compose restart
```

---

## 📚 Documentation

- API documentation available at `/docs` (Swagger UI)
- Interactive API exploration at `/redoc` (ReDoc)
- All endpoints are fully documented with examples

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Development

### For API Development

```bash
# Enter API container
docker-compose exec api /bin/bash

# Install new packages
pip install <package-name>

# Run specific script
python run_pipeline.py
```

### For Frontend Development

```bash
# Enter Frontend container
docker-compose exec frontend /bin/bash

# View dependencies
pip list
```

---

## 🎓 Technologies Used

### Backend
- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend
- **Streamlit** - Web app framework
- **Requests** - HTTP client

### Machine Learning
- **XGBoost** - Gradient boosting
- **CatBoost** - Categorical boosting
- **Scikit-learn** - ML utilities
- **SHAP** - Model explainability

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Preprocessing

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Orchestration

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Check container logs: `docker-compose logs -f`

---

**Last Updated**: April 2, 2026
**Version**: 1.0.0