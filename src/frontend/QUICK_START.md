# ⚡ Quick Start Guide - Frontend Streamlit

## 1️⃣ Cài đặt (2 phút)

```bash
# Từ project root
pip install -r requirements.txt
```

## 2️⃣ Khởi chạy Backend API

```bash
# Terminal 1: Khởi chạy Backend FastAPI
python start_api.py
```

Chờ choẻn:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## 3️⃣ Khởi chạy Frontend Streamlit

```bash
# Terminal 2: Khởi chạy Frontend
streamlit run frontend/app.py
```

Chờ choẻn sẽ mở tự động browser. Nếu không, truy cập:
```
http://localhost:8501
```

## 🎯 Sử dụng Cơ bản

### Step 1: Kiểm tra Server (Sidebar)
- Click **Refresh** nếu cần
- Xem trạng thái 🟢 Server Online

### Step 2: Tải Mô hình (Sidebar)
1. Chọn mô hình từ dropdown (e.g., `final_model_Cat`)
2. Click **📥 Tải Mô hình**
3. Chờ ✅ **Mô hình đã được tải**

### Step 3: Chọn Tính năng (Main Area)
Chọn 1 trong 3 options từ sidebar:

#### 📝 **Tab 1: Định giá Đơn lẻ**
- Nhập các giá trị
- Click **🔮 Dự đoán Giá**
- Xem kết quả 💰

#### 📁 **Tab 2: Định giá Batch**
- Upload file CSV (xem `sample_batch.csv`)
- Click **🚀 Dự đoán Batch**
- Download kết quả 📥

#### 🔍 **Tab 3: Phân tích Chuyên sâu**
- Nhập các giá trị
- Chọn số TOP features
- Click **🔮 Dự đoán & Giải thích**
- Xem Waterfall chart + details

## 📋 Ví dụ CSV Format

File `sample_batch.csv` có 15 dòng test:

```csv
LotArea,OverallQual,OverallCond,YearBuilt,GrLivArea,TotalBsmtSF,1stFlrSF,2ndFlrSF,Bedrooms,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,Fireplaces,GarageCars
8450,7,5,2003,1710,856,856,854,3,1,1,3,1,0,2
9600,8,5,1976,2262,1262,1262,1000,4,2,1,4,1,1,2
...
```

**Lưu ý:**
- ✅ Header row bắt buộc
- ✅ UTF-8 encoding
- ✅ Max 100 rows
- ✅ Columns không có → dùng defaults

## 🐛 Debugging

### Error: "Server Offline"
```bash
# Terminal 1: Chắc Backend chạy
python start_api.py

# Kiểm tra: http://localhost:8000/docs
```

### Error: "Không thể tải mô hình"
```bash
# Kiểm tra file model:
ls artifacts/models/
# output: final_model_Cat.joblib

# Nếu không có, train lại:
python run_pipeline.py
```

### Error: "CSV parsing failed"
```
✅ Xoá Unicode BOM nếu có
✅ Kiểm tra delimiters (phẩy)
✅ Kiểm tra số & text types
```

## 📚 File Structures

```
frontend/
├── app.py                 # Main entry point
├── config.py              # Configuration
├── api_client.py          # API calls
├── ui_components.py       # UI widgets
├── utils.py               # Utilities
├── sample_batch.csv       # Example CSV
├── pages/
│   ├── 1_single_predict.py
│   ├── 2_batch_predict.py
│   └── 3_explain.py
└── .streamlit/
    └── config.toml        # Streamlit config
```

## 🔗 Useful Links

| Resource | URL |
|----------|-----|
| This Guide | [frontend/QUICK_START.md](./QUICK_START.md) |
| Full README | [frontend/README.md](./README.md) |
| API Docs | http://localhost:8000/docs |
| Streamlit App | http://localhost:8501 |

## 💡 Tips

- **Mô hình cached:** Lần tiếp theo reload nhanh hơn
- **Session State:** không reset khi click buttons
- **Visualization:** Hover charts để xem chi tiết
- **CSV Download:** CSV được format tự động từ table

---

**Happy Predicting! 🚀** 

Nếu có issue, check `README.md` section "Troubleshooting"
