# 🏠 Frontend Streamlit - Hệ thống Dự đoán Giá Nhà

Giao diện Streamlit cho hệ thống dự đoán giá nhà sử dụng CatBoost với SHAP explanations.

## 📋 Cấu trúc Thư mục

```
frontend/
├── app.py                    # Main entry point (Streamlit multipage)
├── config.py                 # Configuration & constants
├── api_client.py             # API client với xử lý lỗi
├── ui_components.py          # Reusable UI components
├── utils.py                  # Utility functions
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── pages/
│   ├── 1_single_predict.py  # Tab 1: Single prediction
│   ├── 2_batch_predict.py   # Tab 2: Batch predictions
│   └── 3_explain.py         # Tab 3: SHAP explanations
└── README.md                 # This file
```

## 🚀 Bắt đầu

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Backend API

Trước tiên, hãy chắc rằng Backend FastAPI đang chạy:

```bash
# Từ thư mục project root
python start_api.py
```

API sẽ khởi chạy tại `http://localhost:8000`

### 3. Khởi chạy Frontend Streamlit

```bash
# Từ thư mục frontend
streamlit run app.py
```

hoặc từ project root:

```bash
streamlit run frontend/app.py
```

Frontend sẽ mở tại `http://localhost:8501`

## 📖 Hướng dẫn Sử dụng

### Sidebar Functions

#### 🏥 Trạng thái Server
- Tự động kiểm tra kết nối Backend
- Hiển thị trạng thái model, version, v.v.
- Error messages nếu server offline

#### 📦 Quản lý Mô hình
1. **Lấy danh sách mô hình** từ Backend
2. **Chọn mô hình** từ dropdown
3. **Tải mô hình** - nút "Load Model" sẽ gọi API để nạp mô hình

### Main Pages

#### 📝 Tab 1: Định giá Đơn lẻ (Single Predict)
- Nhập 15 trường chính có ảnh hưởng lớn nhất
- ~60 trường khác tự động sử dụng giá trị mặc định
- Hiển thị:
  - 💰 Giá dự đoán (formatted currency)
  - 📊 Độ tin cậy (confidence %)
  - 🔢 Mô hình sử dụng

**Các nhóm input:**
- **Đánh giá & Vị trí:** YearBuilt, OverallQual, OverallCond, Neighborhood
- **Diện tích (Yếu tố quyết định):** LotArea, LotFrontage, GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF
- **Phòng ốc & Tiện ích:** Bedrooms, Bathrooms, FullBath, HalfBath, Fireplaces, GarageCars

#### 📁 Tab 2: Định giá Hàng loạt (Batch Predict)
- Upload file CSV chứa danh sách nhà
- Tối đa 100 nhà/batch
- Kết quả:
  - 📊 Bảng kết quả predictions
  - 📈 Thống kê (Min, Max, Average)
  - 📊 Biểu đồ phân phối giá
  - 📥 Download CSV kết quả

**CSV Format:**
- Header row bắt buộc
- Columns: LotArea, OverallQual, GrLivArea, v.v.
- Columns không có sẽ dùng default values

#### 🔍 Tab 3: Phân tích Chuyên sâu (Predict with Explanation)
- Nhập thông số tương tự Tab 1
- Tuỳ chọn số lượng TOP features (3-20)
- Kết quả:
  - 💰 Giá dự đoán
  - 📊 **Waterfall Chart** - cách mỗi feature ảnh hưởng đến giá
  - 📈 **Bar Chart** - TOP features ranking by impact
  - 📋 **Chi tiết** - danh sách từng feature với:
    - ⬆️/⬇️ Hướng tác động (TĂNG/GIẢM)
    - 💵 Mức độ đóng góp ($)
    - 📌 Giá trị gốc

## 🔧 Xử lý Lỗi

### Try-Except Coverage

**API Client (`api_client.py`):**
- Connection timeout → APIError
- Connection refused → APIError
- HTTP errors (4xx, 5xx) → APIError + status code
- JSON parsing errors → APIError
- Detailed error messages từ Backend

**UI Components (`ui_components.py`):**
- Chart rendering errors → st.warning
- DataFrame conversion errors → st.error
- Graceful fallbacks cho failures

**Pages:**
- Valid input checking trước API call
- CSV parsing error handling
- Batch size validation
- SHAP explainer unavailable → info message

### Error Messages

Tất cả error messages:
- Tiếng Việt, user-friendly
- Specific error details
- Suggestions để fix (khi có thể)
- Logged cho debugging

## 📊 Features & Capabilities

### ✅ Default Data Management

Hệ thống **tự động gom nhóm & inject** ~60 default features:

```python
# Từ config.py
DEFAULT_FEATURES = {
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "PoolArea": 0,
    ...
}

# Khi user nhập dữ liệu, frontend hợp nhất:
payload = DEFAULT_FEATURES.copy()
payload.update(user_input)  # Override với user values
```

Lợi ích:
- ✅ Giao diện không rối rắm
- ✅ Code clean, dễ maintain
- ✅ Dễ thêm/bớt default values
- ✅ Consistent behavior

### ✅ CSV Batch Processing

```python
# Tự động mapping columns từ CSV
feature_mapping = get_csv_feature_mapping(df)

# Parse records + validate
records, errors = parse_csv_for_batch(df, feature_mapping)

# Merge mỗi record với defaults
records_with_defaults = merge_with_defaults(records)

# Gửi batch payload
batch_payload = {"houses": records_with_defaults}
```

### ✅ SHAP Visualization

- **Waterfall Chart**: Base value → features → final prediction
- **Bar Chart**: Feature contributions ranked by magnitude
- **Color coding**: Green (positive), Red (negative)
- **Interactive**: Hover để xem chi tiết

### ✅ Session State Management

```python
st.session_state.model_loaded      # Bool
st.session_state.current_model     # String
st.session_state.last_prediction   # Dict
st.session_state.api_health        # Dict
```

Tránh reload API không cần thiết

## 🎨 UI/UX Details

### Color Scheme
- 🟢 **Green** (#2ecc71): Positive contributions, success
- 🔴 **Red** (#e74c3c): Negative contributions, errors
- 🔵 **Blue** (#3498db): Neutral, info
- ⚫ **Gray**: Defaults, disabled

### Responsive Layout
- 2-3 columns tùy theo màn hình
- Full-width tables/charts
- Mobile-friendly

### Emoji Integration
- 🏠 Home icon
- 📝 Form/input
- 📁 Batch/file
- 🔍 Analysis/search
- 💰 Money/price
- 📊 Charts/data

---

## 🛠️ Development & Maintenance

### Adding New Features

1. Thêm constants vào `config.py`
2. Thêm API methods vào `api_client.py` (nếu cần)
3. Thêm UI components vào `ui_components.py`
4. Tạo page mới trong `pages/` nếu cần
5. Update `app.py` sidebar nếu cần

### Debugging

```bash
# Streamlit debug logs
streamlit run app.py --logger.level=debug

# Check session state (thêm vào page đó)
import streamlit as st
st.write("Session State:", st.session_state)
```

### Testing Locally

1. **Single predict test:**
   - Fill form, click predict
   - Check price format & confidence

2. **Batch predict test:**
   - Create test CSV (xem `sample_batch.csv`)
   - Upload & check results

3. **Explain test:**
   - Fill form, adjust top_features
   - Check waterfall chart rendering

---

## 📝 Configuration

### Streamlit Config (`.streamlit/config.toml`)
- Theme colors
- Max upload size
- Logger level
- Server settings

### API Config (`config.py`)
- `BACKEND_URL`: API endpoint
- `API_TIMEOUT`: Request timeout
- `MAX_BATCH_SIZE`: Batch limit
- Feature groups & defaults

---

## 📞 Troubleshooting

### Server Offline
```
❌ Server Offline / Mất kết nối
→ Kiểm tra Backend đang chạy: python start_api.py
→ Kiểm tra BACKEND_URL trong config.py
```

### Model Failed to Load
```
❌ Tải mô hình thất bại
→ Kiểm tra file model trong artifacts/models/
→ Kiểm tra logs Backend
```

### CSV Upload Error
```
❌ Batch xử lý thất bại
→ Kiểm tra format CSV (header, delimiters)
→ Kiểm tra encoding UTF-8
→ Kiểm tra số lượng rows < 100
```

### SHAP Explainer Not Available
```
💡 Tính năng giải thích SHAP chưa sẵn sàng
→ Backend cần retrain model with XAI support
→ Kiểm tra: /health endpoint - model_loaded=true
```

---

## 📚 Reference

- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Docs](https://plotly.com/python/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

**Version:** 1.0.0  
**Last Updated:** Q1 2024
