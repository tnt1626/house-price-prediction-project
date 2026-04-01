# XAI (Explainable AI) IMPLEMENTATION GUIDE

## 🎯 Mục tiêu
Tích hợp module Explainable AI (SHAP/LIME) vào hệ thống House Price Prediction hiện tại. Yêu cầu tiên quyết:
1. **Modular**: Tách biệt hoàn toàn logic XAI khỏi luồng inference chính.
2. **Hiệu năng**: Không làm chậm endpoint `/predict` hiện tại. Phải tạo một endpoint riêng cho XAI.
3. **Tái sử dụng**: Phải tính toán và lưu `Explainer` object (joblib) trong quá trình training, API chỉ load lên để dùng (không fit lại).

---

## ⚠️ CHÚ Ý QUAN TRỌNG CHO AI AGENT (CRITICAL WARNINGS)

**VỀ TÊN ĐẶC TRƯNG (FEATURE NAMES MAPPING):**
Người dùng (User) đã đặt lại tên cho các đặc trưng trong dữ liệu đầu vào và schema. Khi dữ liệu đi qua `ColumnTransformer` (bao gồm OrdinalEncoder, TargetEncoder, v.v.), tên cột sẽ bị biến đổi (ví dụ: `Neighborhood` thành `TargetEncoder_Neighborhood` hoặc `x0`, `x1`).

**Nhiệm vụ bắt buộc của Agent:**
1. Khi output SHAP values, **TUYỆT ĐỐI KHÔNG** trả về các tên cột đã bị biến đổi (như `TargetEncoder_Neighborhood`) cho phía client.
2. Phải viết logic để map ngược tên cột từ sau khi qua `preprocessor` về lại đúng tên biến gốc trong schema `HousePriceInput` (hoặc tên human-readable tương ứng).
3. Sử dụng `preprocessor.get_feature_names_out()` nếu có thể, và dùng regex/dictionary để dọn dẹp (clean) các prefix do Scikit-learn thêm vào.

---

## 📂 Cập Nhật Cấu Trúc Thư Mục

Thêm các file và thư mục sau vào project hiện tại:

```text
house-price-prediction-project/
├── src/
│   ├── core/
│   │   └── config.py              # Thêm: EXPLAINER_DIR = BASE_DIR / "artifacts/explainers"
│   ├── ml_pipeline/
│   │   └── explainability.py      # [MỚI] Chứa class ModelExplainer
│   └── api/
│       ├── schemas.py             # Thêm: FeatureExplanation, PredictionWithExplainResponse
│       ├── services.py            # Thêm: logic load explainer và hàm predict_and_explain
│       └── main.py                # Thêm: POST /predict-explain
├── artifacts/
│   └── explainers/                # [MỚI] Nơi lưu shap_explainer.joblib

# XAI (Explainable AI) IMPLEMENTATION GUIDE

## 🎯 Mục tiêu
Tích hợp module Explainable AI (SHAP/LIME) vào hệ thống House Price Prediction hiện tại. Yêu cầu tiên quyết:
1. **Modular**: Tách biệt hoàn toàn logic XAI khỏi luồng inference chính.
2. **Hiệu năng**: Không làm chậm endpoint `/predict` hiện tại. Phải tạo một endpoint riêng cho XAI.
3. **Tái sử dụng**: Phải tính toán và lưu `Explainer` object (joblib) trong quá trình training, API chỉ load lên để dùng (không fit lại).

---

## ⚠️ CHÚ Ý QUAN TRỌNG CHO AI AGENT (CRITICAL WARNINGS)

**VỀ TÊN ĐẶC TRƯNG (FEATURE NAMES MAPPING):**
Người dùng (User) đã đặt lại tên cho các đặc trưng trong dữ liệu đầu vào và schema. Khi dữ liệu đi qua `ColumnTransformer` (bao gồm OrdinalEncoder, TargetEncoder, v.v.), tên cột sẽ bị biến đổi (ví dụ: `Neighborhood` thành `TargetEncoder_Neighborhood` hoặc `x0`, `x1`).

**Nhiệm vụ bắt buộc của Agent:**
1. Khi output SHAP values, **TUYỆT ĐỐI KHÔNG** trả về các tên cột đã bị biến đổi (như `TargetEncoder_Neighborhood`) cho phía client.
2. Phải viết logic để map ngược tên cột từ sau khi qua `preprocessor` về lại đúng tên biến gốc trong schema `HousePriceInput` (hoặc tên human-readable tương ứng).
3. Sử dụng `preprocessor.get_feature_names_out()` nếu có thể, và dùng regex/dictionary để dọn dẹp (clean) các prefix do Scikit-learn thêm vào.

---

## 📂 Cập Nhật Cấu Trúc Thư Mục

Thêm các file và thư mục sau vào project hiện tại:

house-price-prediction-project/
├── src/
│   ├── core/
│   │   └── config.py              # Thêm: EXPLAINER_DIR = BASE_DIR / "artifacts/explainers"
│   ├── ml_pipeline/
│   │   └── explainability.py      # [MỚI] Chứa class ModelExplainer
│   └── api/
│       ├── schemas.py             # Thêm: FeatureExplanation, PredictionWithExplainResponse
│       ├── services.py            # Thêm: logic load explainer và hàm predict_and_explain
│       └── main.py                # Thêm: POST /predict-explain
├── artifacts/
│   └── explainers/                # [MỚI] Nơi lưu shap_explainer.joblib

---

## 🛠️ Chi Tiết Triển Khai (Implementation Steps)

### Bước 1: Cấu hình (Configuration)
Trong `src/core/config.py`:
- Thêm đường dẫn `EXPLAINER_DIR = ARTIFACTS_DIR / "explainers"`.
- Đảm bảo thư mục này được tự động tạo khi khởi chạy bằng thư viện `pathlib`.

### Bước 2: Module Explainability (`src/ml_pipeline/explainability.py`)
Tạo class `ModelExplainer` chịu trách nhiệm:
1. **Khởi tạo & Fit**: Sử dụng `shap.TreeExplainer` cho các mô hình dạng Tree (XGBoost, LightGBM, Random Forest) để tối ưu tốc độ.
2. **Lưu trữ**: Lưu explainer object bằng `joblib`.
3. **Giải thích (Inference)**: Cung cấp hàm `get_local_explanation(X_transformed, original_input)`.
   - *Agent Note:* Tại đây, phải thực hiện logic **Mapping Feature Names** như đã cảnh báo ở trên. Kết hợp `shap_values` với tên gốc từ `original_input.keys()`.

### Bước 3: Tích hợp vào Training Pipeline (`run_pipeline.py` & `trainer.py`)
Ngay sau khi quá trình huấn luyện tìm ra **Best Model**:
1. Khởi tạo `ModelExplainer(best_model, preprocessor)`.
2. Gọi hàm fit explainer với `X_train` đã qua biến đổi.
3. (Tùy chọn) Generate một `shap.summary_plot`, lưu thành file `.png` và log vào MLflow.
4. Lưu explainer vào `artifacts/explainers/best_model_explainer.joblib`.

### Bước 4: Cập nhật API Schemas (`src/api/schemas.py`)
Tạo các Pydantic models mới KHÔNG làm ảnh hưởng đến schema cũ:

```python
from pydantic import BaseModel
from typing import List, Any
from .schemas import PredictionResponse # Import schema hiện tại

class FeatureExplanation(BaseModel):
    feature_name: str
    original_value: Any
    shap_value: float
    contribution_type: str # "positive" (làm tăng giá) hoặc "negative" (làm giảm giá)

class PredictionWithExplainResponse(PredictionResponse):
    base_value: float # Giá trị trung bình nền của SHAP
    explanations: List[FeatureExplanation]

### Bước 5: Cập nhật API Services (src/api/services.py)
Trong PredictionService:

- Thêm hàm `load_explainer()` chạy song song với `load_model()`.
- Tạo hàm `predict_and_explain(self, input_data: HousePriceInput)`.
  - Thực hiện predict như bình thường.
  - Gọi explainer để lấy `shap_values`.
  - Format lại data khớp với list `FeatureExplanation`. Sắp xếp các feature có độ lớn `|shap_value|` giảm dần (chỉ lấy top 10 hoặc 15 feature quan trọng nhất để response gọn nhẹ).

### Bước 6: Thêm API Endpoint (src/api/main.py)
Thêm endpoint mới, độc lập với endpoint dự đoán thông thường:

```python
@app.post("/predict-explain", response_model=PredictionWithExplainResponse, tags=["Predictions"])
async def predict_with_explanation(input_data: HousePriceInput):
    """
    Dự đoán giá nhà và kèm theo XAI (SHAP) để giải thích các yếu tố ảnh hưởng.
    Endpoint này trả về giá dự đoán cộng với Top các đặc trưng làm tăng/giảm giá.
    """
    # Gọi return service.predict_and_explain(input_data)
    pass
```

## ✅ Tiêu chuẩn hoàn thành (Definition of Done)
- [ ] Pipeline training chạy mượt mà, tự động sinh và lưu được file `.joblib` của Explainer.
- [ ] Endpoint `/predict-explain` hoạt động, trả về JSON chuẩn xác.
- [ ] Tên các features trong `explanations` phải đọc được (ví dụ: `OverallQual`, `Neighborhood`), KHÔNG chứa các prefix kỹ thuật (ví dụ: `TargetEncoder_Neighborhood`, `remainder__`).
- [ ] Code có type hints và docstrings đầy đủ, bám sát phong cách của các file hiện tại.