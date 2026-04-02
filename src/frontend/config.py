"""
Frontend Configuration and Constants

Chứa tất cả các hằng số, cấu hình, và dữ liệu mặc định cho ứng dụng Streamlit.
"""

import os

# ============================================================================
# API Configuration
# ============================================================================
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 10  # seconds
HEALTH_CHECK_TIMEOUT = 2

# ============================================================================
# UI Configuration
# ============================================================================
PAGE_TITLE = "Dự đoán Giá Nhà - CatBoost"
PAGE_ICON = "🏠"
LAYOUT = "wide"

# Các nhóm tính năng
FEATURE_GROUPS = {
    "Đánh giá & Vị trí": {
        "YearBuilt": ("Năm xây dựng", "slider", 1872, 2020, 2003),
        "OverallQual": ("Chất lượng tổng thể", "slider", 1, 10, 7),
        "OverallCond": ("Tình trạng tổng thể", "slider", 1, 5, 3),
        "Neighborhood": ("Khu vực", "text", "NAmes"),
    },
    "Diện tích (Yếu tố quyết định)": {
        "LotArea": ("Diện tích đất (sqft)", "number", 8450),
        "LotFrontage": ("Chiều dài tiền đất (ft)", "number", 50),
        "GrLivArea": ("Diện tích sống (sqft)", "number", 1500),
        "TotalBsmtSF": ("Tổng diện tích tầng hầm (sqft)", "number", 500),
        "1stFlrSF": ("Diện tích tầng 1 (sqft)", "number", 850),
        "2ndFlrSF": ("Diện tích tầng 2 (sqft)", "number", 0),
    },
    "Phòng ốc & Tiện ích": {
        "Bedrooms": ("Số phòng ngủ", "slider", 0, 8, 3),
        "Bathrooms": ("Số phòng tắm", "slider", 0, 4, 1),
        "FullBath": ("Phòng tắm đầy đủ", "slider", 0, 4, 1),
        "HalfBath": ("Nửa phòng tắm", "slider", 0, 2, 0),
        "BedroomAbvGr": ("Phòng ngủ trên tầng chính", "slider", 0, 8, 3),
        "KitchenAbvGr": ("Bếp trên tầng chính", "slider", 1, 3, 1),
        "Fireplaces": ("Lò sưởi", "slider", 0, 4, 0),
        "GarageCars": ("Chỗ để xe", "slider", 0, 4, 2),
    }
}

# ============================================================================
# Default Features / Dữ liệu Default
# ============================================================================
# Giá trị mặc định cho ~50-60 trường không hiển thị trên UI
DEFAULT_FEATURES = {
    # Basement
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinSF1": 706,
    "BsmtFinSF2": 0,
    "BsmtFinType1": "GLQ",
    "BsmtFinType2": "Unf",
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "BsmtQual": "Gd",
    "BsmtUnfSF": 150,
    
    # Central
    "CentralAir": "Y",
    
    # Condition
    "Condition1": "Norm",
    "Condition2": "Norm",
    
    # Electrical & Porch
    "Electrical": "SBrkr",
    "EnclosedPorch": 0,
    
    # Exterior
    "ExterCond": "TA",
    "ExterQual": "Gd",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    
    # Fireplace & Foundation
    "FireplaceQu": "Gd",
    "Foundation": "PConc",
    
    # Functional
    "Functional": "Typ",
    
    # Garage
    "GarageCond": "TA",
    "GarageFinish": "RFn",
    "GarageQual": "TA",
    "GarageType": "Attchd",
    "GarageYrBlt": 2003,
    
    # Heating
    "Heating": "GasA",
    "HeatingQC": "Ex",
    
    # Kitchen
    "KitchenQual": "Gd",
    
    # Land
    "LandContour": "Lvl",
    "LandSlope": "Gtl",
    
    # Lot
    "LotConfig": "Inside",
    "LotShape": "Reg",
    
    # MSSubClass & MSZoning
    "MSSubClass": 60,
    "MSZoning": "RL",
    
    # Masonry
    "MasVnrArea": 196,
    "MasVnrType": "BrkFace",
    
    # Misc
    "MiscVal": 0,
    "MoSold": 2,
    
    # Porch
    "OpenPorchSF": 61,
    
    # Paved Drive
    "PavedDrive": "Y",
    
    # Pool
    "PoolArea": 0,
    "PoolQC": "NA",
    
    # Roof
    "RoofMatl": "CompShg",
    "RoofStyle": "Gable",
    
    # Sale
    "SaleCondition": "Normal",
    "SaleType": "WD",
    
    # Screen Porch & Street
    "ScreenPorch": 0,
    "Street": "Pave",
    
    # Three Season Porch
    "ThreeSsnPorch": 0,
    
    # Utilities
    "Utilities": "AllPub",
    
    # Wood Deck
    "WoodDeckSF": 0,
    
    # Year Remod Add & Yr Sold
    "YearRemodAdd": 2003,
    "YrSold": 2008,
    
    # Low Quality Fin
    "LowQualFinSF": 0,
    
    # House Style & Bldg Type (common defaults)
    "HouseStyle": "2Story",
    "BldgType": "1Fam",
    
    # Alley
    "Alley": "NA",
}

# ============================================================================
# Batch Upload Configuration
# ============================================================================
MAX_BATCH_SIZE = 100  # Max số nhà trong batch predictions
ALLOWED_FILE_TYPES = ["csv"]

# ============================================================================
# Visualization Configuration
# ============================================================================
PLOT_HEIGHT = 400
PLOT_WIDTH = 600

# Color palette cho charts
COLOR_POSITIVE = "#2ecc71"  # Green
COLOR_NEGATIVE = "#e74c3c"  # Red
COLOR_NEUTRAL = "#3498db"   # Blue

# ============================================================================
# Messages & Strings
# ============================================================================
MSG_HEALTH_OK = "🟢 Server Online"
MSG_HEALTH_ERROR = "🔴 Server Offline / Mất kết nối"
MSG_HEALTH_WARNING = "🟡 Server phản hồi lỗi"

MSG_MODEL_LOADING = "⏳ Đang tải mô hình..."
MSG_MODEL_LOADED = "✅ Mô hình đã được tải"
MSG_MODEL_FAILED = "❌ Tải mô hình thất bại"

MSG_PREDICTION_ERROR = "❌ Lỗi dự đoán"
MSG_PREDICTION_SUCCESS = "✅ Dự đoán thành công"

MSG_BATCH_PROCESSING = "⏳ Đang xử lý batch..."
MSG_BATCH_SUCCESS = "✅ Batch xử lý thành công"
MSG_BATCH_ERROR = "❌ Batch xử lý thất bại"
