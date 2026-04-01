"""
Utility Functions

Các hàm tiện ích dùng chung cho frontend.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def format_currency(value: float) -> str:
    """
    Format số tiền theo định dạng USD.
    
    Args:
        value: Số tiền cần format
        
    Returns:
        String định dạng: $123,456.78
    """
    try:
        return f"${value:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return f"${value:.2f}"


def format_confidence(confidence: float) -> str:
    """
    Format confidence score thành phần trăm.
    
    Args:
        confidence: Giá trị confidence (0-1)
        
    Returns:
        String: 85.5%
    """
    try:
        return f"{confidence * 100:.1f}%"
    except Exception as e:
        logger.error(f"Error formatting confidence: {e}")
        return f"{confidence:.2%}"


def get_confidence_color(confidence: float) -> str:
    """
    Lấy màu dựa trên confidence score.
    
    Args:
        confidence: Giá trị confidence (0-1)
        
    Returns:
        Màu hex: #2ecc71 (green) nếu cao, #e74c3c (red) nếu thấp
    """
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"


def get_confidence_emoji(confidence: float) -> str:
    """
    Lấy emoji dựa trên confidence score.
    
    Args:
        confidence: Giá trị confidence (0-1)
        
    Returns:
        Emoji string
    """
    if confidence >= 0.9:
        return "🟢"
    elif confidence >= 0.7:
        return "🟡"
    else:
        return "🔴"


def validate_house_data(data: Dict[str, Any], required_fields: Optional[list] = None) -> tuple[bool, str]:
    """
    Validate dữ liệu nhà.
    
    Args:
        data: Dict chứa dữ liệu nhà
        required_fields: List các field bắt buộc (nếu có)
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Kiểm tra fields bắt buộc
        if required_fields:
            for field in required_fields:
                if field not in data or data[field] is None:
                    return False, f"Trường '{field}' là bắt buộc"
        
        # Kiểm tra numeric fields
        numeric_fields = ["LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF", 
                         "1stFlrSF", "2ndFlrSF", "YearBuilt"]
        for field in numeric_fields:
            if field in data and data[field] is not None:
                if data[field] < 0:
                    return False, f"{field} không thể âm"
        
        return True, ""
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, f"Lỗi kiểm tra dữ liệu: {str(e)}"


def parse_csv_for_batch(df: pd.DataFrame, feature_mapping: Dict[str, str]) -> tuple[list, list]:
    """
    Parse CSV DataFrame thành list predictions và list errors.
    
    Args:
        df: Pandas DataFrame từ CSV upload
        feature_mapping: Mapping giữa CSV columns và feature names
        
    Returns:
        Tuple (valid_records, error_messages)
    """
    valid_records = []
    errors = []
    
    try:
        for idx, row in df.iterrows():
            record = {}
            row_errors = []
            
            for csv_col, feature_name in feature_mapping.items():
                if csv_col in df.columns:
                    value = row[csv_col]
                    # Handle NaN
                    if pd.isna(value):
                        record[feature_name] = None
                    else:
                        record[feature_name] = value
            
            if row_errors:
                errors.append(f"Row {idx + 1}: {', '.join(row_errors)}")
            else:
                valid_records.append(record)
        
        return valid_records, errors
    except Exception as e:
        logger.error(f"CSV parsing error: {e}")
        errors.append(f"Lỗi parse CSV: {str(e)}")
        return valid_records, errors


def convert_predictions_to_dataframe(batch_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert batch prediction result thành DataFrame.
    
    Args:
        batch_result: Dict từ API batch predict
        
    Returns:
        DataFrame chứa results
    """
    try:
        predictions = batch_result.get("predictions", [])
        if not predictions:
            return pd.DataFrame()
        
        # Tạo DataFrame từ predictions
        df = pd.DataFrame([
            {
                "ID": idx,
                "Predicted Price": pred.get("predicted_price", 0),
                "Confidence": pred.get("confidence", 0),
                "Model": pred.get("model_name", "Unknown")
            }
            for idx, pred in enumerate(predictions)
        ])
        
        return df
    except Exception as e:
        logger.error(f"Error converting predictions to dataframe: {e}")
        return pd.DataFrame()


def get_shap_contribution_description(shap_value: float, feature_name: str, original_value: Any) -> str:
    """
    Tạo description text cho SHAP contribution.
    
    Args:
        shap_value: Giá trị SHAP
        feature_name: Tên feature
        original_value: Giá trị gốc
        
    Returns:
        String description
    """
    try:
        sign = "⬆️ TĂNG" if shap_value > 0 else "⬇️ GIẢM"
        amount = abs(shap_value)
        return f"{sign} giá {format_currency(amount)} do {feature_name}={original_value}"
    except Exception as e:
        logger.error(f"Error creating SHAP description: {e}")
        return f"Contribution: {feature_name}"


def save_predictions_to_csv(df: pd.DataFrame, filename: str = "predictions.csv") -> bytes:
    """
    Convert DataFrame thành CSV bytes để download.
    
    Args:
        df: DataFrame cần save
        filename: Tên file
        
    Returns:
        CSV bytes
    """
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return b""


def initialize_session_state():
    """
    Khởi tạo session state nếu chưa có.
    """
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    
    if "api_health" not in st.session_state:
        st.session_state.api_health = None
