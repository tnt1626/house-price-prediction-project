"""
Single Prediction Page - Tab 1

Định giá đơn lẻ cho một căn nhà specific.
Frontend gom nhóm dữ liệu mặc định tự động.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import logging
from typing import Dict, Any

from api_client import get_api_client, APIError
from config import FEATURE_GROUPS, DEFAULT_FEATURES
from ui_components import (
    create_input_form,
    display_prediction_result,
    display_shap_explanations
)
from utils import format_currency, validate_house_data, initialize_session_state

logger = logging.getLogger(__name__)


def prepare_prediction_payload(user_input: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kết hợp user input với default features để tạo payload hoàn chỉnh.
    
    Args:
        user_input: Dict từ form input
        defaults: Dict chứa các default features
        
    Returns:
        Payload hoàn chỉnh để gửi tới API
    """
    # Start with defaults
    payload = defaults.copy()
    
    # Override with user inputs
    payload.update(user_input)
    
    return payload


def main():
    """Main function for Single Predict page"""
    st.set_page_config(page_title="Định giá Đơn lẻ", page_icon="📝", layout="wide")
    initialize_session_state()
    
    st.title("📝 Định giá Đơn lẻ (Single Predict)")
    st.write("""
    Nhập thông tin chi tiết về căn nhà để nhận dự đoán giá trị tức thời.
    Các trường không hiển thị sẽ tự động sử dụng giá trị mặc định.
    """)
    
    # Main content area
    st.divider()
    
    # Tạo form input với feature groups
    st.write("### 📋 Thông tin Căn nhà")
    user_input = create_input_form(FEATURE_GROUPS)
    
    # Add predict button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_clicked = st.button(
            "🔮 Dự đoán Giá",
            width="stretch",
            type="primary"
        )
    
    with col2:
        reset_clicked = st.button(
            "🔄 Reset",
            width="stretch"
        )
    
    if reset_clicked:
        st.rerun()
    
    # Xử lý dự đoán
    if predict_clicked:
        try:
            # Validate input
            is_valid, error_msg = validate_house_data(user_input)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return
            
            # Chuẩn bị payload (merge user input + defaults)
            payload = prepare_prediction_payload(user_input, DEFAULT_FEATURES)
            
            # Gọi API
            with st.spinner("⏳ Đang dự đoán..."):
                client = get_api_client()
                result = client.predict(payload)
            
            # Lưu vào session state
            st.session_state.last_prediction = result
            
            # Hiển thị kết quả
            st.divider()
            st.success("✅ Dự đoán thành công!")
            st.divider()
            
            display_prediction_result(result)
            
            # Lưu kết quả vào session
            st.toast(f"💾 Kết quả: {format_currency(result.get('predicted_price', 0))}", icon="✨")
        
        except APIError as e:
            st.error(f"❌ Lỗi API: {str(e)}")
            logger.error(f"API error in single predict: {e}")
        
        except Exception as e:
            st.error(f"❌ Lỗi không xác định: {str(e)}")
            logger.error(f"Unexpected error in single predict: {e}")


if __name__ == "__main__":
    main()
