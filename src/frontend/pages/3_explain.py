"""
Explanation Page - Tab 3

Dự đoán giá với giải thích SHAP (XAI - Explainable AI).
Hiển thị biểu đồ Waterfall, Bar chart, và chi tiết feature contributions.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import logging
from typing import Dict, Any

from src.frontend.api_client import get_api_client, APIError
from src.frontend.config import FEATURE_GROUPS, DEFAULT_FEATURES
from src.frontend.ui_components import (
    create_input_form,
    display_prediction_result,
    display_shap_explanations
)
from src.frontend.utils import format_currency, validate_house_data, initialize_session_state

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
    """Main function for Explanation/XAI page"""
    st.set_page_config(page_title="Phân tích Chuyên sâu", page_icon="🔍", layout="wide")
    initialize_session_state()
    
    st.title("🔍 Phân tích Chuyên sâu (Predict with Explanation)")
    st.write("""
    Nhận dự đoán giá nhà kèm theo giải thích chi tiết về tác động của từng feature.
    
    **Tính năng:**
    - 💰 Dự đoán giá chính xác
    - 📊 Waterfall chart: Hiển thị cách mỗi feature ảnh hưởng đến giá
    - 📈 Top features: Xác định những yếu tố quyết định nhất
    - 📋 Chi tiết: Giá trị gốc và mức độ đóng góp của mỗi feature
    """)
    
    # Main content area
    st.divider()
    
    # Tạo form input với feature groups
    st.write("### 📋 Thông tin Căn nhà")
    user_input = create_input_form(FEATURE_GROUPS)
    
    # Top features selector
    st.write("### ⚙️ Tuỳ chọn Giải thích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_features = st.slider(
            "Số lượng TOP Features hiển thị",
            min_value=3,
            max_value=20,
            value=10,
            key="top_features_slider"
        )
    
    st.divider()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_clicked = st.button(
            "🔮 Dự đoán & Giải thích",
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
    
    # Xử lý dự đoán + giải thích
    if predict_clicked:
        try:
            # Validate input
            is_valid, error_msg = validate_house_data(user_input)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return
            
            # Chuẩn bị payload
            payload = prepare_prediction_payload(user_input, DEFAULT_FEATURES)
            
            # Gọi API predict với explanation
            with st.spinner("⏳ Đang phân tích..."):
                client = get_api_client()
                result = client.predict_with_explanation(payload, top_features=top_features)
            
            # Store in session
            st.session_state.last_prediction = result
            
            # Hiển thị kết quả
            st.divider()
            st.success("✅ Phân tích thành công!")
            st.divider()
            
            # Hiển thị basic prediction info
            st.write("### 💰 Dự đoán Giá")
            display_prediction_result(result, show_details=True)
            
            st.divider()
            
            # Hiển thị SHAP explanations với biểu đồ
            st.write("### 📊 SHAP - Giải thích Chi tiết")
            display_shap_explanations(result)
            
            # Save result
            st.toast(f"💾 Kết quả: {format_currency(result.get('predicted_price', 0))}", icon="✨")
        
        except APIError as e:
            st.error(f"❌ Lỗi API: {str(e)}")
            
            # Provide helpful suggestions
            if "SHAP explainer not loaded" in str(e).lower():
                st.info("""
                💡 **Gợi ý:** Tính năng giải thích SHAP chưa sẵn sàng.
                
                Vui lòng:
                1. Đảm bảo mô hình đã được tải với SHAP explainer
                2. Liên hệ Admin để huấn luyện lại mô hình với XAI support
                3. Thử lại sau
                """)
            
            logger.error(f"API error in explain: {e}")
        
        except Exception as e:
            st.error(f"❌ Lỗi không xác định: {str(e)}")
            logger.error(f"Unexpected error in explain: {e}")


if __name__ == "__main__":
    main()
