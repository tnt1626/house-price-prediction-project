"""
Streamlit Main Application

Entry point cho ứng dụng định giá nhà.
Xử lý Sidebar (Health check, Model management) và định hướng tới các pages.
"""

import streamlit as st
import logging
from datetime import datetime

from api_client import get_api_client, APIError, HealthCheckError
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    MSG_HEALTH_OK, MSG_HEALTH_ERROR, MSG_HEALTH_WARNING,
    MSG_MODEL_LOADING, MSG_MODEL_LOADED, MSG_MODEL_FAILED
)
from ui_components import display_health_status
from utils import initialize_session_state

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Streamlit Configuration
# ============================================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State Initialization
# ============================================================================
initialize_session_state()


# ============================================================================
# SIDEBAR: Health Check & Model Management
# ============================================================================
def sidebar_health_check():
    """Hiển thị health check và trạng thái server"""
    st.subheader("🏥 Trạng thái Server")
    
    try:
        client = get_api_client()
        health_data = client.health_check()
        
        # Lưu health data vào session
        st.session_state.api_health = health_data
        
        # Display health status
        display_health_status(health_data)
        
        st.success(MSG_HEALTH_OK)
        return health_data
    
    except HealthCheckError as e:
        st.error(MSG_HEALTH_ERROR)
        st.error(f"Chi tiết: {str(e)}")
        logger.error(f"Health check error: {e}")
        return None
    
    except Exception as e:
        st.error(MSG_HEALTH_ERROR)
        logger.error(f"Unexpected error in health check: {e}")
        return None


def sidebar_model_management():
    """Quản lý tải và chuyển đổi mô hình"""
    st.subheader("📦 Quản lý Mô hình")
    
    try:
        client = get_api_client()
        
        # Get available models
        with st.spinner("⏳ Lấy danh sách mô hình..."):
            models = client.get_models()
        
        if not models:
            st.warning("❌ Không tìm thấy mô hình nào")
            return
        
        # Select model
        selected_model = st.selectbox(
            "Chọn Mô hình",
            options=models,
            key="model_selector"
        )
        
        # Load model button
        if st.button("📥 Tải Mô hình", width="stretch", type="primary"):
            try:
                with st.spinner(MSG_MODEL_LOADING):
                    result = client.load_model(selected_model)
                
                st.success(MSG_MODEL_LOADED)
                st.session_state.model_loaded = True
                st.session_state.current_model = selected_model
                logger.info(f"Model loaded: {selected_model}")
                st.rerun()
            
            except APIError as e:
                st.error(MSG_MODEL_FAILED)
                st.error(f"Chi tiết: {str(e)}")
                logger.error(f"Model load error: {e}")
            
            except Exception as e:
                st.error(MSG_MODEL_FAILED)
                logger.error(f"Unexpected error loading model: {e}")
        
        # Show current model
        if st.session_state.model_loaded and st.session_state.current_model:
            st.info(f"✅ Mô hình hiện tại: **{st.session_state.current_model}**")
    
    except APIError as e:
        st.error(f"❌ Lỗi lấy danh sách mô hình: {str(e)}")
        logger.error(f"Error getting models: {e}")
    
    except Exception as e:
        st.error(f"❌ Lỗi quản lý mô hình: {str(e)}")
        logger.error(f"Unexpected error in model management: {e}")


def main():
    """Main function"""
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Health check
        sidebar_health_check()
        
        st.divider()
        
        # Model management
        sidebar_model_management()
        
        st.divider()
        
        # Info section
        st.subheader("ℹ️ Thông tin")
        st.write("""
        **Hệ thống Dự đoán Giá Nhà**
        
        - **Mô hình:** CatBoost
        - **Dataset:** Ames Housing
        - **Features:** ~75 đặc trưng
        - **XAI:** SHAP Explanations
        """)
        
        st.divider()
        
        # Footer
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    st.title("🏠 Hệ thống Dự đoán Giá Nhà")
    
    st.write("""
    Chào mừng đến với hệ thống dự đoán giá nhà sử dụng CatBoost.
    
    ### 🚀 Bắt đầu:
    1. **Kiểm tra Server:** Xem trạng thái ở Sidebar
    2. **Tải Mô hình:** Chọn và tải mô hình từ Sidebar
    3. **Chọn tính năng:** 
       - 📝 **Định giá Đơn lẻ** - Dự đoán 1 căn nhà
       - 📁 **Định giá Batch** - Dự đoán từ file CSV
       - 🔍 **Phân tích Chuyên sâu** - Xem giải thích SHAP
    
    ### 📋 Các tính năng:
    - ✅ Dự đoán giá chính xác
    - ✅ Upload file CSV hàng loạt
    - ✅ Giải thích SHAP (XAI)
    - ✅ Xử lý lỗi toàn diện
    - ✅ Download kết quả
    
    💡 **Tip:** Các trường dữ liệu không bắt buộc sẽ tự động sử dụng giá trị mặc định.
    """)
    
    st.divider()
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("⚠️ Vui lòng tải một mô hình từ Sidebar để bắt đầu!")
    else:
        st.success(f"✅ Mô hình **{st.session_state.current_model}** đã được tải và sẵn sàng!")
    
    st.divider()
    
    # Feature groups info
    st.write("### 📊 Nhóm Dữ liệu")
    st.markdown("""
    Ứng dụng tự động xử lý ~60 trường dữ liệu phụ với giá trị mặc định.
    Người dùng chỉ cần nhập 15 trường chính có ảnh hưởng lớn nhất:
    
    - **Đánh giá & Vị trí:** Năm xây dựng, Chất lượng, Khu vực...
    - **Diện tích:** Đất, Sinh hoạt, Tầng hầm...
    - **Phòng ốc & Tiện ích:** Phòng ngủ, Tắm, Garage...
    """)


if __name__ == "__main__":
    main()