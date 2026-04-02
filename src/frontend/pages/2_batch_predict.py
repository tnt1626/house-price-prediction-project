"""
Batch Prediction Page - Tab 2

Định giá hàng loạt cho nhiều căn nhà từ file CSV.
Hỗ trợ upload, xử lý lỗi, và download kết quả.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List

from src.frontend.api_client import get_api_client, APIError
from src.frontend.config import DEFAULT_FEATURES, MAX_BATCH_SIZE, ALLOWED_FILE_TYPES
from src.frontend.ui_components import display_batch_results, download_button_csv
from src.frontend.utils import parse_csv_for_batch, initialize_session_state

logger = logging.getLogger(__name__)


def merge_with_defaults(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge mỗi record từ CSV với default features.
    
    Args:
        records: List các records từ CSV
        
    Returns:
        List các records với defaults merged
    """
    merged = []
    for record in records:
        # Start with defaults
        merged_record = DEFAULT_FEATURES.copy()
        # Override with CSV values
        merged_record.update(record)
        merged.append(merged_record)
    
    return merged


def get_csv_feature_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Tạo mapping giữa CSV columns và feature names.
    Nếu column name giống feature name, auto-map.
    
    Args:
        df: DataFrame từ CSV
        
    Returns:
        Dict mapping
    """
    mapping = {}
    
    # Auto-detect từ DataFrame columns
    for col in df.columns:
        # Giả sử CSV column names khớp với feature names hoặc có tiền tố
        mapping[col] = col
    
    return mapping


def validate_batch_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate batch CSV data.
    
    Args:
        df: DataFrame từ CSV
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if df.empty:
        return False, "File CSV trống"
    
    if len(df) > MAX_BATCH_SIZE:
        return False, f"Quá nhiều records (max: {MAX_BATCH_SIZE}), có {len(df)}"
    
    # Check required columns (at least some feature columns)
    required_columns = ["LotArea", "OverallQual", "GrLivArea"]
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        st.info(f"💡 Tip: File CSV có thể thiếu các cột: {missing}. Những cột này sẽ dùng giá trị mặc định.")
    
    return True, ""


def main():
    """Main function for Batch Predict page"""
    st.set_page_config(page_title="Định giá Batch", page_icon="📁", layout="wide")
    initialize_session_state()
    
    st.title("📁 Định giá Hàng loạt (Batch Predict)")
    st.write(f"""
    Tải lên file CSV chứa danh sách các căn nhà để dự đoán giá hàng loạt.
    
    **Hạn chế:**
    - Tối đa {MAX_BATCH_SIZE} căn nhà
    - Format: CSV với header
    - Cột không có trong file sẽ dùng giá trị mặc định
    
    **Ví dụ cột cần có:** LotArea, OverallQual, GrLivArea, ...
    """)
    
    st.divider()
    
    # Upload file section
    st.write("### 📄 Upload File CSV")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=ALLOWED_FILE_TYPES,
        key="batch_csv_upload"
    )
    
    if uploaded_file is None:
        st.info("👈 Chọn file CSV để bắt đầu")
        return
    
    try:
        # Read CSV
        with st.spinner("⏳ Đang đọc file..."):
            df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Tải file thành công: {len(df)} dòng")
        
        # Preview data
        st.write("### 👀 Preview Data")
        st.dataframe(df.head(10), width="stretch")
        
        # Validate
        is_valid, error_msg = validate_batch_data(df)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        st.divider()
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            process_clicked = st.button(
                "🚀 Dự đoán Batch",
                width=True,
                type="primary"
            )
        
        if process_clicked:
            try:
                # Create feature mapping
                feature_mapping = get_csv_feature_mapping(df)
                
                # Parse records
                records, errors = parse_csv_for_batch(df, feature_mapping)
                
                if errors:
                    st.warning("⚠️ Một số lỗi khi parse records:")
                    for error in errors:
                        st.write(f"  - {error}")
                
                if not records:
                    st.error("❌ Không có records hợp lệ")
                    return
                
                # Merge with defaults
                records_with_defaults = merge_with_defaults(records)
                
                # Chuẩn bị batch payload
                batch_payload = {"houses": records_with_defaults}
                
                # Gọi API batch predict
                with st.spinner(f"⏳ Đang dự đoán {len(records)} căn nhà..."):
                    client = get_api_client()
                    result = client.predict_batch(batch_payload)
                
                st.success(f"✅ Dự đoán thành công!")
                st.divider()
                
                # Display results
                display_batch_results(result)
                
                # Prepare download dataframe
                predictions = result.get("predictions", [])
                if predictions:
                    df_results = pd.DataFrame([
                        {
                            "STT": idx + 1,
                            "Predicted Price": pred.get("predicted_price", 0),
                            "Confidence": pred.get("confidence", 0),
                            "Model": pred.get("model_name", "Unknown")
                        }
                        for idx, pred in enumerate(predictions)
                    ])
                    
                    st.divider()
                    st.write("### 📥 Download Kết quả")
                    download_button_csv(
                        df_results,
                        filename="batch_predictions.csv",
                        button_key="download_batch"
                    )
                
                # Store in session
                st.session_state.last_prediction = result
                st.toast("💾 Kết quả đã được lưu", icon="✨")
            
            except APIError as e:
                st.error(f"❌ Lỗi API: {str(e)}")
                logger.error(f"API error in batch predict: {e}")
            
            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {str(e)}")
                logger.error(f"Unexpected error in batch predict: {e}")
    
    except pd.errors.ParserError as e:
        st.error(f"❌ Lỗi đọc CSV: {str(e)}")
        logger.error(f"CSV parsing error: {e}")
    
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
