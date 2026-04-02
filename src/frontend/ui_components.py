"""
UI Components Module

Các component UI tái sử dụng cho Streamlit interface.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from config import (
    COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL,
    PLOT_HEIGHT, PLOT_WIDTH
)
from utils import format_currency, format_confidence, get_confidence_emoji

logger = logging.getLogger(__name__)


def display_health_status(health_data: Dict[str, Any]):
    """
    Hiển thị trạng thái health check.
    
    Args:
        health_data: Dict từ API health endpoint
    """
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = health_data.get("status", "unknown").upper()
            emoji = "🟢" if status == "OK" else "🟡" if status == "DEGRADED" else "🔴"
            st.metric("Status", f"{emoji} {status}")
        
        with col2:
            model_loaded = health_data.get("model_loaded", False)
            st.metric("Model Loaded", "✅ Yes" if model_loaded else "❌ No")
        
        with col3:
            model_name = health_data.get("model_name", "None")
            st.metric("Current Model", model_name)
    except Exception as e:
        logger.error(f"Error displaying health status: {e}")
        st.error(f"Lỗi hiển thị trạng thái: {str(e)}")


def display_prediction_result(prediction: Dict[str, Any], show_details: bool = True):
    """
    Hiển thị kết quả dự đoán (Single Predict).
    
    Args:
        prediction: Dict chứa predicted_price, confidence, model_name
        show_details: Có hiển thị chi tiết hay không
    """
    try:
        predicted_price = prediction.get("predicted_price", 0)
        confidence = prediction.get("confidence", 0)
        model_name = prediction.get("model_name", "Unknown")
        
        # Hiển thị result chính
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "💰 Giá Dự đoán",
                format_currency(predicted_price),
                delta=None
            )
        
        with col2:
            emoji = get_confidence_emoji(confidence)
            st.metric(
                "📊 Độ Tin cậy",
                format_confidence(confidence),
                delta=emoji
            )
        
        if show_details:
            st.divider()
            st.write(f"**Mô hình sử dụng:** {model_name}")
    except Exception as e:
        logger.error(f"Error displaying prediction result: {e}")
        st.error(f"Lỗi hiển thị kết quả: {str(e)}")


def display_shap_explanations(explanation_data: Dict[str, Any]):
    """
    Hiển thị SHAP explanations với biểu đồ và chi tiết.
    
    Args:
        explanation_data: Dict chứa predicted_price, base_value, explanations list
    """
    try:
        predicted_price = explanation_data.get("predicted_price", 0)
        base_value = explanation_data.get("base_value", 0)
        explanations = explanation_data.get("explanations", [])
        
        # Header
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Giá Dự đoán", format_currency(predicted_price))
        with col2:
            st.metric("📈 Base Value (Mean)", format_currency(base_value))
        
        st.divider()
        
        if not explanations:
            st.info("Không có giải thích feature nào có sẵn")
            return
        
        # ===== Biểu đồ Waterfall: Hiển thị từng feature contribution =====
        st.subheader("📊 Waterfall - Đóng góp từng Feature")
        
        try:
            # Chuẩn bị dữ liệu cho waterfall
            features = []
            values = []
            colors_list = []
            
            # Add base value
            features.append("Base Value")
            values.append(base_value)
            colors_list.append("#cccccc")
            
            # Add contributions
            for exp in explanations:
                feature_name = exp.get("feature_name", "Unknown")
                shap_value = exp.get("shap_value", 0)
                contribution_type = exp.get("contribution_type", "neutral")
                
                features.append(feature_name)
                values.append(shap_value)
                
                if contribution_type == "positive":
                    colors_list.append(COLOR_POSITIVE)
                elif contribution_type == "negative":
                    colors_list.append(COLOR_NEGATIVE)
                else:
                    colors_list.append(COLOR_NEUTRAL)
            
            # Add final prediction
            features.append("Predicted Price")
            values.append(predicted_price - base_value)
            colors_list.append("#333333")
            
            # Tạo waterfall chart
            fig = go.Figure(go.Waterfall(
                x=features,
                y=values,
                increasing=dict(marker=dict(color=COLOR_POSITIVE)),
                decreasing=dict(marker=dict(color=COLOR_NEGATIVE)),
                totals=dict(marker=dict(color="#666666")),
                connector={"line": {"color": "rgba(0,0,0,0.2)"}},
                textposition="auto"
            ))
            
            fig.update_layout(
                title="Feature Contributions to Price Prediction",
                xaxis_title="Features",
                yaxis_title="Contribution ($)",
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            logger.error(f"Error creating waterfall chart: {e}")
            st.warning(f"Lỗi tạo waterfall chart: {str(e)}")
        
        st.divider()
        
        # ===== Bar Chart: TOP Features =====
        st.subheader("🔝 Top Features - Impact Analysis")
        
        try:
            # Chuẩn bị dữ liệu
            df_explanations = pd.DataFrame([
                {
                    "Feature": exp.get("feature_name", "Unknown"),
                    "Impact": abs(exp.get("shap_value", 0)),
                    "Direction": "Positive" if exp.get("contribution_type") == "positive" else "Negative",
                    "Value": exp.get("original_value", "N/A")
                }
                for exp in explanations
            ])
            
            # Sort by Impact
            df_explanations = df_explanations.sort_values("Impact", ascending=True)
            
            # Tạo bar chart (horizontal)
            fig = px.bar(
                df_explanations,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Positive": COLOR_POSITIVE, "Negative": COLOR_NEGATIVE},
                title="Feature Impact Magnitude",
                labels={"Impact": "SHAP Value ($)", "Feature": "Features"}
            )
            
            fig.update_layout(
                height=max(300, len(df_explanations) * 25),
                width=PLOT_WIDTH,
                hovermode="y unified"
            )
            
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            st.warning(f"Lỗi tạo bar chart: {str(e)}")
        
        st.divider()
        
        # ===== Chi tiết từng feature =====
        st.subheader("📋 Chi tiết từng Feature Contribution")
        
        for idx, exp in enumerate(explanations, 1):
            feature_name = exp.get("feature_name", "Unknown")
            original_value = exp.get("original_value", "N/A")
            shap_value = exp.get("shap_value", 0)
            contribution_type = exp.get("contribution_type", "neutral")
            
            # Emoji indicator
            emoji = "⬆️" if contribution_type == "positive" else "⬇️"
            direction_text = "TĂNG GIÁ" if contribution_type == "positive" else "GIẢM GIÁ"
            
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.write(f"**#{idx}**")
            
            with col2:
                st.write(f"{emoji} **{feature_name}**")
                st.caption(f"Value: {original_value}")
            
            with col3:
                st.write(f"{direction_text}")
                st.write(f"{format_currency(abs(shap_value))}")
    
    except Exception as e:
        logger.error(f"Error displaying SHAP explanations: {e}")
        st.error(f"Lỗi hiển thị giải thích: {str(e)}")


def display_batch_results(batch_result: Dict[str, Any]):
    """
    Hiển thị kết quả batch predictions dưới dạng bảng.
    
    Args:
        batch_result: Dict chứa predictions list
    """
    try:
        predictions = batch_result.get("predictions", [])
        total_processed = batch_result.get("total_processed", 0)
        
        st.write(f"**Tổng xử lý:** {total_processed} căn nhà")
        
        if not predictions:
            st.warning("Không có kết quả dự đoán")
            return
        
        # Tạo DataFrame
        df_results = pd.DataFrame([
            {
                "STT": idx + 1,
                "Giá Dự đoán": format_currency(pred.get("predicted_price", 0)),
                "Độ Tin cậy": format_confidence(pred.get("confidence", 0)),
                "Mô hình": pred.get("model_name", "Unknown")
            }
            for idx, pred in enumerate(predictions)
        ])
        
        st.dataframe(df_results, width="stretch", hide_index=True)
        
        # Statistics
        st.divider()
        st.subheader("📊 Thống kê")
        
        prices = [pred.get("predicted_price", 0) for pred in predictions]
        confidences = [pred.get("confidence", 0) for pred in predictions]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Price", format_currency(min(prices) if prices else 0))
        
        with col2:
            st.metric("Max Price", format_currency(max(prices) if prices else 0))
        
        with col3:
            avg_price = sum(prices) / len(prices) if prices else 0
            st.metric("Avg Price", format_currency(avg_price))
        
        with col4:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.metric("Avg Confidence", format_confidence(avg_confidence))
        
        # Distribution chart
        fig = go.Figure(data=[
            go.Histogram(x=prices, name="Predicted Prices", nbinsx=20)
        ])
        fig.update_layout(
            title="Price Distribution",
            xaxis_title="Predicted Price ($)",
            yaxis_title="Frequency",
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")
    
    except Exception as e:
        logger.error(f"Error displaying batch results: {e}")
        st.error(f"Lỗi hiển thị batch results: {str(e)}")


def create_input_form(feature_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tạo form input dựa trên feature groups.
    
    Args:
        feature_groups: Dict chứa các nhóm features
        
    Returns:
        Dict chứa các giá trị input từ user
    """
    try:
        form_data = {}
        
        for group_name, features in feature_groups.items():
            st.subheader(f"📌 {group_name}")
            
            with st.container():
                # Tạo columns động dựa trên số lượng features
                cols = st.columns(min(2, len(features)))
                
                for idx, (feature_key, feature_info) in enumerate(features.items()):
                    col_idx = idx % len(cols)
                    
                    with cols[col_idx]:
                        label, input_type, *params = feature_info
                        
                        if input_type == "slider":
                            min_val, max_val, default_val = params
                            form_data[feature_key] = st.slider(
                                label,
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"input_{feature_key}"
                            )
                        
                        elif input_type == "number":
                            default_val = params[0] if params else 0
                            form_data[feature_key] = st.number_input(
                                label,
                                value=float(default_val),
                                step=1.0,
                                key=f"input_{feature_key}"
                            )
                        
                        elif input_type == "text":
                            default_val = params[0] if params else ""
                            form_data[feature_key] = st.text_input(
                                label,
                                value=default_val,
                                key=f"input_{feature_key}"
                            )
        
        return form_data
    except Exception as e:
        logger.error(f"Error creating input form: {e}")
        st.error(f"Lỗi tạo form: {str(e)}")
        return {}


def file_uploader_csv(key: str = "csv_upload") -> Optional[Any]:
    """
    Widget upload file CSV.
    
    Args:
        key: Session state key
        
    Returns:
        Uploaded file object hoặc None
    """
    try:
        uploaded_file = st.file_uploader(
            "📄 Chọn file CSV",
            type=["csv"],
            key=key
        )
        return uploaded_file
    except Exception as e:
        logger.error(f"Error in file uploader: {e}")
        st.error(f"Lỗi upload file: {str(e)}")
        return None


def download_button_csv(df: pd.DataFrame, filename: str = "predictions.csv", button_key: str = None):
    """
    Button download DataFrame dưới dạng CSV.
    
    Args:
        df: Pandas DataFrame
        filename: Tên file download
        button_key: Session state key
    """
    try:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=button_key
        )
    except Exception as e:
        logger.error(f"Error creating download button: {e}")
        st.error(f"Lỗi tạo button download: {str(e)}")
