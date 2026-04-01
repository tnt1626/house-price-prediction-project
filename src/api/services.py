"""
Service layer for business logic.
Handles model loading, preprocessing, and predictions, including SHAP-based explanations.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging

from src.core.config import MODELS_DIR, SCALERS_DIR, EXPLAINER_DIR, SCHEMA_TO_DATA_MAPPING
from src.api.schemas import HousePriceInput, PredictionResponse
from src.ml_pipeline.explainability import ModelExplainer

 
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making price predictions."""
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Initialize prediction service.
        
        Parameters:
            model_path: Path to trained model
            scaler_path: Path to preprocessing scaler
        """
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.model_name = "None"
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_preprocessor(scaler_path)
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load trained model from disk.
        
        Parameters:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.model_name = model_path.stem
            logger.info(f"[OK] Model loaded: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_preprocessor(self, scaler_path: Path) -> bool:
        """
        Load preprocessing pipeline from disk.
        
        Parameters:
            scaler_path: Path to preprocessor file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            scaler_path = Path(scaler_path)
            if not scaler_path.exists():
                logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            self.preprocessor = joblib.load(scaler_path)
            logger.info(f"[OK] Preprocessor loaded: {scaler_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.model is not None
    
    def input_to_dataframe(self, data: HousePriceInput) -> pd.DataFrame:
        """
        Convert input schema to DataFrame.
        Renames columns from schema names to data column names if needed.
        Ensures all required columns exist with proper defaults.
        
        Parameters:
            data: Input data
            
        Returns:
            DataFrame with feature values (using actual data column names)
        """
        # Get all field names from the schema
        schema_fields = set(HousePriceInput.model_fields.keys())
        
        # Convert to dict, including all fields with exclude_unset=False
        data_dict = data.model_dump(exclude_unset=False)
        
        # Apply column name mapping from schema to data column names
        mapped_dict = {}
        for k, v in data_dict.items():
            # Get the mapped column name from SCHEMA_TO_DATA_MAPPING, or use original if not mapped
            actual_col_name = SCHEMA_TO_DATA_MAPPING.get(k, k)
            mapped_dict[actual_col_name] = v
        
        # Also add columns using schema names as fallback
        for field_name in schema_fields:
            if field_name not in mapped_dict:
                actual_col_name = SCHEMA_TO_DATA_MAPPING.get(field_name, field_name)
                if actual_col_name not in mapped_dict:
                    mapped_dict[actual_col_name] = data_dict.get(field_name, None)
        
        # Create DataFrame with a single row
        df = pd.DataFrame([mapped_dict])
        
        # Ensure no duplicate columns, remove fully NaN columns to avoid issues
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.debug(f"[DEBUG] Input DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
        return df
    
    def _ensure_preprocessor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns expected by preprocessor are present in DataFrame.
        This handles the case where preprocessor was trained on specific columns
        but prediction input may have missing columns.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            DataFrame with all required columns
        """
        if self.preprocessor is None:
            return df
        
        try:
            # Try to get feature names from preprocessor
            original_columns = None
            
            # Get all input feature names expected by preprocessor
            if hasattr(self.preprocessor, 'named_steps'):
                # Get the add_domain step's output columns if available
                if 'add_domain' in self.preprocessor.named_steps:
                    # The add_domain step doesn't change column names drastically
                    pass
                
                # Get the preprocessing step's column names
                if 'preproc' in self.preprocessor.named_steps:
                    preproc = self.preprocessor.named_steps['preproc']
                    if hasattr(preproc, 'named_steps'):
                        if 'ct' in preproc.named_steps:
                            ct = preproc.named_steps['ct']
                            if hasattr(ct, 'transformers_'):
                                # List all columns that ColumnTransformer expects
                                all_transformer_cols = []
                                for name, transformer, cols in ct.transformers_:
                                    all_transformer_cols.extend(cols)
                                original_columns = all_transformer_cols
            
            if original_columns:
                # Add missing columns with default values
                for col in original_columns:
                    if col not in df.columns:
                        # Infer type and add appropriate default
                        df[col] = 0  # Default to 0 for numeric columns
                        logger.debug(f"[DEBUG] Added missing column '{col}' with default value 0")
        
        except Exception as e:
            logger.warning(f"Could not ensure preprocessor columns: {e}")
        
        return df
    
    def predict_single(self, input_data: HousePriceInput) -> Dict[str, Any]:
        """
        Thực hiện dự báo cho một căn nhà duy nhất.
        """
        if not self.is_ready():
            raise RuntimeError("Model chưa được tải. Vui lòng kiểm tra lại đường dẫn model.")
        
        try:
            # 1. Chuyển đổi dữ liệu từ Schema (Pydantic) sang DataFrame của Pandas
            # Hàm này sẽ mapping lại tên cột từ Schema sang tên cột thật trong data (LotArea, LotFrontage, ...)
            df = self.input_to_dataframe(input_data)
            
            # 2. Đảm bảo tất cả các cột mà preprocessor yêu cầu đều có mặt (tránh lỗi KeyError)
            df = self._ensure_preprocessor_columns(df)
            
            # 3. Xử lý logic Tiền xử lý và Dự báo
            # Kiểm tra xem self.model có phải là một Pipeline của sklearn không
            from sklearn.pipeline import Pipeline as SklearnPipeline
            
            if isinstance(self.model, SklearnPipeline):
                # Nếu model là một Pipeline (đã bao gồm bước transform), ta truyền trực tiếp df gốc vào
                # Điều này tránh lỗi "Double Transformation" dẫn đến mất cột 'LotFrontage'
                logger.info("Model is a full Pipeline. Predicting directly from raw features.")
                X_input_for_model = df
                
                # QUAN TRỌNG: Đừng transform df lần nữa vì model.predict() sẽ tự làm
                # Thay vào đó, ta trích xuất X_transformed từ các bước transform của pipeline
                # Để SHAP có dữ liệu đã xử lý (engineered features)
                try:
                    # Lấy bước preprocessor từ pipeline
                    preprocessor_steps = self.model.named_steps
                    if 'preprocessor' in preprocessor_steps:
                        # Transform bằng preprocessor từ pipeline
                        X_transformed = preprocessor_steps['preprocessor'].transform(df)
                    else:
                        # Nếu không tìm thấy, transform bằng self.preprocessor
                        if self.preprocessor:
                            X_transformed = self.preprocessor.transform(df)
                        else:
                            X_transformed = df
                except Exception:
                    # Fallback: dùng self.preprocessor
                    if self.preprocessor:
                        X_transformed = self.preprocessor.transform(df)
                    else:
                        X_transformed = df
            else:
                # Nếu model chỉ là thuật toán thuần túy (XGB, CatBoost), ta phải transform thủ công
                if self.preprocessor is None:
                    logger.warning("No preprocessor loaded, using raw features")
                    X_transformed = df
                else:
                    X_transformed = self.preprocessor.transform(df)
                X_input_for_model = X_transformed

            # 4. Chuyển đổi dữ liệu đã xử lý sang DataFrame nếu nó đang là NumPy array
            # Bước này cực kỳ quan trọng để module XAI/SHAP không bị lỗi '.columns'
            if not isinstance(X_transformed, pd.DataFrame):
                try:
                    # Lấy tên các cột sau khi biến đổi từ preprocessor
                    cols = self.preprocessor.get_feature_names_out()
                except Exception:
                    # Nếu không lấy được tên thật, dùng tên tạm f0, f1... để không bị dừng chương trình
                    cols = [f"f{i}" for i in range(X_transformed.shape[1])]
                
                X_transformed = pd.DataFrame(X_transformed, columns=cols)
            
            # 5. Thực hiện dự báo giá
            # Nếu model là pipeline, nó nhận X_input_for_model (df gốc)
            # Nếu model là estimator, nó nhận X_input_for_model (X_transformed)
            prediction = self.model.predict(X_input_for_model)
            predicted_price = float(prediction[0])
            
            # 6. Tạo giải thích SHAP nếu có Explainer
            if self.explainer:
                try:
                    # Truyền X_transformed đã là DataFrame vào đây
                    self.explainer.get_local_explanation(X_transformed, input_data)
                except Exception as e:
                    logger.warning(f"Không thể tạo giải thích XAI: {e}")

            return {
                "predicted_price": predicted_price,
                "confidence": 0.85,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data_list: list) -> list:
        """
        Make predictions for multiple houses.
        
        Parameters:
            data_list: List of HousePriceInput objects
            
        Returns:
            List of predictions
        """
        predictions = []
        for data in data_list:
            try:
                pred = self.predict_single(data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict for house {len(predictions)}: {e}")
                predictions.append({
                    "predicted_price": None,
                    "confidence": 0.0,
                    "model_name": self.model_name,
                    "error": str(e)
                })
        
        return predictions
    
    def load_explainer(self, explainer_path: Optional[Path] = None) -> bool:
        """
        Load SHAP explainer from disk.
        
        Parameters:
            explainer_path: Path to explainer file (default: EXPLAINER_DIR/shap_explainer.joblib)
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None or self.preprocessor is None:
            logger.error("Model and preprocessor must be loaded before explainer")
            return False
        
        if explainer_path is None:
            explainer_path = EXPLAINER_DIR / "shap_explainer.joblib"
        
        try:
            explainer_path = Path(explainer_path)
            if not explainer_path.exists():
                logger.warning(f"Explainer file not found: {explainer_path}")
                return False
            
            # Get original feature names from schema
            original_feature_names = list(HousePriceInput.model_fields.keys())
            
            self.explainer = ModelExplainer.load(
                self.model, 
                self.preprocessor, 
                explainer_path,
                original_feature_names=original_feature_names
            )
            logger.info(f"[OK] Explainer loaded: {explainer_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load explainer: {e}")
            return False
    
    def predict_and_explain(
        self,
        data: HousePriceInput,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Thực hiện dự báo và cung cấp giải thích dựa trên SHAP, xử lý lỗi mảng NumPy.
        """
        if not self.is_ready():
            raise RuntimeError("Model chưa được tải. Vui lòng kiểm tra lại đường dẫn.")
        
        if self.explainer is None:
            raise RuntimeError("Explainer chưa được tải. Vui lòng kiểm tra lại cấu hình XAI.")
        
        try:
            # 1. Chuyển đổi Schema thành DataFrame (Pydantic V2 dùng model_dump)
            df = self.input_to_dataframe(data)
            original_input = data.model_dump() 
            
            # 2. Xử lý logic Tiền xử lý dựa trên loại Model (Tránh lỗi Double Transformation)
            from sklearn.pipeline import Pipeline as SklearnPipeline
            
            # Nếu model là một Pipeline hoàn chỉnh (đã có bước preprocessing bên trong)
            if isinstance(self.model, SklearnPipeline):
                logger.info("Model is a full Pipeline. Using raw features for prediction.")
                X_input_for_model = df # Dùng DataFrame gốc cho model
                
                # QUAN TRỌNG: Đừng transform df lần nữa vì model.predict() sẽ tự làm
                # Thay vào đó, ta trích xuất X_processed từ các bước transform của pipeline
                # Để SHAP có dữ liệu đã xử lý (engineered features)
                try:
                    # Lấy bước preprocessor từ pipeline
                    preprocessor_steps = self.model.named_steps
                    if 'preprocessor' in preprocessor_steps:
                        # Transform bằng preprocessor từ pipeline
                        df_processed = preprocessor_steps['preprocessor'].transform(df)
                    else:
                        # Nếu không tìm thấy, transform bằng self.preprocessor
                        if self.preprocessor:
                            df_processed = self.preprocessor.transform(df)
                        else:
                            df_processed = df
                except Exception:
                    # Fallback: dùng self.preprocessor
                    if self.preprocessor:
                        df_processed = self.preprocessor.transform(df)
                    else:
                        df_processed = df
            else:
                # Nếu model chỉ là thuật toán (XGB, CatBoost...), ta phải tự transform trước
                if self.preprocessor:
                    df_processed = self.preprocessor.transform(df)
                else:
                    df_processed = df
                X_input_for_model = df_processed

            # 3. QUAN TRỌNG: Đảm bảo df_processed LUÔN là DataFrame có tên cột
            # Điều này để SHAP không bị lỗi 'numpy.ndarray object has no attribute columns'
            if not isinstance(df_processed, pd.DataFrame):
                try:
                    # Lấy tên cột thật từ preprocessor (nếu có)
                    cols = self.preprocessor.get_feature_names_out()
                except Exception:
                    # Nếu không lấy được, dùng tên cột tạm f0, f1...
                    cols = [f"f{i}" for i in range(df_processed.shape[1])]
                
                df_processed = pd.DataFrame(df_processed, columns=cols)
            
            # 4. Thực hiện dự báo (Sử dụng đúng biến X_input_for_model)
            prediction = self.model.predict(X_input_for_model)[0]
            
            # 5. Lấy dữ liệu giải thích từ Explainer
            # Lúc này df_processed chắc chắn là DataFrame, không còn lỗi .columns nữa
            explanation_data = self.explainer.get_local_explanation(
                X_single=df_processed,
                original_input=original_input,
                top_k=top_k
            )
            
            return {
                "predicted_price": float(prediction),
                "confidence": 0.85,
                "model_name": self.model_name,
                "base_value": explanation_data["base_value"],
                "explanations": explanation_data["explanations"]
            }
            
        except Exception as e:
            logger.error(f"Prediction with explanation failed: {e}")
            raise


class ModelRegistry:
    """Simple registry for managing models."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.scalers_dir = SCALERS_DIR
    
    def list_available_models(self) -> list:
        """List all available trained models."""
        if not self.models_dir.exists():
            return []
        return [f.stem for f in self.models_dir.glob("*.pkl") 
                if f.is_file() and f.stem != "preprocessor"]
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to a model."""
        return self.models_dir / f"{model_name}.pkl"
    
    def get_scaler_path(self) -> Path:
        """Get path to preprocessing scaler."""
        return self.scalers_dir / "preprocessor.joblib"
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists."""
        return self.get_model_path(model_name).exists()
    
    def scaler_exists(self) -> bool:
        """Check if scaler exists."""
        return self.get_scaler_path().exists()


def create_default_service() -> PredictionService:
    """
    Create a prediction service with default models and explainer.
    
    Returns:
        Configured PredictionService
    """
    registry = ModelRegistry()
    
    # Try to load the latest model
    models = registry.list_available_models()
    if not models:
        logger.warning("No models found in model registry")
        return PredictionService()
    
    # Use the first available model (sorted alphabetically)
    model_name = sorted(models)[0]
    model_path = registry.get_model_path(model_name)
    scaler_path = registry.get_scaler_path() if registry.scaler_exists() else None
    
    service = PredictionService(model_path=model_path, scaler_path=scaler_path)
    
    # Try to load explainer (non-blocking; continue even if it fails)
    if (EXPLAINER_DIR / "shap_explainer.joblib").exists():
        if not service.load_explainer():
            logger.warning("Could not load SHAP explainer, XAI features will be unavailable")
    
    return service