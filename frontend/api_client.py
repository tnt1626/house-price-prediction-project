"""
API Client Module

Xử lý tất cả các gọi API đến Backend với error handling toàn diện.
"""

import requests
import logging
from typing import Optional, Dict, Any, List
import json

from config import BACKEND_URL, API_TIMEOUT, HEALTH_CHECK_TIMEOUT

# Cấu hình logging
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception cho API errors"""
    pass


class HealthCheckError(APIError):
    """Exception cho health check failures"""
    pass


class APIClient:
    """Client để gọi API Backend"""
    
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra trạng thái Server.
        
        Returns:
            Dict chứa status, model_loaded, model_name, version
            
        Raises:
            HealthCheckError: Nếu server không phản hồi
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=HEALTH_CHECK_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error("Health check timeout")
            raise HealthCheckError("Server timeout - kiểm tra kết nối mạng")
        except requests.ConnectionError:
            logger.error("Failed to connect to backend")
            raise HealthCheckError("Không thể kết nối đến Server")
        except requests.HTTPError as e:
            logger.error(f"Health check HTTP error: {e}")
            raise HealthCheckError(f"Server lỗi: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HealthCheckError(f"Health check thất bại: {str(e)}")
    
    def get_models(self) -> List[str]:
        """
        Lấy danh sách các mô hình có sẵn.
        
        Returns:
            List các tên mô hình
            
        Raises:
            APIError: Nếu không lấy được danh sách
        """
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Xử lý response format - Backend trả về các format khác nhau
            if isinstance(data, dict):
                if "available_models" in data:
                    return data["available_models"]
                elif "models" in data:
                    return data["models"]
            
            if isinstance(data, list):
                return data
            
            logger.warning(f"Unexpected models response format: {data}")
            return []
        except requests.Timeout:
            logger.error("Get models timeout")
            raise APIError("Timeout khi lấy danh sách mô hình")
        except requests.ConnectionError:
            logger.error("Failed to connect to backend for models")
            raise APIError("Không thể kết nối đến Server")
        except Exception as e:
            logger.error(f"Get models failed: {e}")
            raise APIError(f"Lỗi lấy danh sách mô hình: {str(e)}")
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Tải một mô hình cụ thể.
        
        Args:
            model_name: Tên mô hình cần tải
            
        Returns:
            Dict chứa status của load operation
            
        Raises:
            APIError: Nếu tải mô hình thất bại
        """
        try:
            url = f"{self.base_url}/models/load/{model_name}"
            response = requests.post(url, timeout=API_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error(f"Load model timeout for {model_name}")
            raise APIError(f"Timeout khi tải mô hình {model_name}")
        except requests.HTTPError as e:
            logger.error(f"Load model HTTP error: {e.response.text}")
            raise APIError(f"Tải mô hình thất bại: {e.response.text}")
        except Exception as e:
            logger.error(f"Load model failed: {e}")
            raise APIError(f"Lỗi tải mô hình: {str(e)}")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán giá cho một căn nhà.
        
        Args:
            data: Dict chứa các features của nhà
            
        Returns:
            Dict chứa predicted_price, confidence, model_name
            
        Raises:
            APIError: Nếu dự đoán thất bại
        """
        try:
            url = f"{self.base_url}/predict"
            response = requests.post(
                url,
                json=data,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error("Prediction timeout")
            raise APIError("Timeout dự đoán - vui lòng thử lại")
        except requests.HTTPError as e:
            error_detail = e.response.text
            try:
                json_response = e.response.json()
                if isinstance(json_response, dict):
                    error_detail = json_response.get("detail", error_detail)
                    if isinstance(error_detail, list):
                        errors = []
                        for err in error_detail:
                            loc = err.get("loc", [])
                            msg = err.get("msg", "Unknown error")
                            errors.append(f"{'.'.join(str(l) for l in loc)}: {msg}")
                        error_detail = " | ".join(errors)
            except:
                pass
            logger.error(f"Prediction HTTP error: {error_detail}")
            raise APIError(f"Lỗi dự đoán: {error_detail}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise APIError(f"Lỗi dự đoán: {str(e)}")
    
    def predict_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán giá cho nhiều căn nhà (batch).
        
        Args:
            batch_data: Dict chứa "houses" list
            
        Returns:
            Dict chứa predictions list và total_processed
            
        Raises:
            APIError: Nếu batch dự đoán thất bại
        """
        try:
            url = f"{self.base_url}/predict-batch"
            response = requests.post(
                url,
                json=batch_data,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error("Batch prediction timeout")
            raise APIError("Timeout batch dự đoán - vui lòng thử lại")
        except requests.HTTPError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except:
                pass
            logger.error(f"Batch prediction HTTP error: {error_detail}")
            raise APIError(f"Lỗi batch dự đoán: {error_detail}")
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise APIError(f"Lỗi batch dự đoán: {str(e)}")
    
    def predict_with_explanation(self, data: Dict[str, Any], top_features: int = 10) -> Dict[str, Any]:
        """
        Dự đoán giá với giải thích (SHAP).
        
        Args:
            data: Dict chứa các features
            top_features: Số lượng features top được trả về
            
        Returns:
            Dict chứa predicted_price, confidence, model_name, base_value, explanations
            
        Raises:
            APIError: Nếu dự đoán/giải thích thất bại
        """
        try:
            url = f"{self.base_url}/predict-explain"
            params = {"top_features": top_features}
            response = requests.post(
                url,
                json=data,
                params=params,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error("Prediction with explanation timeout")
            raise APIError("Timeout dự đoán với giải thích - vui lòng thử lại")
        except requests.HTTPError as e:
            error_detail = e.response.text
            try:
                json_response = e.response.json()
                # Lấy detail từ response
                if isinstance(json_response, dict):
                    error_detail = json_response.get("detail", error_detail)
                    # Nếu là validation error từ Pydantic, show chi tiết
                    if isinstance(error_detail, list):
                        errors = []
                        for err in error_detail:
                            loc = err.get("loc", [])
                            msg = err.get("msg", "Unknown error")
                            errors.append(f"{'.'.join(str(l) for l in loc)}: {msg}")
                        error_detail = " | ".join(errors)
            except:
                pass
            logger.error(f"Prediction with explanation HTTP error: {error_detail}")
            raise APIError(f"Lỗi dự đoán với giải thích: {error_detail}")
        except Exception as e:
            logger.error(f"Prediction with explanation failed: {e}")
            raise APIError(f"Lỗi dự đoán với giải thích: {str(e)}")


# Khởi tạo singleton client
_api_client = None


def get_api_client() -> APIClient:
    """Lấy singleton API client"""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
