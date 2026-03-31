"""
API schemas and data models for request/response validation.
Uses Pydantic for type validation and serialization.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class HousePriceInput(BaseModel):
    """Input schema for house price prediction"""
    
    LotArea: float = Field(..., description="Lot area in square feet")
    OverallQual: int = Field(..., description="Overall quality (1-10)")
    OverallCond: int = Field(..., description="Overall condition (1-10)")
    YearBuilt: int = Field(..., description="Year built")
    YearRemodAdd: int = Field(..., description="Year remodeled")
    TotalBsmtSF: float = Field(default=0, description="Total basement square feet")
    FirstFlrSF: float = Field(..., description="First floor square feet (1stFlrSF)")
    SecondFlrSF: float = Field(default=0, description="Second floor square feet (2ndFlrSF)")
    GrLivArea: float = Field(..., description="Above grade living area")
    FullBath: int = Field(default=0, description="Full bathrooms")
    HalfBath: int = Field(default=0, description="Half bathrooms")
    BsmtFullBath: int = Field(default=0, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(default=0, description="Basement half bathrooms")
    Bedroom: int = Field(default=3, description="Bedrooms above grade (BedroomAbvGr)")
    Kitchen: int = Field(default=1, description="Kitchens above grade (KitchenAbvGr)")
    TotRmsAbvGrd: int = Field(default=6, description="Total rooms above grade")
    Fireplaces: int = Field(default=0, description="Number of fireplaces")
    GarageCars: int = Field(default=1, description="Garage car capacity")
    GarageSF: float = Field(default=0, description="Garage square feet")
    
    # Categorical features
    MSZoning: str = Field(default="RL", description="Zoning classification")
    Neighborhood: str = Field(default="NAmes", description="Neighborhood")
    BldgType: str = Field(default="1Fam", description="Building type")
    ExterQual: str = Field(default="TA", description="Exterior quality")
    ExterCond: str = Field(default="TA", description="Exterior condition")
    BsmtQual: Optional[str] = Field(default="TA", description="Basement quality")
    BsmtCond: Optional[str] = Field(default="TA", description="Basement condition")
    HeatingQC: str = Field(default="TA", description="Heating quality")
    KitchenQual: str = Field(default="TA", description="Kitchen quality")
    FireplaceQu: Optional[str] = Field(default=None, description="Fireplace quality")
    GarageQual: Optional[str] = Field(default="TA", description="Garage quality")
    GarageCond: Optional[str] = Field(default="TA", description="Garage condition")
    
    # Sale information
    MoSold: int = Field(default=6, description="Month sold")
    YrSold: int = Field(default=2022, description="Year sold")
    SaleType: str = Field(default="WD", description="Sale type")
    SaleCondition: str = Field(default="Normal", description="Sale condition")
    
    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 8450,
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "TotalBsmtSF": 1000,
                "FirstFlrSF": 856,
                "SecondFlrSF": 854,
                "GrLivArea": 1710,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "Bedroom": 3,
                "Kitchen": 1,
                "TotRmsAbvGrd": 8,
                "Fireplaces": 1,
                "GarageCars": 2,
                "GarageSF": 548,
                "MSZoning": "RL",
                "Neighborhood": "NAmes",
                "BldgType": "1Fam",
                "ExterQual": "Gd",
                "ExterCond": "TA",
                "BsmtQual": "Gd",
                "BsmtCond": "TA",
                "HeatingQC": "Ex",
                "KitchenQual": "Gd",
                "FireplaceQu": "Ta",
                "GarageQual": "TA",
                "GarageCond": "TA",
                "MoSold": 7,
                "YrSold": 2022,
                "SaleType": "WD",
                "SaleCondition": "Normal"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for price prediction"""
    
    predicted_price: float = Field(..., description="Predicted sale price")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 180000.50,
                "confidence": 0.85,
                "model_name": "XGBoost"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    
    houses: List[HousePriceInput] = Field(..., description="List of houses to predict")
    
    class Config:
        json_schema_extra = {
            "example": {
                "houses": [
                    {
                        "LotArea": 8450,
                        "OverallQual": 7,
                        "OverallCond": 5,
                        "YearBuilt": 2003,
                        "YearRemodAdd": 2003,
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total houses processed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "predicted_price": 180000.50,
                        "confidence": 0.85,
                        "model_name": "XGBoost"
                    }
                ],
                "total_processed": 1
            }
        }


class HealthResponse(BaseModel):
    """Response for health check endpoint"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of loaded model")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_name": "XGBoost",
                "version": "1.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error message")
    detail: str = Field(default="", description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid input",
                "detail": "LotArea must be positive"
            }
        }
