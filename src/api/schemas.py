"""
API schemas and data models for request/response validation.
Uses Pydantic for type validation and serialization.
"""

from typing import Optional, List, Union, Any
from pydantic import BaseModel, Field


class HousePriceInput(BaseModel):
    """Schema for single house features."""

    MSSubClass: Optional[int] = Field(None, description="Building class")
    MSZoning: Optional[str] = Field(None, description="Zoning classification")
    LotFrontage: Optional[float] = Field(None, description="Linear feet of street connected to property")
    LotArea: Optional[int] = Field(None, description="Lot size in square feet")
    Street: Optional[str] = Field(None, description="Type of road access")
    Alley: Optional[str] = Field(None, description="Type of alley access")
    LotShape: Optional[str] = Field(None, description="General shape of property")
    LandContour: Optional[str] = Field(None, description="Flatness of the property")
    Utilities: Optional[str] = Field(None, description="Type of utilities available")
    LotConfig: Optional[str] = Field(None, description="Lot configuration")
    LandSlope: Optional[str] = Field(None, description="Slope of property")
    Neighborhood: Optional[str] = Field(None, description="Physical locations within Ames city limits")
    Condition1: Optional[str] = Field(None, description="Proximity to various conditions")
    Condition2: Optional[str] = Field(None, description="Proximity to various conditions (if more than one)")
    BldgType: Optional[str] = Field(None, description="Type of dwelling")
    HouseStyle: Optional[str] = Field(None, description="Style of dwelling")
    OverallQual: Optional[int] = Field(None, description="Overall material and finish quality")
    OverallCond: Optional[int] = Field(None, description="Overall condition rating")
    YearBuilt: Optional[int] = Field(None, description="Original construction date")
    YearRemodAdd: Optional[int] = Field(None, description="Remodel date")
    RoofStyle: Optional[str] = Field(None, description="Type of roof")
    RoofMatl: Optional[str] = Field(None, description="Roof material")
    Exterior1st: Optional[str] = Field(None, description="Exterior covering on house")
    Exterior2nd: Optional[str] = Field(None, description="Exterior covering on house (if more than one material)")
    MasVnrType: Optional[str] = Field(None, description="Masonry veneer type")
    MasVnrArea: Optional[float] = Field(None, description="Masonry veneer area in square feet")
    ExterQual: Optional[str] = Field(None, description="Exterior material quality")
    ExterCond: Optional[str] = Field(None, description="Present condition of the material on the exterior")
    Foundation: Optional[str] = Field(None, description="Type of foundation")
    BsmtQual: Optional[str] = Field(None, description="Height of the basement")
    BsmtCond: Optional[str] = Field(None, description="General condition of the basement")
    BsmtExposure: Optional[str] = Field(None, description="Refers to walkout or garden level walls")
    BsmtFinType1: Optional[str] = Field(None, description="Rating of basement finished area")
    BsmtFinSF1: Optional[float] = Field(None, description="Type 1 finished square feet")
    BsmtFinType2: Optional[str] = Field(None, description="Rating of basement finished area (if multiple types)")
    BsmtFinSF2: Optional[float] = Field(None, description="Type 2 finished square feet")
    BsmtUnfSF: Optional[float] = Field(None, description="Unfinished square feet of basement area")
    TotalBsmtSF: Optional[float] = Field(None, description="Total square feet of basement area")
    Heating: Optional[str] = Field(None, description="Type of heating")
    HeatingQC: Optional[str] = Field(None, description="Heating quality and condition")
    CentralAir: Optional[str] = Field(None, description="Central air conditioning")
    Electrical: Optional[str] = Field(None, description="Electrical system")
    FirstFlrSF: Optional[int] = Field(None, description="First Floor square feet")
    SecondFlrSF: Optional[int] = Field(None, description="Second floor square feet")
    LowQualFinSF: Optional[int] = Field(None, description="Low quality finished square feet (all floors)")
    GrLivArea: Optional[int] = Field(None, description="Above grade (ground) living area square feet")
    BsmtFullBath: Optional[int] = Field(None, description="Basement full bathrooms")
    BsmtHalfBath: Optional[int] = Field(None, description="Basement half bathrooms")
    FullBath: Optional[int] = Field(None, description="Full bathrooms above grade")
    HalfBath: Optional[int] = Field(None, description="Half baths above grade")
    BedroomAbvGr: Optional[int] = Field(None, description="Bedrooms above grade (does NOT include basement bedrooms)")
    KitchenAbvGr: Optional[int] = Field(None, description="Kitchens above grade")
    KitchenQual: Optional[str] = Field(None, description="Kitchen quality")
    TotRmsAbvGrd: Optional[int] = Field(None, description="Total rooms above grade (does not include bathrooms)")
    Functional: Optional[str] = Field(None, description="Home functionality rating")
    Fireplaces: Optional[int] = Field(None, description="Number of fireplaces")
    FireplaceQu: Optional[str] = Field(None, description="Fireplace quality")
    GarageType: Optional[str] = Field(None, description="Garage location")
    GarageYrBlt: Optional[float] = Field(None, description="Year garage was built")
    GarageFinish: Optional[str] = Field(None, description="Interior finish of the garage")
    GarageCars: Optional[int] = Field(None, description="Size of garage in car capacity")
    GarageArea: Optional[float] = Field(None, description="Size of garage in square feet")
    GarageQual: Optional[str] = Field(None, description="Garage quality")
    GarageCond: Optional[str] = Field(None, description="Garage condition")
    PavedDrive: Optional[str] = Field(None, description="Paved driveway")
    WoodDeckSF: Optional[int] = Field(None, description="Wood deck area in square feet")
    OpenPorchSF: Optional[int] = Field(None, description="Open porch area in square feet")
    EnclosedPorch: Optional[int] = Field(None, description="Enclosed porch area in square feet")
    ThreeSsnPorch: Optional[int] = Field(None, description="Three season porch area in square feet")
    ScreenPorch: Optional[int] = Field(None, description="Screen porch area in square feet")
    PoolArea: Optional[int] = Field(None, description="Pool area in square feet")
    PoolQC: Optional[str] = Field(None, description="Pool quality")
    Fence: Optional[str] = Field(None, description="Fence quality")
    MiscFeature: Optional[str] = Field(None, description="Miscellaneous feature not covered in other categories")
    MiscVal: Optional[int] = Field(None, description="Value of miscellaneous feature")
    MoSold: Optional[int] = Field(None, description="Month sold")
    YrSold: Optional[int] = Field(None, description="Year sold")
    SaleType: Optional[str] = Field(None, description="Type of sale")
    SaleCondition: Optional[str] = Field(None, description="Condition of sale")
    
    class Config:
        json_schema_extra = {
            "example": {
                "MSSubClass": 60,
                "MSZoning": "RL",
                "LotArea": 8450,
                "Street": "Pave",
                "LotShape": "Reg",
                "LotFrontage": 50,
                "LandContour": "Lvl",
                "Utilities": "AllPub",
                "LotConfig": "Inside",
                "LandSlope": "Gtl",
                "Neighborhood": "CollgCr",
                "Condition1": "Norm",
                "Condition2": "Norm",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "RoofStyle": "Gable",
                "RoofMatl": "CompShg",
                "Exterior1st": "VinylSd",
                "Exterior2nd": "VinylSd",
                "MasVnrType": "BrkFace",
                "MasVnrArea": 196.0,
                "ExterQual": "Gd",
                "ExterCond": "TA",
                "Foundation": "PConc",
                "BsmtQual": "Gd",
                "BsmtCond": "TA",
                "BsmtExposure": "No",
                "BsmtFinType1": "GLQ",
                "BsmtFinSF1": 706.0,
                "BsmtFinType2": "Unf",
                "BsmtFinSF2": 0.0,
                "BsmtUnfSF": 150.0,
                "TotalBsmtSF": 856.0,
                "Heating": "GasA",
                "HeatingQC": "Ex",
                "CentralAir": "Y",
                "Electrical": "SBrkr",
                "FirstFlrSF": 856,
                "SecondFlrSF": 854,
                "LowQualFinSF": 0,
                "GrLivArea": 1710,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "FullBath": 2,
                "HalfBath": 1,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "KitchenQual": "Gd",
                "TotRmsAbvGrd": 8,
                "Functional": "Typ",
                "Fireplaces": 0,
                "FireplaceQu": None,
                "GarageType": "Attchd",
                "GarageYrBlt": 2003.0,
                "GarageFinish": "RFn",
                "GarageCars": 2,
                "GarageArea": 548.0,
                "GarageQual": "TA",
                "GarageCond": "TA",
                "PavedDrive": "Y",
                "WoodDeckSF": 0,
                "OpenPorchSF": 61,
                "EnclosedPorch": 0,
                "ThreeSsnPorch": 0,
                "ScreenPorch": 0,
                "PoolArea": 0,
                "PoolQC": None,
                "Fence": None,
                "MiscFeature": None,
                "MiscVal": 0,
                "MoSold": 2,
                "YrSold": 2008,
                "SaleType": "WD",
                "SaleCondition": "Normal",
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


class FeatureExplanation(BaseModel):
    """Individual feature explanation from SHAP analysis"""
    
    feature_name: str = Field(..., description="Human-readable feature name")
    original_value: Optional[Union[float, str, int]] = Field(default=None, description="Original input value for this feature (numeric or categorical)")
    shap_value: float = Field(..., description="SHAP value indicating contribution to prediction")
    contribution_type: str = Field(
        ..., 
        description="Whether this feature increases ('positive') or decreases ('negative') the predicted price"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_name": "OverallQual",
                "original_value": 7.0,
                "shap_value": 45000.5,
                "contribution_type": "positive"
            }
        }


class PredictionWithExplainResponse(BaseModel):
    """Response schema for prediction with XAI explanations"""
    
    predicted_price: float = Field(..., description="Predicted sale price")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_name: str = Field(..., description="Name of the model used")
    base_value: float = Field(..., description="SHAP base value (expected value / mean prediction)")
    explanations: List[FeatureExplanation] = Field(
        ..., 
        description="Top contributing features with SHAP-based explanations (sorted by impact)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 285000.50,
                "confidence": 0.92,
                "model_name": "CatBoost",
                "base_value": 250000.0,
                "explanations": [
                    {
                        "feature_name": "OverallQual",
                        "original_value": 8.0,
                        "shap_value": 25000.5,
                        "contribution_type": "positive"
                    },
                    {
                        "feature_name": "GrLivArea",
                        "original_value": 2500.0,
                        "shap_value": 15000.2,
                        "contribution_type": "positive"
                    },
                    {
                        "feature_name": "Neighborhood",
                        "original_value": None,
                        "shap_value": -5000.3,
                        "contribution_type": "negative"
                    }
                ]
            }
        }

