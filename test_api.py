"""
API Test Script
Demonstrates how to use the prediction API endpoints.
"""

import requests
import json
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"


def print_response(endpoint: str, response: requests.Response) -> None:
    """Pretty print API response"""
    print(f"\n{'='*80}")
    print(f"Endpoint: {endpoint}")
    print(f"Status: {response.status_code}")
    print(f"Response:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print(f"{'='*80}\n")


def test_health_check() -> bool:
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print_response("GET /health", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_list_models() -> bool:
    """Test list models endpoint"""
    print("\n📋 Testing List Models...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        print_response("GET /models", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ List models failed: {e}")
        return False


def test_single_prediction() -> bool:
    """Test single prediction"""
    print("\n🏠 Testing Single Prediction...")
    
    house_data = {
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
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=house_data
        )
        print_response("POST /predict", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False


def test_batch_prediction() -> bool:
    """Test batch prediction"""
    print("\n🏘️  Testing Batch Prediction...")
    
    house_data = {
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
    
    batch_data = {"houses": [house_data, house_data]}  # Predict for 2 houses
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-batch",
            json=batch_data
        )
        print_response("POST /predict-batch", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False


def run_all_tests() -> None:
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 RUNNING API TESTS")
    print("="*80)
    
    tests = [
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    import sys
    
    print("\n⚠️  Make sure the API server is running:")
    print("   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("\nStarting tests in 2 seconds...")
    
    import time
    time.sleep(2)
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tests failed: {e}")
        sys.exit(1)
