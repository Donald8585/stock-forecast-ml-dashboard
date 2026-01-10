import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    print("✅ Root endpoint working")

def test_models():
    response = requests.get(f"{BASE_URL}/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data['available_models']) == 2
    print("✅ Models endpoint working")

def test_forecast_es():
    payload = {"periods": 7, "model": "exponential_smoothing"}
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data['predictions']) == 7
    print("✅ Exponential Smoothing forecast working")

def test_forecast_arima():
    payload = {"periods": 14, "model": "arima"}
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data['predictions']) == 14
    print("✅ ARIMA forecast working")

def test_metrics():
    response = requests.get(f"{BASE_URL}/metrics/exponential_smoothing")
    assert response.status_code == 200
    data = response.json()
    assert 'MAPE' in data
    assert 'RMSE' in data
    print("✅ Metrics endpoint working")

if __name__ == "__main__":
    print("🧪 Running API tests...\n")
    try:
        test_root()
        test_models()
        test_forecast_es()
        test_forecast_arima()
        test_metrics()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
