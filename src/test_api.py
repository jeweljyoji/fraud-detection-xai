from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

sample_transaction = {
    "Time": 146026.0, "Amount": 2.22,
    "V1": 1.894, "V2": 1.906, "V3": -3.516, "V4": 4.509,
    "V5": 2.044, "V6": -1.420, "V7": 1.057, "V8": -0.5,
    "V9": -1.2, "V10": -0.9, "V11": 1.1, "V12": -0.8,
    "V13": 0.3, "V14": -2.1, "V15": 0.4, "V16": -0.6,
    "V17": -0.3, "V18": -0.2, "V19": 0.1, "V20": 0.2,
    "V21": 0.3, "V22": -0.1, "V23": -0.05, "V24": -0.1,
    "V25": 0.1, "V26": -0.1, "V27": 0.05, "V28": 0.02
}

def test_root_endpoint_returns_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Fraud detection API is running"

def test_predict_returns_valid_response_shape():
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert "threshold_used" in body

def test_predict_probability_is_between_zero_and_one():
    response = client.post("/predict", json=sample_transaction)
    probability = response.json()["fraud_probability"]
    assert 0 <= probability <= 1

def test_predict_rejects_missing_fields():
    incomplete_transaction = {"Time": 0.0, "Amount": 10.0}
    response = client.post("/predict", json=incomplete_transaction)
    assert response.status_code == 422