import pytest
import json
import os
import sys

# Add the backend src directory to the Python path for testing.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the Flask app instance.
from app import app

@pytest.fixture
def client():
    """Configures the app for testing, disabling error catching."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the basic health check endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "Backend is running!"}

def test_predict_success(client):
    """
    Test the /predict endpoint with valid data.
    """
    # Sample data that your dummy model can handle.
    test_features = {
        "GrLivArea": 1500,
        "GarageCars": 2,
        "OverallQual": 7,
        "TotalBsmtSF": 1000,
        "YearBuilt": 2005
    }
    response = client.post('/predict', json=test_features)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))

def test_predict_invalid_json(client):
    """Test the /predict endpoint with invalid content type."""
    response = client.post('/predict', data="not a json")
    assert response.status_code == 400
    assert json.loads(response.data) == {"error": "Request must be JSON"}

def test_predict_missing_feature(client):
    """
    Test /predict when a required feature is missing.
    """
    test_features = {
        "GrLivArea": 1500,
        "GarageCars": 2,
        "OverallQual": 7,
        "TotalBsmtSF": 1000,
    }
    response = client.post('/predict', json=test_features)
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data
    assert "Failed to make prediction" in data["error"] or "KeyError" in data["error"]
