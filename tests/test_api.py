import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction_endpoint():
    client = TestClient(app)
    health = client.get('/health')
    assert health.status_code == 200
    response = client.post('/predict', json={'features': {'wind_speed_knts': 25, 'wind_gust_knts': 35, 'visibility_m': 3000, 'n_approaches': 1}})
    assert response.status_code == 200
    data = response.json()
    assert {'prediction', 'probability', 'label', 'threshold', 'model'}.issubset(data)
