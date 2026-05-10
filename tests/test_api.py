"""Tests for the FastAPI endpoints."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Go-Around" in resp.text


def test_predict_minimal_input(client):
    payload = {"airport": "EDDF", "wtc": "M", "wind_speed_knts": 12.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert "probability_go_around" in data
    assert data["predicted_class"] in {0, 1}
    assert 0.0 <= data["probability_go_around"] <= 1.0


def test_predict_full_input(client):
    payload = {
        "airport": "EGLL", "runway": "27L", "wtc": "H",
        "typecode": "B77W", "icaoaircrafttype": "L2J",
        "airport_country": "GB", "airport_region": "EU",
        "n_approaches": 1.0, "glide_slope_angle": 3.0, "rwy_length": 3658.0,
        "month": 11.0, "day_of_week": 4.0, "hour_utc": 18.0,
        "wind_speed_knts": 20.0, "wind_gust_knts": 28.0,
        "visibility_m": 4000.0, "temperature_deg": 8.0,
        "press_sea_level_p": 1008.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert abs(data["probability_go_around"] + data["probability_normal_landing"] - 1.0) < 0.01


def test_predict_empty_input(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
