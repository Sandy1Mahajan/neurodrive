"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.main import create_app
import os
import tempfile
import yaml


@pytest.fixture
def test_config_file():
    """Create temporary config file for testing."""
    config = {
        "risk_weights": {
            "drowsiness": 0.25,
            "distraction": 0.30,
            "headPose": 0.25,
            "unauthorizedObjects": 0.20
        },
        "thresholds": {
            "eyeClosureTimeSeconds": 2.0,
            "distractionDetectedThreshold": 0.5,
            "headPoseDegreesThreshold": 15.0,
            "unauthorizedObjectsThreshold": 1,
            "speedLimitKmh": 80
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def client(test_config_file):
    """Create test client."""
    os.environ["NEURODRIVE_CONFIG"] = test_config_file
    os.environ["DEV_MODE"] = "true"
    
    app = create_app()
    client = TestClient(app)
    
    yield client
    
    # Cleanup
    del os.environ["NEURODRIVE_CONFIG"]
    if "DEV_MODE" in os.environ:
        del os.environ["DEV_MODE"]


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns OK."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "model_ready" in data
        assert "uptime_seconds" in data


class TestConfigEndpoint:
    """Tests for /config endpoint."""
    
    def test_get_config(self, client):
        """Test getting configuration."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_weights" in data
        assert "thresholds" in data
        assert "drowsiness" in data["risk_weights"]
        assert "eyeClosureTimeSeconds" in data["thresholds"]
    
    def test_update_config(self, client):
        """Test updating configuration."""
        update_data = {
            "risk_weights": {
                "drowsiness": 0.3,
                "distraction": 0.25,
                "headPose": 0.25,
                "unauthorizedObjects": 0.20
            }
        }
        
        response = client.put("/config", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["config"]["risk_weights"]["drowsiness"] == 0.3


class TestInferenceEndpoint:
    """Tests for /api/v1/infer endpoint."""
    
    def test_inference_basic(self, client):
        """Test basic inference request."""
        payload = {
            "eyeClosureRatio": 0.2,
            "phoneUsage": False,
            "speed": 65,
            "headPoseDegrees": 5.0,
            "unauthorizedObjectsCount": 0
        }
        
        response = client.post("/api/v1/infer", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "riskScore" in data
        assert "statusText" in data
        assert "metrics" in data
        assert "alerts" in data
        assert "weights" in data
        assert "thresholds" in data
        
        assert 0 <= data["riskScore"] <= 100
        assert "drowsiness" in data["metrics"]
        assert "distraction" in data["metrics"]
    
    def test_inference_high_risk(self, client):
        """Test inference with high risk scenario."""
        payload = {
            "eyeClosureRatio": 0.8,
            "phoneUsage": True,
            "speed": 120,
            "headPoseDegrees": 30.0,
            "unauthorizedObjectsCount": 2
        }
        
        response = client.post("/api/v1/infer", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have low risk score (high risk)
        assert data["riskScore"] < 50
        assert len(data["alerts"]) > 0
    
    def test_inference_validation(self, client):
        """Test input validation."""
        # Invalid eye closure ratio (>1.0)
        payload = {
            "eyeClosureRatio": 1.5,
            "phoneUsage": False,
            "speed": 65
        }
        
        response = client.post("/api/v1/infer", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_inference_optional_fields(self, client):
        """Test inference with optional fields omitted."""
        payload = {
            "eyeClosureRatio": 0.2,
            "phoneUsage": False,
            "speed": 65
        }
        
        response = client.post("/api/v1/infer", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "riskScore" in data


class TestMetricsEndpoint:
    """Tests for /metrics endpoint (Prometheus)."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        # May or may not be available depending on prometheus_client installation
        assert response.status_code in [200, 404]


class TestOpenAPIDocs:
    """Tests for OpenAPI documentation."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI docs."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc docs."""
        response = client.get("/redoc")
        
        assert response.status_code == 200


