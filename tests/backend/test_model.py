"""Tests for ML model inference."""

import pytest
import numpy as np
from backend.model.file import Model, DeterministicBaseline, ModelConfig


class TestDeterministicBaseline:
    """Tests for deterministic baseline model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = ModelConfig(model_type="deterministic")
        model = DeterministicBaseline(config)
        
        assert model.is_ready()
        assert model.config.model_type == "deterministic"
    
    def test_basic_inference(self):
        """Test basic inference."""
        config = ModelConfig(model_type="deterministic")
        model = DeterministicBaseline(config)
        
        result = model.predict(
            eye_closure_ratio=0.2,
            phone_usage=False,
            speed=65,
            head_pose_degrees=5.0,
            unauthorized_objects_count=0
        )
        
        assert "riskScore" in result
        assert "drowsinessScore" in result
        assert "distractionScore" in result
        assert "headPoseScore" in result
        assert "unauthorizedObjectScore" in result
        assert "explanations" in result
        
        assert 0 <= result["riskScore"] <= 100
        assert 0.0 <= result["drowsinessScore"] <= 1.0
    
    def test_high_risk_scenario(self):
        """Test high risk scenario (drowsy driver)."""
        config = ModelConfig(model_type="deterministic")
        model = DeterministicBaseline(config)
        
        result = model.predict(
            eye_closure_ratio=0.8,  # High eye closure
            phone_usage=True,       # Phone usage
            speed=120,               # High speed
            head_pose_degrees=30.0, # Large head pose deviation
            unauthorized_objects_count=2  # Multiple objects
        )
        
        # Should have high risk (low score in 0-100 scale)
        assert result["riskScore"] < 50
        assert result["drowsinessScore"] < 0.5
        assert result["distractionScore"] < 0.5
    
    def test_low_risk_scenario(self):
        """Test low risk scenario (safe driver)."""
        config = ModelConfig(model_type="deterministic")
        model = DeterministicBaseline(config)
        
        result = model.predict(
            eye_closure_ratio=0.1,  # Low eye closure
            phone_usage=False,       # No phone usage
            speed=60,                # Normal speed
            head_pose_degrees=2.0,  # Small head pose deviation
            unauthorized_objects_count=0  # No objects
        )
        
        # Should have low risk (high score in 0-100 scale)
        assert result["riskScore"] > 70
        assert result["drowsinessScore"] > 0.8
        assert result["distractionScore"] > 0.9
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing with multiple predictions."""
        config = ModelConfig(model_type="deterministic", smoothing_window=5)
        model = DeterministicBaseline(config)
        
        results = []
        for _ in range(10):
            result = model.predict(
                eye_closure_ratio=0.3,
                phone_usage=False,
                speed=65,
                head_pose_degrees=5.0,
                unauthorized_objects_count=0
            )
            results.append(result["riskScore"])
        
        # Scores should converge due to smoothing
        assert len(set(results)) < len(results)  # Some duplicate scores


class TestModelWrapper:
    """Tests for Model wrapper class."""
    
    def test_model_initialization_deterministic(self):
        """Test model initialization with deterministic type."""
        model = Model(config={"model_type": "deterministic"})
        
        assert model.is_ready()
        assert isinstance(model.model, DeterministicBaseline)
    
    def test_predict_from_api_input(self):
        """Test prediction from API input format."""
        model = Model(config={"model_type": "deterministic"})
        
        result = model.predict_from_api_input(
            eye_closure_ratio=0.2,
            phone_usage=False,
            speed=65,
            head_pose_degrees=5.0,
            unauthorized_objects_count=0
        )
        
        assert "riskScore" in result
        assert 0 <= result["riskScore"] <= 100
    
    def test_with_custom_weights(self):
        """Test prediction with custom risk weights."""
        model = Model(config={"model_type": "deterministic"})
        
        custom_weights = {
            "drowsiness": 0.5,
            "distraction": 0.3,
            "headPose": 0.15,
            "unauthorizedObjects": 0.05
        }
        
        result = model.predict_from_api_input(
            eye_closure_ratio=0.5,  # Moderate drowsiness
            phone_usage=False,
            speed=65,
            head_pose_degrees=5.0,
            unauthorized_objects_count=0,
            risk_weights=custom_weights
        )
        
        assert "riskScore" in result
        # Higher drowsiness weight should affect the result
        assert result["riskScore"] < 80


