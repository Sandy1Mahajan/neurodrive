"""
Production-ready ML model for NeuroDrive DMS inference.

Supports two modes:
1. Deterministic baseline (default): Rule-based calculations with temporal smoothing
2. ML model: PyTorch/ONNX model for advanced inference

The model accepts metrics from the FastAPI endpoint and returns risk scores.
"""

from __future__ import annotations

import os
import json
import logging
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("onnxruntime not available; ONNX inference disabled")


@dataclass
class ModelConfig:
    """Configuration for model inference."""
    model_type: str = "deterministic"  # "deterministic" or "onnx"
    model_path: Optional[str] = None
    smoothing_window: int = 10
    ear_threshold: float = 0.25


class DeterministicBaseline:
    """Lightweight deterministic baseline model using rule-based calculations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.risk_history = deque(maxlen=config.smoothing_window)
        self.initialized = True
        logger.info("Deterministic baseline model initialized")
    
    def is_ready(self) -> bool:
        """Check if model is ready."""
        return self.initialized
    
    def predict(self, 
                eye_closure_ratio: float,
                phone_usage: bool,
                speed: int,
                head_pose_degrees: Optional[float] = None,
                unauthorized_objects_count: Optional[int] = None,
                risk_weights: Optional[Dict[str, float]] = None,
                thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict risk score from input metrics.
        
        Returns:
            Dict with riskScore, drowsinessScore, distractionScore, headPoseScore,
            unauthorizedObjectScore, and explanations
        """
        if not self.is_ready():
            raise RuntimeError("Model not ready")
        
        # Default weights and thresholds
        weights = risk_weights or {
            "drowsiness": 0.25,
            "distraction": 0.30,
            "headPose": 0.25,
            "unauthorizedObjects": 0.20
        }
        thresh = thresholds or {
            "eyeClosureTimeSeconds": 2.0,
            "distractionDetectedThreshold": 0.5,
            "headPoseDegreesThreshold": 15.0,
            "unauthorizedObjectsThreshold": 1,
            "speedLimitKmh": 80
        }
        
        # Calculate individual scores (0-1 scale)
        eye_closure_time = eye_closure_ratio * 3.0
        
        # Drowsiness score (lower is worse)
        if eye_closure_time < thresh["eyeClosureTimeSeconds"]:
            drowsiness_score = 1.0
        else:
            excess = eye_closure_time - thresh["eyeClosureTimeSeconds"]
            drowsiness_score = max(0.0, 1.0 - (excess / 3.0))
        
        # Distraction score
        distraction_score = 0.0 if phone_usage else 1.0
        
        # Head pose score
        head_pose_val = abs(head_pose_degrees) if head_pose_degrees is not None else 0.0
        if head_pose_val <= thresh["headPoseDegreesThreshold"]:
            head_pose_score = 1.0
        else:
            excess = head_pose_val - thresh["headPoseDegreesThreshold"]
            head_pose_score = max(0.0, 1.0 - (excess / 45.0))
        
        # Unauthorized objects score
        objects_count = unauthorized_objects_count or 0
        if objects_count <= thresh["unauthorizedObjectsThreshold"]:
            unauthorized_object_score = 1.0
        else:
            unauthorized_object_score = max(0.0, 1.0 - (objects_count - thresh["unauthorizedObjectsThreshold"]) / 3.0)
        
        # Speed factor
        speed_limit = thresh.get("speedLimitKmh", 80)
        if speed <= speed_limit:
            speed_factor = 1.0
        else:
            excess_speed = speed - speed_limit
            speed_factor = max(0.5, 1.0 - (excess_speed / 50.0))
        
        # Weighted risk score
        total_weight = sum(weights.values())
        base_risk = (
            drowsiness_score * weights["drowsiness"] +
            distraction_score * weights["distraction"] +
            head_pose_score * weights["headPose"] +
            unauthorized_object_score * weights["unauthorizedObjects"]
        ) / total_weight
        
        # Apply speed factor
        risk_score = base_risk * speed_factor
        
        # Temporal smoothing
        self.risk_history.append(risk_score)
        smoothed_risk = float(np.mean(self.risk_history)) if len(self.risk_history) > 0 else risk_score
        
        # Convert to 0-100 scale for API compatibility
        risk_score_100 = (1.0 - smoothed_risk) * 100.0
        
        return {
            "riskScore": max(0, min(100, int(round(risk_score_100)))),
            "drowsinessScore": float(drowsiness_score),
            "distractionScore": float(distraction_score),
            "headPoseScore": float(head_pose_score),
            "unauthorizedObjectScore": float(unauthorized_object_score),
            "explanations": {
                "drowsiness": f"Eye closure time: {eye_closure_time:.2f}s (threshold: {thresh['eyeClosureTimeSeconds']}s)",
                "distraction": f"Phone usage detected: {phone_usage}",
                "headPose": f"Head pose deviation: {head_pose_val:.1f}° (threshold: {thresh['headPoseDegreesThreshold']}°)",
                "unauthorizedObjects": f"Unauthorized objects count: {objects_count} (threshold: {thresh['unauthorizedObjectsThreshold']})",
                "speed": f"Speed: {speed} km/h (limit: {speed_limit} km/h)"
            }
        }


class ONNXModel:
    """ONNX model for ML-based inference."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = None
        self.initialized = False
        
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNX inference")
        
        if config.model_path and os.path.exists(config.model_path):
            try:
                self.session = ort.InferenceSession(
                    config.model_path,
                    providers=['CPUExecutionProvider']
                )
                self.initialized = True
                logger.info(f"ONNX model loaded from {config.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                raise
        else:
            logger.warning(f"ONNX model path not found: {config.model_path}")
    
    def is_ready(self) -> bool:
        """Check if model is ready."""
        return self.initialized and self.session is not None
    
    def predict(self,
                eye_closure_ratio: float,
                phone_usage: bool,
                speed: int,
                head_pose_degrees: Optional[float] = None,
                unauthorized_objects_count: Optional[int] = None,
                risk_weights: Optional[Dict[str, float]] = None,
                thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run ONNX inference."""
        if not self.is_ready():
            raise RuntimeError("ONNX model not ready")
        
        # Prepare input features (normalized)
        head_pose_val = abs(head_pose_degrees) if head_pose_degrees is not None else 0.0
        objects_count = float(unauthorized_objects_count or 0)
        
        # Normalize inputs
        features = np.array([[
            eye_closure_ratio,  # 0-1
            float(phone_usage),  # 0 or 1
            speed / 200.0,  # normalized to 0-1 (assuming max 200 km/h)
            head_pose_val / 90.0,  # normalized to 0-1 (max 90 degrees)
            objects_count / 5.0  # normalized to 0-1 (max 5 objects)
        ]], dtype=np.float32)
        
        # Get input/output names from model
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {input_name: features})
        inference_time = time.time() - start_time
        
        # Extract predictions (assuming model outputs risk scores)
        # Model should output: [drowsiness_score, distraction_score, head_pose_score, object_score, overall_risk]
        if len(outputs[0].shape) > 1:
            predictions = outputs[0][0]
        else:
            predictions = outputs[0]
        
        # Map outputs (adjust based on your model architecture)
        if len(predictions) >= 5:
            drowsiness_score = float(predictions[0])
            distraction_score = float(predictions[1])
            head_pose_score = float(predictions[2])
            unauthorized_object_score = float(predictions[3])
            base_risk = float(predictions[4])
        else:
            # Fallback: use single output as risk score
            base_risk = float(predictions[0]) if len(predictions) > 0 else 0.5
            drowsiness_score = distraction_score = head_pose_score = unauthorized_object_score = base_risk
        
        # Apply speed factor
        speed_limit = (thresholds or {}).get("speedLimitKmh", 80)
        speed_factor = 1.0 if speed <= speed_limit else max(0.5, 1.0 - ((speed - speed_limit) / 50.0))
        risk_score = base_risk * speed_factor
        
        # Convert to 0-100 scale
        risk_score_100 = (1.0 - risk_score) * 100.0
        
        return {
            "riskScore": max(0, min(100, int(round(risk_score_100)))),
            "drowsinessScore": float(drowsiness_score),
            "distractionScore": float(distraction_score),
            "headPoseScore": float(head_pose_score),
            "unauthorizedObjectScore": float(unauthorized_object_score),
            "explanations": {
                "model": "ONNX ML model",
                "inference_time_ms": f"{inference_time * 1000:.2f}",
                "model_path": self.config.model_path
            }
        }


class Model:
    """Production-ready model wrapper supporting both deterministic and ML inference."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model with configuration."""
        self.config_dict = config or {}
        
        # Load model config
        model_type = self.config_dict.get("model_type", os.getenv("MODEL_TYPE", "deterministic"))
        model_path = self.config_dict.get("model_path") or os.getenv("MODEL_PATH")
        
        model_config = ModelConfig(
            model_type=model_type,
            model_path=model_path,
            smoothing_window=int(self.config_dict.get("smoothing_window", 10))
        )
        
        # Initialize appropriate model
        if model_config.model_type == "onnx":
            try:
                self.model = ONNXModel(model_config)
                logger.info("Using ONNX model for inference")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model, falling back to deterministic: {e}")
                self.model = DeterministicBaseline(model_config)
        else:
            self.model = DeterministicBaseline(model_config)
            logger.info("Using deterministic baseline model for inference")
        
        self.initialized = self.model.is_ready()
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.initialized and self.model.is_ready()
    
    def predict_from_api_input(self,
                              eye_closure_ratio: float,
                              phone_usage: bool,
                              speed: int,
                              head_pose_degrees: Optional[float] = None,
                              unauthorized_objects_count: Optional[int] = None,
                              risk_weights: Optional[Dict[str, float]] = None,
                              thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict from API input format.
        
        This is the main method called by the FastAPI endpoint.
        """
        if not self.is_ready():
            raise RuntimeError("Model not ready for inference")
        
        return self.model.predict(
            eye_closure_ratio=eye_closure_ratio,
            phone_usage=phone_usage,
            speed=speed,
            head_pose_degrees=head_pose_degrees,
            unauthorized_objects_count=unauthorized_objects_count,
            risk_weights=risk_weights,
            thresholds=thresholds
        )
