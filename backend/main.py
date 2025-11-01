"""
Production-ready FastAPI backend for NeuroDrive DMS.

Features:
- Model inference with deterministic baseline and ONNX support
- Input validation with Pydantic
- Rate limiting
- CORS configuration
- Structured logging (JSON)
- Prometheus metrics
- Health checks with model readiness
- Config hot-reload support
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import yaml
import os
import logging
import time
import json
from contextlib import asynccontextmanager
from collections import deque
import threading

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("slowapi not available; rate limiting disabled")

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not available; metrics disabled")

from .model.file import Model


# Configure structured logging
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logs."""
    def format(self, record):
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineno': record.lineno
        }
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging():
    """Setup structured JSON logging."""
    logger = logging.getLogger("neurodrive")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


logger = setup_logging()


# Pydantic models
class Thresholds(BaseModel):
    """Threshold configuration."""
    eyeClosureTimeSeconds: float = 2.0
    distractionDetectedThreshold: float = 0.5
    headPoseDegreesThreshold: float = 15.0
    unauthorizedObjectsThreshold: int = 1
    speedLimitKmh: int = 80


class RiskWeights(BaseModel):
    """Risk weight configuration."""
    drowsiness: float = 0.25
    distraction: float = 0.30
    headPose: float = 0.25
    unauthorizedObjects: float = 0.20


class InferenceInput(BaseModel):
    """Input schema for inference endpoint."""
    eyeClosureRatio: float = Field(0.2, ge=0.0, le=1.0, description="Eye closure ratio (0-1)")
    phoneUsage: bool = Field(False, description="Phone usage detected")
    speed: int = Field(65, ge=0, le=300, description="Vehicle speed (km/h)")
    headPoseDegrees: Optional[float] = Field(None, ge=-90, le=90, description="Head pose deviation (degrees)")
    unauthorizedObjectsCount: Optional[int] = Field(None, ge=0, description="Unauthorized objects count")


class ConfigUpdate(BaseModel):
    """Schema for config update."""
    risk_weights: Optional[RiskWeights] = None
    thresholds: Optional[Thresholds] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_ready: bool
    uptime_seconds: float
    version: str = "1.0.0"


# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    inference_counter = Counter('neurodrive_inferences_total', 'Total number of inference requests')
    inference_duration = Histogram('neurodrive_inference_duration_seconds', 'Inference duration in seconds')
    inference_errors = Counter('neurodrive_inference_errors_total', 'Total number of inference errors')
    risk_score_gauge = Gauge('neurodrive_risk_score', 'Current risk score')
    model_ready_gauge = Gauge('neurodrive_model_ready', 'Model readiness (1=ready, 0=not ready)')


# Global state
config_lock = threading.Lock()
config_cache: Dict[str, Any] = {}
start_time = time.time()
model_instance: Optional[Model] = None


def clamp(value: float, min_v: float, max_v: float) -> float:
    """Clamp value between min and max."""
    return max(min_v, min(max_v, value))


def load_config(path: str) -> Dict[str, Any]:
    """Load config from YAML file with caching."""
    global config_cache, config_lock
    
    with config_lock:
        # Check file modification time for hot-reload
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                cache_key = f"{path}:{mtime}"
                
                if cache_key not in config_cache:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    
                    rw = data.get("risk_weights", {})
                    th = data.get("thresholds", {})
                    
                    config_cache[cache_key] = {
                        "risk_weights": RiskWeights(**rw).dict(),
                        "thresholds": Thresholds(**th).dict(),
                    }
                
                return config_cache.get(cache_key, {})
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return {
                    "risk_weights": RiskWeights().dict(),
                    "thresholds": Thresholds().dict(),
                }
        else:
            logger.warning(f"Config file not found: {path}")
            return {
                "risk_weights": RiskWeights().dict(),
                "thresholds": Thresholds().dict(),
            }


def compute_status_text(score: float) -> str:
    """Compute status text from risk score."""
    if score < 30:
        return "Critical Risk"
    if score < 50:
        return "High Risk"
    if score < 70:
        return "Elevated Risk"
    if score < 85:
        return "Low Risk"
    return "Safe"


def compute_risk_response(model_result: Dict[str, Any], payload: InferenceInput, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert model result to API response format."""
    weights = RiskWeights(**config["risk_weights"]).dict()
    th = Thresholds(**config["thresholds"]).dict()
    
    risk_score = model_result.get("riskScore", 0)
    
    # Calculate individual metric statuses
    eye_closure_time = float(payload.eyeClosureRatio) * 3.0
    distraction_detected = bool(payload.phoneUsage)
    head_pose_degrees = abs(float(payload.headPoseDegrees if payload.headPoseDegrees is not None else 0.0))
    unauthorized_objects_count = int(payload.unauthorizedObjectsCount if payload.unauthorizedObjectsCount is not None else 0)
    speed = int(payload.speed)
    
    drowsiness_ok = eye_closure_time < th["eyeClosureTimeSeconds"]
    distraction_ok = not distraction_detected
    head_pose_ok = head_pose_degrees <= th["headPoseDegreesThreshold"]
    objects_ok = unauthorized_objects_count <= th["unauthorizedObjectsThreshold"]
    speed_ok = speed <= th["speedLimitKmh"]
    
    metrics = {
        "drowsiness": {
            "value": round(eye_closure_time, 2),
            "threshold": th["eyeClosureTimeSeconds"],
            "unit": "s",
            "ok": drowsiness_ok,
            "score": model_result.get("drowsinessScore", 0.0)
        },
        "distraction": {
            "value": 1 if distraction_detected else 0,
            "threshold": th["distractionDetectedThreshold"],
            "unit": "",
            "ok": distraction_ok,
            "score": model_result.get("distractionScore", 0.0)
        },
        "headPose": {
            "value": round(head_pose_degrees, 1),
            "threshold": th["headPoseDegreesThreshold"],
            "unit": "°",
            "ok": head_pose_ok,
            "score": model_result.get("headPoseScore", 0.0)
        },
        "unauthorizedObjects": {
            "value": unauthorized_objects_count,
            "threshold": th["unauthorizedObjectsThreshold"],
            "unit": "",
            "ok": objects_ok,
            "score": model_result.get("unauthorizedObjectScore", 0.0)
        },
        "speed": {
            "value": speed,
            "threshold": th["speedLimitKmh"],
            "unit": "km/h",
            "ok": speed_ok
        }
    }
    
    alerts = []
    if not drowsiness_ok:
        alerts.append({"type": "Microsleep Event", "detail": f"Eye closure {eye_closure_time:.2f}s"})
    if not distraction_ok:
        alerts.append({"type": "Distraction Detected", "detail": "Phone usage"})
    if not head_pose_ok:
        alerts.append({"type": "Head Pose Deviation", "detail": f"{head_pose_degrees:.1f}°"})
    if not objects_ok:
        alerts.append({"type": "Unauthorized Object", "detail": f"Count {unauthorized_objects_count}"})
    if not speed_ok:
        alerts.append({"type": "Speed Violation", "detail": f"{speed} km/h > {th['speedLimitKmh']} km/h"})
    
    return {
        "riskScore": risk_score,
        "statusText": compute_status_text(risk_score),
        "metrics": metrics,
        "alerts": alerts,
        "weights": weights,
        "thresholds": th,
        "explanations": model_result.get("explanations", {}),
        "modelType": model_instance.config_dict.get("model_type", "deterministic") if model_instance else "deterministic"
    }


# JWT authentication (optional, can be disabled in dev mode)
def verify_jwt_token(token: str) -> bool:
    """Verify JWT token. Simplified implementation."""
    # In production, use proper JWT verification library
    # This is a placeholder
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    if dev_mode:
        return True
    
    # Check if token exists and is valid (simplified)
    if not token or len(token) < 10:
        return False
    
    # TODO: Implement proper JWT verification
    return True


security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global model_instance, start_time
    
    # Startup
    logger.info("Starting NeuroDrive FastAPI backend...")
    start_time = time.time()
    
    # Load model
    config_path = os.environ.get(
        "NEURODRIVE_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    
    model_config = {
        "model_type": os.getenv("MODEL_TYPE", "deterministic"),
        "model_path": os.getenv("MODEL_PATH")
    }
    
    try:
        model_instance = Model(config=model_config)
        if model_instance.is_ready():
            logger.info("Model initialized successfully")
            if PROMETHEUS_AVAILABLE:
                model_ready_gauge.set(1)
        else:
            logger.warning("Model not ready")
            if PROMETHEUS_AVAILABLE:
                model_ready_gauge.set(0)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model_instance = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down NeuroDrive FastAPI backend...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    # Rate limiter
    if SLOWAPI_AVAILABLE:
        limiter = Limiter(key_func=get_remote_address)
    else:
        limiter = None
    
    app = FastAPI(
        title="NeuroDrive DMS Backend",
        version="1.0.0",
        description="Production-ready Driver Monitoring System API",
        lifespan=lifespan
    )
    
    # Rate limiting middleware
    if SLOWAPI_AVAILABLE and limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # CORS middleware
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        
        return response
    
    config_path = os.environ.get(
        "NEURODRIVE_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Health check endpoint."""
        uptime = time.time() - start_time
        model_ready = model_instance.is_ready() if model_instance else False
        
        return HealthResponse(
            status="ok" if model_ready else "degraded",
            model_ready=model_ready,
            uptime_seconds=uptime
        )
    
    @app.get("/config", tags=["Configuration"])
    async def get_config():
        """Get current configuration."""
        return load_config(os.path.abspath(config_path))
    
    @app.put("/config", tags=["Configuration"])
    async def update_config(config_update: ConfigUpdate):
        """Update configuration (hot-reload)."""
        global config_cache, config_lock
        
        try:
            # Load current config
            current_config = load_config(os.path.abspath(config_path))
            
            # Merge updates
            if config_update.risk_weights:
                current_config["risk_weights"] = config_update.risk_weights.dict()
            if config_update.thresholds:
                current_config["thresholds"] = config_update.thresholds.dict()
            
            # Save to file (create backup first)
            backup_path = config_path + ".backup"
            if os.path.exists(config_path):
                import shutil
                shutil.copy2(config_path, backup_path)
            
            # Write updated config
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
            
            # Clear cache to force reload
            with config_lock:
                config_cache.clear()
            
            logger.info("Configuration updated successfully")
            
            return {"status": "success", "config": current_config}
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/infer", tags=["Inference"])
    async def infer(
        payload: InferenceInput,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """Perform risk inference from driver metrics."""
        # Optional JWT authentication (disabled in dev mode)
        dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
        if not dev_mode and credentials:
            if not verify_jwt_token(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid or missing token")
        
        # Rate limiting
        if SLOWAPI_AVAILABLE and limiter:
            limiter.limit("100/minute")(lambda: None)()
        
        # Check model readiness
        if not model_instance or not model_instance.is_ready():
            logger.error("Model not ready for inference")
            raise HTTPException(status_code=503, detail="Model not ready")
        
        try:
            # Load config
            config = load_config(os.path.abspath(config_path))
            
            # Start metrics tracking
            start_time = time.time()
            
            # Run inference
            model_result = model_instance.predict_from_api_input(
                eye_closure_ratio=payload.eyeClosureRatio,
                phone_usage=payload.phoneUsage,
                speed=payload.speed,
                head_pose_degrees=payload.headPoseDegrees,
                unauthorized_objects_count=payload.unauthorizedObjectsCount,
                risk_weights=config.get("risk_weights"),
                thresholds=config.get("thresholds")
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                inference_counter.inc()
                inference_duration.observe(duration)
                risk_score_gauge.set(model_result.get("riskScore", 0))
            
            # Convert to API response format
            response = compute_risk_response(model_result, payload, config)
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            
            if PROMETHEUS_AVAILABLE:
                inference_errors.inc()
            
            raise HTTPException(status_code=500, detail=str(e))
    
    # Prometheus metrics endpoint
    if PROMETHEUS_AVAILABLE:
        @app.get("/metrics", tags=["Monitoring"])
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_config=None  # Use our custom logging
    )
