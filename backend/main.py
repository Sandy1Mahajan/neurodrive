from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import yaml
import os
from .model.file import Model


class Thresholds(BaseModel):
    eyeClosureTimeSeconds: float = 2.0
    distractionDetectedThreshold: float = 0.5
    headPoseDegreesThreshold: float = 15.0
    unauthorizedObjectsThreshold: int = 1
    speedLimitKmh: int = 80


class RiskWeights(BaseModel):
    drowsiness: float = 0.25
    distraction: float = 0.30
    headPose: float = 0.25
    unauthorizedObjects: float = 0.20


class InferenceInput(BaseModel):
    eyeClosureRatio: float = Field(0.2, ge=0.0, le=1.0)
    phoneUsage: bool = False
    speed: int = Field(65, ge=0)
    headPoseDegrees: Optional[float] = None
    unauthorizedObjectsCount: Optional[int] = None


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "risk_weights": RiskWeights().dict(),
            "thresholds": Thresholds().dict(),
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rw = data.get("risk_weights", {})
    th = data.get("thresholds", {})
    return {
        "risk_weights": RiskWeights(**rw).dict(),
        "thresholds": Thresholds(**th).dict(),
    }


def compute_status_text(score: float) -> str:
    if score < 30:
        return "Critical Risk"
    if score < 50:
        return "High Risk"
    if score < 70:
        return "Elevated Risk"
    if score < 85:
        return "Low Risk"
    return "Alert"


def compute_risk(payload: InferenceInput, cfg: Dict[str, Any]) -> Dict[str, Any]:
    weights = RiskWeights(**cfg["risk_weights"]).dict()
    th = Thresholds(**cfg["thresholds"]).dict()

    eyeClosureTimeSeconds = float(payload.eyeClosureRatio) * 3.0
    distractionDetected = bool(payload.phoneUsage)
    headPoseDegrees = abs(float(payload.headPoseDegrees if payload.headPoseDegrees is not None else 10.0))
    unauthorizedObjectsCount = int(payload.unauthorizedObjectsCount if payload.unauthorizedObjectsCount is not None else 0)
    speed = int(payload.speed)

    drowsinessOk = eyeClosureTimeSeconds < th["eyeClosureTimeSeconds"]
    distractionOk = not distractionDetected
    headPoseOk = headPoseDegrees <= th["headPoseDegreesThreshold"]
    objectsOk = unauthorizedObjectsCount <= th["unauthorizedObjectsThreshold"]
    speedOk = speed <= th["speedLimitKmh"]

    drowsinessSub = 100.0 if drowsinessOk else clamp(100 - ((eyeClosureTimeSeconds - th["eyeClosureTimeSeconds"]) * 40), 0, 100)
    distractionSub = 100.0 if distractionOk else 35.0
    headPoseSub = 100.0 if headPoseOk else clamp(100 - ((headPoseDegrees - th["headPoseDegreesThreshold"]) * 3), 0, 100)
    objectsSub = 100.0 if objectsOk else 50.0
    speedSub = 100.0 if speedOk else clamp(100 - ((speed - th["speedLimitKmh"]) * 1.2), 0, 100)

    denom = weights["drowsiness"] + weights["distraction"] + weights["headPose"] + weights["unauthorizedObjects"]
    baseWeighted = (
        drowsinessSub * weights["drowsiness"] +
        distractionSub * weights["distraction"] +
        headPoseSub * weights["headPose"] +
        objectsSub * weights["unauthorizedObjects"]
    ) / denom

    riskScore = clamp(round(baseWeighted * (speedSub / 100.0)), 0, 100)
    statusText = compute_status_text(riskScore)

    metrics = {
        "drowsiness": {"value": round(eyeClosureTimeSeconds, 2), "threshold": th["eyeClosureTimeSeconds"], "unit": "s", "ok": drowsinessOk},
        "distraction": {"value": 1 if distractionDetected else 0, "threshold": th["distractionDetectedThreshold"], "unit": "", "ok": distractionOk},
        "headPose": {"value": round(headPoseDegrees, 1), "threshold": th["headPoseDegreesThreshold"], "unit": "°", "ok": headPoseOk},
        "unauthorizedObjects": {"value": unauthorizedObjectsCount, "threshold": th["unauthorizedObjectsThreshold"], "unit": "", "ok": objectsOk},
        "speed": {"value": speed, "threshold": th["speedLimitKmh"], "unit": "km/h", "ok": speedOk},
    }

    alerts = []
    if not drowsinessOk:
        alerts.append({"type": "Microsleep Event", "detail": f"Eye closure {eyeClosureTimeSeconds:.2f}s"})
    if not distractionOk:
        alerts.append({"type": "Distraction Detected", "detail": "Phone usage"})
    if not headPoseOk:
        alerts.append({"type": "Head Pose Deviation", "detail": f"{headPoseDegrees:.1f}°"})
    if not objectsOk:
        alerts.append({"type": "Unauthorized Object", "detail": f"Count {unauthorizedObjectsCount}"})
    if not speedOk:
        alerts.append({"type": "Speed Violation", "detail": f"{speed} km/h > {th['speedLimitKmh']} km/h"})

    return {
        "riskScore": riskScore,
        "statusText": statusText,
        "metrics": metrics,
        "alerts": alerts,
        "weights": weights,
        "thresholds": th,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="NeuroDrive DMS Backend", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config_path = os.environ.get("NEURODRIVE_CONFIG", os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    config = load_config(os.path.abspath(config_path))

    model = Model()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/config")
    def get_config():
        return config

    @app.post("/api/v1/infer")
    def infer(payload: InferenceInput):
        try:
            # Model stub: could use webcam/image inputs; here we focus on metrics -> risk
            _ = model.is_ready()
            result = compute_risk(payload, config)
            return result
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


