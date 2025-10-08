#!/usr/bin/env python3
"""
NeuroDrive Main Integration Module
Combines monitoring modules into a unified production-ready system
"""

import cv2
import numpy as np
import threading
import time
import yaml
import logging
import argparse
from collections import deque
import json
from datetime import datetime
import sys
import os
from typing import Dict, Any

# ---- Module imports (adjust names if your modules differ) ----
try:
    from in_cabin.audio_analysis import AudioAnalyzer
except Exception:
    AudioAnalyzer = None

try:
    from environment.low_light import LowLightDetector
except Exception:
    LowLightDetector = None

try:
    from environment.weather_detection import WeatherDetector
except Exception:
    WeatherDetector = None

try:
    from driver_monitoring.head_pose import HeadPoseTracker
except Exception:
    HeadPoseTracker = None

# -----------------------------------------------------------------------------
# Utility: safe loader for config
# -----------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    default_cfg = {
        "system": {
            "log_level": "INFO",
            "log_file": "neurodrive.log",
            "fps": 15,
        },
        "cameras": {
            "driver_cam": 0,
            "cabin_cam": 1,
            "external_cam": 2,
            "ir_cam": -1
        },
        "alerts": {
            "low_risk": 20,
            "medium_risk": 40,
            "high_risk": 70,
            "critical_risk": 90
        },
        "risk_weights": {
            "distraction": 0.4,
            "phone_use": 0.2,
            "weather_risk": 0.2,
            "low_light": 0.2
        }
    }
    if not os.path.exists(path):
        print(f"[WARN] Config {path} not found, using default values")
        return default_cfg

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Merge defaults (shallow)
    for k, v in default_cfg.items():
        if k not in cfg:
            cfg[k] = v
    return cfg

# -----------------------------------------------------------------------------
# Main NeuroDrive System
# -----------------------------------------------------------------------------
class NeuroDriveSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self._setup_logging()

        self.is_running = False
        self.system_start_time = None

        # Camera resources
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.camera_frames: Dict[str, Any] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}

        # Modules
        self.modules: Dict[str, Any] = {}

        # Buffers
        self.results_buffer = deque(maxlen=2000)
        self.alerts_buffer = deque(maxlen=500)
        self.risk_history = deque(maxlen=100)

        self.current_risk_score = 0.0

        # Initialize modules
        self._initialize_modules()

        self.processing_thread: threading.Thread = None

        self.logger.info("NeuroDrive system instance created")

    # ---------------------------
    # Logging & initialization
    # ---------------------------
    def _setup_logging(self):
        level_name = self.config.get("system", {}).get("log_level", "INFO")
        log_level = getattr(logging, level_name.upper(), logging.INFO)
        log_file = self.config.get("system", {}).get("log_file", "neurodrive.log")

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger("NeuroDrive")

    def _initialize_modules(self):
        """Instantiate available modules and attach them to self.modules"""
        try:
            # Audio
            if AudioAnalyzer is not None:
                try:
                    self.modules["audio"] = AudioAnalyzer()
                    self.logger.info("AudioAnalyzer initialized")
                except Exception as e:
                    self.logger.warning(f"AudioAnalyzer failed to initialize: {e}")
                    self.modules["audio"] = None
            else:
                self.logger.warning("AudioAnalyzer class not found; audio disabled")
                self.modules["audio"] = None

            # Low-light
            if LowLightDetector is not None:
                try:
                    self.modules["low_light"] = LowLightDetector()
                    self.logger.info("LowLightDetector initialized")
                except Exception as e:
                    self.logger.warning(f"LowLightDetector failed to initialize: {e}")
                    self.modules["low_light"] = None
            else:
                self.logger.warning("LowLightDetector class not found; low_light disabled")
                self.modules["low_light"] = None

            # Weather
            if WeatherDetector is not None:
                try:
                    self.modules["weather"] = WeatherDetector()
                    self.logger.info("WeatherDetector initialized")
                except Exception as e:
                    self.logger.warning(f"WeatherDetector failed to initialize: {e}")
                    self.modules["weather"] = None
            else:
                self.logger.warning("WeatherDetector class not found; weather disabled")
                self.modules["weather"] = None

            # Head pose
            if HeadPoseTracker is not None:
                try:
                    self.modules["head_pose"] = HeadPoseTracker()
                    self.logger.info("HeadPoseTracker initialized")
                except Exception as e:
                    self.logger.warning(f"HeadPoseTracker failed to initialize: {e}")
                    self.modules["head_pose"] = None
            else:
                self.logger.warning("HeadPoseTracker class not found; head_pose disabled")
                self.modules["head_pose"] = None

        except Exception as e:
            self.logger.error(f"Error during module initialization: {e}")

    # ---------------------------
    # Camera management
    # ---------------------------
    def initialize_cameras(self):
        camera_cfg = self.config.get("cameras", {})
        opened = 0
        for cam_name, cam_index in camera_cfg.items():
            try:
                # skip invalid index
                if cam_index is None or cam_index < 0:
                    self.logger.info(f"Skipping camera '{cam_name}' (index {cam_index})")
                    continue
                cap = cv2.VideoCapture(int(cam_index))
                if not cap.isOpened():
                    self.logger.warning(f"Camera '{cam_name}' index {cam_index} could not be opened")
                    continue

                # set properties (best effort)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                try:
                    cap.set(cv2.CAP_PROP_FPS, int(self.config["system"]["fps"]))
                except Exception:
                    pass

                self.cameras[cam_name] = cap
                self.camera_frames[cam_name] = None
                opened += 1
                self.logger.info(f"Camera '{cam_name}' opened (index {cam_index})")
            except Exception as e:
                self.logger.error(f"Error opening camera {cam_name}: {e}")

        return opened > 0

    def _camera_capture_loop(self, cam_name: str, cap: cv2.VideoCapture):
        fps = max(1, int(self.config.get("system", {}).get("fps", 10)))
        period = 1.0 / fps
        self.logger.debug(f"Camera loop started for {cam_name} with {fps} fps")

        while self.is_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    self.logger.debug(f"Camera {cam_name} read failed; retrying")
                    time.sleep(0.05)
                    continue
                self.camera_frames[cam_name] = frame
                time.sleep(period)
            except Exception as e:
                self.logger.error(f"Camera {cam_name} capture error: {e}")
                time.sleep(0.1)

    def start_camera_threads(self):
        for name, cap in self.cameras.items():
            t = threading.Thread(target=self._camera_capture_loop, args=(name, cap), daemon=True)
            t.start()
            self.camera_threads[name] = t
            self.logger.info(f"Started camera thread: {name}")

    # ---------------------------
    # System start / stop
    # ---------------------------
    def start_system(self) -> bool:
        try:
            if not self.initialize_cameras():
                self.logger.error("No cameras available; startup aborted")
                return False

            # audio: If audio module has start() method, call it
            audio_mod = self.modules.get("audio")
            if audio_mod is not None:
                try:
                    start_fn = getattr(audio_mod, "start", None)
                    if callable(start_fn):
                        start_fn()
                        self.logger.info("Audio monitoring started")
                except Exception as e:
                    self.logger.warning(f"Audio monitoring start failed: {e}")

            self.is_running = True
            self.system_start_time = time.time()
            self.start_camera_threads()

            self.processing_thread = threading.Thread(target=self._main_processing_loop, daemon=True)
            self.processing_thread.start()

            self.logger.info("NeuroDrive system started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            return False

    def stop_system(self):
        self.logger.info("Stopping NeuroDrive system...")
        self.is_running = False
        # stop audio if available
        audio_mod = self.modules.get("audio")
        if audio_mod is not None:
            stop_fn = getattr(audio_mod, "stop", None)
            if callable(stop_fn):
                try:
                    stop_fn()
                except Exception:
                    pass

        # release camera handles
        for name, cap in self.cameras.items():
            try:
                cap.release()
            except Exception:
                pass

        # join threads (best effort)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.logger.info("NeuroDrive system stopped")

    # ---------------------------
    # Main processing loop
    # ---------------------------
    def _main_processing_loop(self):
        fps = max(1, int(self.config.get("system", {}).get("fps", 10)))
        period = 1.0 / fps
        self.logger.debug("Entering main processing loop")
        while self.is_running:
            t0 = time.time()
            results: Dict[str, Any] = {}

            # Gather frames
            driver_frame = self.camera_frames.get("driver_cam")
            cabin_frame = self.camera_frames.get("cabin_cam")
            external_frame = self.camera_frames.get("external_cam")
            ir_frame = self.camera_frames.get("ir_cam")

            # 1) Head pose (driver-facing)
            try:
                head_mod = self.modules.get("head_pose")
                if head_mod and driver_frame is not None:
                    # expects (annotated_frame, analysis)
                    annotated, analysis = head_mod.process_frame(driver_frame)
                    results["head_pose"] = analysis or {}
                    results["annotated_driver_frame"] = annotated
                else:
                    results["head_pose"] = {}
            except Exception as e:
                self.logger.exception(f"Error in head_pose processing: {e}")
                results["head_pose"] = {}

            # 2) Low light detection (use driver or IR)
            try:
                ll_mod = self.modules.get("low_light")
                if ll_mod and (driver_frame is not None):
                    ll_res = ll_mod.detect(driver_frame)
                    results["low_light"] = ll_res or {}
                else:
                    results["low_light"] = {}
            except Exception as e:
                self.logger.exception(f"Error in low_light processing: {e}")
                results["low_light"] = {}

            # 3) Weather detection (use external_frame)
            try:
                w_mod = self.modules.get("weather")
                if w_mod and external_frame is not None:
                    w_res = w_mod.detect(external_frame)
                    # try to normalize to a structure with risk_score and is_poor_weather
                    results["weather"] = {
                        "weather": w_res.get("weather") if isinstance(w_res, dict) else str(w_res),
                        "is_poor_weather": bool(w_res.get("is_poor_weather")) if isinstance(w_res, dict) else False,
                        "risk_score": float(w_res.get("risk_score", 0.0)) if isinstance(w_res, dict) else 0.0
                    }
                else:
                    results["weather"] = {}
            except Exception as e:
                self.logger.exception(f"Error in weather processing: {e}")
                results["weather"] = {}

            # 4) Audio analysis (non-blocking)
            try:
                a_mod = self.modules.get("audio")
                if a_mod:
                    # prefer an analyze() method that returns latest analysis
                    if hasattr(a_mod, "analyze"):
                        audio_res = a_mod.analyze()
                    elif hasattr(a_mod, "get_analysis_results"):
                        audio_res = a_mod.get_analysis_results()
                    else:
                        audio_res = {}
                    results["audio"] = audio_res or {}
                else:
                    results["audio"] = {}
            except Exception as e:
                self.logger.exception(f"Error in audio processing: {e}")
                results["audio"] = {}

            # 5) Combine into risk score
            try:
                risk_value = self._calculate_combined_risk_score(results)
                results["risk"] = {
                    "score": float(risk_value),
                    "level": self._get_risk_level(risk_value),
                    "timestamp": time.time()
                }
                self.current_risk_score = float(risk_value)
            except Exception as e:
                self.logger.exception(f"Error calculating risk score: {e}")
                results["risk"] = {"score": 0.0, "level": "unknown", "timestamp": time.time()}

            # 6) Alerts and logging
            try:
                self._process_alerts(results["risk"]["score"], results)
            except Exception as e:
                self.logger.exception(f"Error processing alerts: {e}")

            # 7) Store results
            results["timestamp"] = time.time()
            results["processing_time"] = time.time() - t0
            self.results_buffer.append(results)

            # 8) (Optional) Display annotated driver frame and overlays if available
            try:
                if "annotated_driver_frame" in results and results["annotated_driver_frame"] is not None:
                    frame = results["annotated_driver_frame"]
                    # overlay the risk score
                    risk_score = results["risk"]["score"]
                    level = results["risk"]["level"]
                    color = (0, 255, 0)
                    if level in ("high", "critical"):
                        color = (0, 0, 255)
                    elif level == "medium":
                        color = (0, 165, 255)
                    cv2.putText(frame, f"Risk: {risk_score:.1f} ({level})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow("NeuroDrive - Driver", frame)
                    # handle key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.logger.info("User requested quit via UI")
                        self.stop_system()
                        break
            except Exception:
                # non-critical: continue
                pass

            # Sleep to maintain target fps
            elapsed = time.time() - t0
            sleep_for = max(0.0, period - elapsed)
            time.sleep(sleep_for)

    # ---------------------------
    # Risk calculation & alerts
    # ---------------------------
    def _calculate_combined_risk_score(self, results: Dict[str, Any]) -> float:
        weights = self.config.get("risk_weights", {})
        # fallback defaults
        w_distraction = float(weights.get("distraction", 0.4))
        w_phone = float(weights.get("phone_use", 0.2))
        w_weather = float(weights.get("weather_risk", 0.2))
        w_low_light = float(weights.get("low_light", 0.2))

        total_score = 0.0
        total_weight = 0.0

        # Head pose -> compute simple distraction score (0..100)
        head = results.get("head_pose", {})
        if head:
            # example: if analysis contains 'looking_forward' boolean and 'gaze_deviation'
            looking_forward = head.get("looking_forward")
            gaze_dev = head.get("gaze_deviation", 0.0)
            # high penalty if not looking forward and large deviation
            head_score = 0.0
            if looking_forward is False:
                head_score = min(100.0, 100.0 * (abs(gaze_dev) / max(1.0, self.config.get("head_pose", {}).get("max_dev", 45.0))))
            else:
                head_score = 0.0
            total_score += head_score * w_distraction
            total_weight += w_distraction

        # Audio -> if loud, create phone/use noise risk
        audio = results.get("audio", {})
        audio_risk = 0.0
        if audio:
            loud = audio.get("loud", False)
            rms = float(audio.get("rms", 0.0) or 0.0)
            if loud:
                audio_risk = 100.0
            else:
                # small risk proportional to RMS
                audio_risk = min(50.0, (rms / 1000.0) * 100.0)
            total_score += audio_risk * w_phone
            total_weight += w_phone

        # Weather
        weather = results.get("weather", {})
        weather_risk = 0.0
        if weather:
            if weather.get("is_poor_weather", False):
                weather_risk = float(weather.get("risk_score", 50.0))
            else:
                weather_risk = 0.0
            total_score += weather_risk * w_weather
            total_weight += w_weather

        # Low-light
        low_light = results.get("low_light", {})
        ll_risk = 0.0
        if low_light:
            if low_light.get("low_light", False):
                ll_risk = 80.0
            else:
                ll_risk = 0.0
            total_score += ll_risk * w_low_light
            total_weight += w_low_light

        final = 0.0
        if total_weight > 0:
            final = total_score / total_weight

        # Temporal smoothing
        self.risk_history.append(final)
        smoothed = float(np.mean(list(self.risk_history))) if len(self.risk_history) > 0 else final

        return smoothed

    def _get_risk_level(self, risk_score: float) -> str:
        alerts = self.config.get("alerts", {})
        if risk_score >= float(alerts.get("critical_risk", 90)):
            return "critical"
        if risk_score >= float(alerts.get("high_risk", 70)):
            return "high"
        if risk_score >= float(alerts.get("medium_risk", 40)):
            return "medium"
        if risk_score >= float(alerts.get("low_risk", 20)):
            return "low"
        return "normal"

    def _process_alerts(self, risk_score: float, results: Dict[str, Any]):
        """Create alert events and push to alerts_buffer/log as needed"""
        level = self._get_risk_level(risk_score)
        ts = time.time()

        if level in ("medium", "high", "critical"):
            alert = {
                "timestamp": ts,
                "level": level,
                "type": "risk_level",
                "score": float(risk_score),
            }
            self.alerts_buffer.append(alert)
            self.logger.warning(f"[ALERT] {level.upper()} risk: {risk_score:.1f}")

        # specific head pose alert: looking away for long duration
        head = results.get("head_pose", {})
        if head:
            if head.get("is_looking_away", False):
                dur = float(head.get("looking_away_duration", 0.0))
                if dur > 2.0:  # seconds threshold for alert
                    alert = {"timestamp": ts, "type": "distraction", "message": f"Looking away {dur:.1f}s", "severity": "high"}
                    self.alerts_buffer.append(alert)
                    self.logger.warning(f"[ALERT] Distracted: {alert['message']}")

        # specific audio events (if module returns detected_events)
        audio = results.get("audio", {})
        if audio:
            events = audio.get("detected_events", [])
            for ev in events:
                if ev.get("type") in ("argument", "phone_ringing", "shouting"):
                    alert = {"timestamp": ts, "type": "audio", "message": ev.get("type"), "confidence": ev.get("confidence", 1.0)}
                    self.alerts_buffer.append(alert)
                    self.logger.info(f"[AUDIO ALERT] {ev}")

    # ---------------------------
    # Utilities to export results
    # ---------------------------
    def export_recent_results(self, path: str = "neurodrive_recent.json"):
        """Dump results_buffer to JSON file (last N results)."""
        try:
            with open(path, "w") as f:
                json.dump(list(self.results_buffer), f, default=str, indent=2)
            self.logger.info(f"Exported recent results to {path}")
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")

# -----------------------------------------------------------------------------
# CLI Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NeuroDrive integrated system (main)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV display windows")
    args = parser.parse_args()

    system = NeuroDriveSystem(config_path=args.config)
    ok = system.start_system()
    if not ok:
        print("Failed to start system. Check logs.")
        return

    try:
        # Keep the main thread alive until system stops
        while system.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        system.logger.info("Keyboard interrupt received - shutting down")
    finally:
        # Export a snapshot of recent results
        system.export_recent_results()
        system.stop_system()


if __name__ == "__main__":
    main()
