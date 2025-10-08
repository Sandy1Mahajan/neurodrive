"""
NeuroDrive Head Pose Estimation Module
Advanced head pose tracking for gaze direction and distraction detection
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import math
from typing import Dict, List, Tuple, Optional
from collections import deque
import yaml
import logging
from dataclasses import dataclass


@dataclass
class HeadPose:
    """Head pose data structure"""
    pitch: float = 0.0  # Up/down rotation (degrees)
    yaw: float = 0.0    # Left/right rotation (degrees)
    roll: float = 0.0   # Tilt rotation (degrees)
    confidence: float = 0.0
    valid: bool = False


@dataclass
class GazeMetrics:
    """Gaze tracking metrics"""
    looking_forward: bool = True
    gaze_deviation: float = 0.0  # Degrees from center
    attention_score: float = 1.0  # 0-1 scale
    distraction_type: str = "none"  # none, phone, passenger, mirror, window
    distraction_duration: float = 0.0


@dataclass
class DistractionState:
    """Overall distraction state"""
    is_distracted: bool = False
    distraction_level: int = 0  # 0-4 scale
    alert_triggered: bool = False
    consecutive_frames: int = 0
    total_distractions: int = 0
    avg_attention: float = 1.0


class HeadPoseTracker:
    """Advanced head pose estimation and gaze tracking"""

    def __init__(self, config_path: str = "config.yaml"):
        # Logging
        self.logger = logging.getLogger(__name__)
        # Load config safely
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except Exception:
            self.logger.warning(f"Config {config_path} not found or invalid — using defaults.")
            self.config = {}

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Note: FaceMesh should be closed when the object is destroyed (we don't do explicit close here,
        # but you may call `tracker.face_mesh.close()` if desired)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 3D model points for head pose estimation (standard approx)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        # MediaPipe face mesh indices for key points (these indices are taken from mediapipe mesh)
        self.key_point_indices = [1, 152, 33, 263, 61, 291]  # nose, chin, left eye, right eye, left mouth, right mouth

        # Thresholds from config with defaults
        pose_config = self.config.get('models', {}).get('head_pose', {})
        self.PITCH_THRESHOLD = float(pose_config.get('pitch_threshold', 30))
        self.YAW_THRESHOLD = float(pose_config.get('yaw_threshold', 45))
        self.ROLL_THRESHOLD = float(pose_config.get('roll_threshold', 25))

        # State tracking
        self.head_pose = HeadPose()
        self.gaze_metrics = GazeMetrics()
        self.distraction_state = DistractionState()

        # Historical data for smoothing and analysis
        self.pose_history = deque(maxlen=30)  # e.g. 1 second at 30fps
        self.attention_history = deque(maxlen=300)  # e.g. 10 seconds
        self.distraction_events = deque(maxlen=100)

        # Calibration data
        self.baseline_pose = None
        self.calibration_frames = 0
        self.calibrated = False

        # Timing
        self.distraction_start_time = None
        self.last_forward_gaze_time = time.time()

        # Threading
        self.processing_lock = threading.Lock()

    def calibrate_baseline(self, frame: np.ndarray) -> bool:
        """Calibrate baseline head pose for the driver. Return True when calibration completes."""
        with self.processing_lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pose = self._estimate_head_pose(frame, face_landmarks)

                if pose.valid:
                    if self.baseline_pose is None:
                        self.baseline_pose = HeadPose(pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll, confidence=pose.confidence, valid=True)
                        self.calibration_frames = 1
                    else:
                        # Running average for stable baseline
                        n = self.calibration_frames
                        self.baseline_pose.pitch = (self.baseline_pose.pitch * n + pose.pitch) / (n + 1)
                        self.baseline_pose.yaw = (self.baseline_pose.yaw * n + pose.yaw) / (n + 1)
                        self.baseline_pose.roll = (self.baseline_pose.roll * n + pose.roll) / (n + 1)
                        self.baseline_pose.confidence = (self.baseline_pose.confidence * n + pose.confidence) / (n + 1)
                        self.calibration_frames += 1

                    # Consider calibrated after 30 frames (approx 1s at 30fps)
                    if self.calibration_frames >= int(self.config.get('calibration_frames', 30)):
                        self.calibrated = True
                        self.logger.info(
                            f"Head pose calibrated - Baseline: Pitch={self.baseline_pose.pitch:.1f}°, "
                            f"Yaw={self.baseline_pose.yaw:.1f}°, Roll={self.baseline_pose.roll:.1f}°"
                        )
                        return True

            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single BGR frame and return (annotated_frame, analysis_dict).
        This is the cleaned up processing routine (no UI/key handling here).
        """
        with self.processing_lock:
            start_time = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            output_frame = frame.copy()
            analysis_data: Dict = {}

            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Estimate head pose
                self.head_pose = self._estimate_head_pose(frame, face_landmarks)

                if self.head_pose.valid:
                    # Add to history
                    self.pose_history.append(self.head_pose)

                    # Analyze gaze and attention (requires calibration)
                    if self.calibrated:
                        self._analyze_gaze_attention()
                        self._update_distraction_state()

                    # Draw pose visualization / landmarks
                    output_frame = self._draw_pose_visualization(output_frame, face_landmarks)
                else:
                    # If pose estimation failed but face exists, treat as not valid
                    self._handle_no_face_detected()
            else:
                # No face found
                self._handle_no_face_detected()

            # Compile analysis data
            analysis_data = self._compile_analysis_data()

            # Performance info
            processing_time = time.time() - start_time
            analysis_data['processing_time'] = processing_time

            return output_frame, analysis_data

    def _estimate_head_pose(self, frame: np.ndarray, face_landmarks) -> HeadPose:
        """Estimate head pose using PnP algorithm (return HeadPose)."""
        h, w = frame.shape[:2]

        # Extract 2D landmarks array (x*w, y*h)
        try:
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark], dtype=np.float32)
        except Exception as e:
            self.logger.debug(f"Landmark extraction failed: {e}")
            return HeadPose(valid=False)

        # Guard that requested indices exist
        if max(self.key_point_indices) >= len(landmarks):
            self.logger.debug("Face landmark indices out-of-range for model points.")
            return HeadPose(valid=False)

        # Get key points for pose estimation
        image_points = landmarks[self.key_point_indices].astype(np.float32)

        # Camera matrix (assume focal length == width; principal point at center)
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients (assume zero)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return HeadPose(valid=False)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)

            confidence = self._calculate_pose_confidence(face_landmarks, image_points)

            return HeadPose(pitch=float(pitch), yaw=float(yaw), roll=float(roll), confidence=float(confidence), valid=True)

        except Exception as e:
            self.logger.debug(f"Pose estimation error: {e}")
            return HeadPose(valid=False)

    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])  # Roll
            y = math.atan2(-R[2, 0], sy)      # Pitch
            z = math.atan2(R[1, 0], R[0, 0])  # Yaw
        else:
            x = math.atan2(-R[1, 2], R[1, 1])  # Roll
            y = math.atan2(-R[2, 0], sy)       # Pitch
            z = 0                              # Yaw

        # Convert to degrees
        pitch = math.degrees(y)
        yaw = math.degrees(z)
        roll = math.degrees(x)

        return pitch, yaw, roll

    def _calculate_pose_confidence(self, face_landmarks, image_points: np.ndarray) -> float:
        """Calculate a confidence score for pose estimation (0..1)."""
        # MediaPipe landmarks may not include a 'visibility' attribute for all versions.
        vis_list = []
        for lm in face_landmarks.landmark:
            v = getattr(lm, 'visibility', None)
            if v is not None:
                vis_list.append(float(v))
        landmark_confidence = float(min(vis_list)) if vis_list else 1.0

        # Bounds penalty: check that image_points are inside reasonable frame bounds
        # Use actual frame size from image_points
        h = int(np.max(image_points[:, 1]) + 1) if image_points.size else 480
        w = int(np.max(image_points[:, 0]) + 1) if image_points.size else 640
        bounds_penalty = 0.0
        for point in image_points:
            if point[0] < 0 or point[0] > w or point[1] < 0 or point[1] > h:
                bounds_penalty += 0.1

        confidence = max(0.0, landmark_confidence - bounds_penalty)

        # Smooth confidence with recent history
        if len(self.pose_history) > 0:
            recent_confidences = [p.confidence for p in list(self.pose_history)[-5:] if p.valid]
            if recent_confidences:
                confidence = 0.7 * confidence + 0.3 * float(np.mean(recent_confidences))

        return float(min(1.0, confidence))

    def _analyze_gaze_attention(self):
        """Analyze gaze direction and attention level — requires self.baseline_pose to exist."""
        if not self.head_pose.valid or not self.calibrated or self.baseline_pose is None:
            return

        pitch_dev = abs(self.head_pose.pitch - self.baseline_pose.pitch)
        yaw_dev = abs(self.head_pose.yaw - self.baseline_pose.yaw)
        roll_dev = abs(self.head_pose.roll - self.baseline_pose.roll)

        # Total gaze deviation (Euclidean in angle-space)
        self.gaze_metrics.gaze_deviation = float(math.sqrt(pitch_dev ** 2 + yaw_dev ** 2 + roll_dev ** 2))

        # Looking forward if each deviation below thresholds
        self.gaze_metrics.looking_forward = (
            pitch_dev < self.PITCH_THRESHOLD and
            yaw_dev < self.YAW_THRESHOLD and
            roll_dev < self.ROLL_THRESHOLD
        )

        # Distraction classification
        self.gaze_metrics.distraction_type = self._classify_distraction_direction(
            self.head_pose.pitch - self.baseline_pose.pitch,
            self.head_pose.yaw - self.baseline_pose.yaw,
            self.head_pose.roll - self.baseline_pose.roll
        )

        max_dev = max(self.PITCH_THRESHOLD, self.YAW_THRESHOLD, self.ROLL_THRESHOLD)
        attention_factor = max(0.0, 1.0 - (self.gaze_metrics.gaze_deviation / max_dev))

        # Weight by pose confidence
        self.gaze_metrics.attention_score = float(attention_factor * self.head_pose.confidence)

        # Update histories
        self.attention_history.append(self.gaze_metrics.attention_score)

        # Timing updates
        current_time = time.time()
        if self.gaze_metrics.looking_forward:
            self.last_forward_gaze_time = current_time
            if self.distraction_start_time:
                duration = current_time - self.distraction_start_time
                self.distraction_events.append({
                    'start_time': self.distraction_start_time,
                    'duration': duration,
                    'type': self.gaze_metrics.distraction_type
                })
                self.distraction_start_time = None
        else:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            self.gaze_metrics.distraction_duration = float(current_time - self.distraction_start_time)

    def _classify_distraction_direction(self, pitch_diff: float, yaw_diff: float, roll_diff: float) -> str:
        """Classify distraction type based on head movement direction"""
        if abs(pitch_diff) < 15 and abs(yaw_diff) < 15 and abs(roll_diff) < 15:
            return "none"

        max_movement = max(abs(pitch_diff), abs(yaw_diff), abs(roll_diff))

        if abs(yaw_diff) == max_movement:
            if yaw_diff > 20:
                return "passenger"  # Looking right
            elif yaw_diff < -20:
                return "window"  # Looking left
            elif abs(yaw_diff) > 15:
                return "mirror"
        elif abs(pitch_diff) == max_movement:
            if pitch_diff > 15:
                return "dashboard"  # Looking down
            elif pitch_diff < -15:
                return "mirror"
        elif abs(roll_diff) == max_movement:
            return "phone"
        return "general"

    def _update_distraction_state(self):
        """Compute distraction level, triggers and averages"""
        current_time = time.time()

        is_currently_distracted = not self.gaze_metrics.looking_forward
        if is_currently_distracted:
            self.distraction_state.consecutive_frames += 1
            dur = self.gaze_metrics.distraction_duration
            if dur > 5.0:
                self.distraction_state.distraction_level = 4
            elif dur > 3.0:
                self.distraction_state.distraction_level = 3
            elif dur > 2.0:
                self.distraction_state.distraction_level = 2
            elif dur > 1.0:
                self.distraction_state.distraction_level = 1
            else:
                self.distraction_state.distraction_level = 0
        else:
            self.distraction_state.consecutive_frames = 0
            self.distraction_state.distraction_level = 0

        self.distraction_state.is_distracted = (
            self.distraction_state.distraction_level > 0 or
            self.distraction_state.consecutive_frames > 30
        )

        if len(self.attention_history) >= 10:
            self.distraction_state.avg_attention = float(np.mean(list(self.attention_history)[-30:]))

        if (self.distraction_state.distraction_level >= 3 or
                self.gaze_metrics.distraction_duration > 3.0):
            if not self.distraction_state.alert_triggered:
                self.distraction_state.alert_triggered = True
                self.distraction_state.total_distractions += 1
        elif self.distraction_state.distraction_level < 2:
            self.distraction_state.alert_triggered = False

    def _handle_no_face_detected(self):
        """Handle missing face landmark case"""
        self.head_pose.valid = False
        self.distraction_state.consecutive_frames += 1
        if self.distraction_state.consecutive_frames > 60:
            self.gaze_metrics.looking_forward = False
            self.gaze_metrics.distraction_type = "face_not_visible"
            self.distraction_state.is_distracted = True
            self.distraction_state.distraction_level = 3

    def _draw_pose_visualization(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw landmarks and overlays onto frame"""
        h, w = frame.shape[:2]

        # Draw face mesh contours (MediaPipe drawing)
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                None,
                self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        except Exception:
            # Some versions may not have drawing styles or other attributes; ignore drawing errors
            try:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
            except Exception:
                pass

        # Draw axes if valid
        if self.head_pose.valid:
            self._draw_pose_axes(frame, face_landmarks)

        # Draw status text
        y_offset = 30
        if self.head_pose.valid:
            cv2.putText(frame, f"Pitch: {self.head_pose.pitch:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y_offset += 20
            cv2.putText(frame, f"Yaw: {self.head_pose.yaw:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y_offset += 20
            cv2.putText(frame, f"Roll: {self.head_pose.roll:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y_offset += 20

        attention_color = (0, 255, 0)
        if self.gaze_metrics.attention_score < 0.7:
            attention_color = (0, 165, 255)
        if self.gaze_metrics.attention_score < 0.5:
            attention_color = (0, 0, 255)
        cv2.putText(frame, f"Attention: {self.gaze_metrics.attention_score:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_color, 2)
        y_offset += 20

        if self.distraction_state.is_distracted:
            distraction_color = (0, 0, 255) if self.distraction_state.distraction_level >= 3 else (0, 165, 255)
            cv2.putText(frame, f"DISTRACTED: {self.gaze_metrics.distraction_type.upper()}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, distraction_color, 2)
            if self.gaze_metrics.distraction_duration > 0:
                cv2.putText(frame, f"Duration: {self.gaze_metrics.distraction_duration:.1f}s", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, distraction_color, 2)

        if self.distraction_state.alert_triggered:
            cv2.putText(frame, "DISTRACTION ALERT!", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if not self.calibrated:
            cv2.putText(frame, f"CALIBRATING... ({self.calibration_frames}/30)", (w - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_pose_axes(self, frame: np.ndarray, face_landmarks):
        """Draw approximate 3D axes from nose tip using the Euler angles"""
        h, w = frame.shape[:2]
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        nose_tip = landmarks[1].astype(int)

        axis_length = int(min(w, h) * 0.15)
        pitch_rad = math.radians(self.head_pose.pitch)
        yaw_rad = math.radians(self.head_pose.yaw)
        roll_rad = math.radians(self.head_pose.roll)

        # Simple axis endpoints (visual heuristic)
        x_axis = np.array([axis_length * math.cos(yaw_rad), -axis_length * math.sin(pitch_rad)])
        y_axis = np.array([-axis_length * math.sin(roll_rad), -axis_length * math.cos(roll_rad)])
        z_axis = np.array([axis_length * math.sin(yaw_rad), -axis_length * math.cos(yaw_rad) * math.sin(pitch_rad)])

        try:
            cv2.arrowedLine(frame, tuple(nose_tip), tuple((nose_tip + x_axis).astype(int)), (0, 0, 255), 2)
            cv2.arrowedLine(frame, tuple(nose_tip), tuple((nose_tip + y_axis).astype(int)), (0, 255, 0), 2)
            cv2.arrowedLine(frame, tuple(nose_tip), tuple((nose_tip + z_axis).astype(int)), (255, 0, 0), 2)
        except Exception:
            pass

    def _compile_analysis_data(self) -> Dict:
        """Compile analysis dictionary for external use"""
        current_time = time.time()
        recent_distractions = [e for e in self.distraction_events if current_time - e['start_time'] < 300]

        return {
            'timestamp': current_time,
            'calibrated': self.calibrated,

            # Head pose
            'head_pose_valid': self.head_pose.valid,
            'pitch': float(self.head_pose.pitch),
            'yaw': float(self.head_pose.yaw),
            'roll': float(self.head_pose.roll),
            'pose_confidence': float(self.head_pose.confidence),

            # Deviations (if calibrated)
            'pitch_deviation': float(abs(self.head_pose.pitch - self.baseline_pose.pitch)) if (self.calibrated and self.head_pose.valid and self.baseline_pose) else 0.0,
            'yaw_deviation': float(abs(self.head_pose.yaw - self.baseline_pose.yaw)) if (self.calibrated and self.head_pose.valid and self.baseline_pose) else 0.0,
            'roll_deviation': float(abs(self.head_pose.roll - self.baseline_pose.roll)) if (self.calibrated and self.head_pose.valid and self.baseline_pose) else 0.0,

            # Gaze
            'looking_forward': bool(self.gaze_metrics.looking_forward),
            'gaze_deviation': float(self.gaze_metrics.gaze_deviation),
            'attention_score': float(self.gaze_metrics.attention_score),
            'distraction_type': self.gaze_metrics.distraction_type,
            'distraction_duration': float(self.gaze_metrics.distraction_duration),

            # Distraction state
            'is_distracted': bool(self.distraction_state.is_distracted),
            'distraction_level': int(self.distraction_state.distraction_level),
            'alert_triggered': bool(self.distraction_state.alert_triggered),
            'consecutive_distracted_frames': int(self.distraction_state.consecutive_frames),
            'total_distractions': int(self.distraction_state.total_distractions),
            'avg_attention': float(self.distraction_state.avg_attention),

            # Stats
            'recent_distraction_count': len(recent_distractions),
            'avg_recent_attention': float(np.mean(list(self.attention_history)[-60:])) if len(self.attention_history) >= 60 else float(self.gaze_metrics.attention_score),

            # timing
            'time_since_last_forward_gaze': float(current_time - self.last_forward_gaze_time)
        }

    def get_distraction_summary(self) -> Dict:
        """Get summarized distraction assessment"""
        current_time = time.time()

        if self.distraction_state.distraction_level >= 3:
            attention_state = "SEVERELY_DISTRACTED"
        elif self.distraction_state.distraction_level >= 2:
            attention_state = "DISTRACTED"
        elif self.distraction_state.distraction_level >= 1:
            attention_state = "MILDLY_DISTRACTED"
        elif self.gaze_metrics.attention_score > 0.8:
            attention_state = "FULLY_ATTENTIVE"
        else:
            attention_state = "SOMEWHAT_ATTENTIVE"

        recent_events = [e for e in self.distraction_events if current_time - e['start_time'] < 300]
        distraction_types = {}
        for event in recent_events:
            distraction_types[event['type']] = distraction_types.get(event['type'], 0) + 1

        return {
            'attention_state': attention_state,
            'distraction_level': self.distraction_state.distraction_level,
            'attention_score': self.gaze_metrics.attention_score,
            'current_distraction': (self.gaze_metrics.distraction_type if self.distraction_state.is_distracted else None),
            'distraction_duration': self.gaze_metrics.distraction_duration,
            'frequent_distractions': distraction_types,
            'time_since_attentive': float(current_time - self.last_forward_gaze_time),
            'recommendations': self._get_attention_recommendations()
        }

    def _get_attention_recommendations(self) -> List[str]:
        """Human-readable recommendations based on distraction level/type."""
        recommendations: List[str] = []
        if self.distraction_state.distraction_level >= 3:
            recommendations.extend([
                "FOCUS ON THE ROAD IMMEDIATELY",
                "Pull over if distraction continues",
                "Minimize in-cabin activities"
            ])
        elif self.distraction_state.distraction_level >= 2:
            recommendations.extend([
                "Return attention to the road",
                "Limit non-driving activities",
                "Check mirrors briefly only when safe"
            ])
        elif self.distraction_state.distraction_level >= 1:
            recommendations.extend([
                "Maintain forward focus",
                "Be aware of distraction tendency"
            ])

        if self.gaze_metrics.distraction_type == "phone":
            recommendations.append("Put phone away or use hands-free mode")
        elif self.gaze_metrics.distraction_type == "passenger":
            recommendations.append("Keep conversations brief while driving")
        elif self.gaze_metrics.distraction_type == "dashboard":
            recommendations.append("Use voice commands for navigation/controls")

        return recommendations

    def reset_session(self):
        """Reset tracker session and histories (keeps face_mesh/model loaded)."""
        self.head_pose = HeadPose()
        self.gaze_metrics = GazeMetrics()
        self.distraction_state = DistractionState()
        self.pose_history.clear()
        self.attention_history.clear()
        self.distraction_events.clear()
        self.baseline_pose = None
        self.calibration_frames = 0
        self.calibrated = False
        self.distraction_start_time = None
        self.last_forward_gaze_time = time.time()
        self.logger.info("Head pose tracking session reset")


# ---------- __main__ test harness with fallback CameraManager ----------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Try to import user CameraManager, otherwise fall back to simple capture
    try:
        from utils.camera_utils import CameraManager  # type: ignore

        camera_manager = CameraManager()
        has_camera_manager = True
    except Exception:
        has_camera_manager = False

        class CameraManagerSimple:
            """Simple fallback camera manager using OpenCV VideoCapture(0)"""
            def __init__(self, cam_index=0):
                self.cap = cv2.VideoCapture(cam_index)

            def initialize_cameras(self) -> bool:
                return self.cap.isOpened()

            def start_capture(self):
                return True

            def stop_capture(self):
                try:
                    self.cap.release()
                except Exception:
                    pass

            def get_frame(self, cam_name='driver_cam'):
                if not self.cap or not self.cap.isOpened():
                    return None
                ret, frame = self.cap.read()
                if not ret:
                    return None
                return frame

        camera_manager = CameraManagerSimple()

    tracker = HeadPoseTracker()

    if not camera_manager.initialize_cameras():
        print("Failed to initialize camera(s). Exiting.")
        sys.exit(1)

    camera_manager.start_capture()

    try:
        frame_count = 0
        calibration_phase = True
        print("Starting calibration phase - look straight ahead for ~2-3 seconds... (press 'q' to quit)")

        while True:
            frame = camera_manager.get_frame('driver_cam')
            if frame is None:
                time.sleep(0.01)
                continue

            # Calibration stage
            if calibration_phase and not tracker.calibrated:
                ok = tracker.calibrate_baseline(frame)
                if ok:
                    calibration_phase = False
                    print("Calibration completed! Now running head pose tracking.")
                # show the frame anyway so user sees camera
                cv2.imshow("Head Pose Tracking (calibrating)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue

            # Process frame normally
            out_frame, analysis = tracker.process_frame(frame)
            cv2.imshow("Head Pose Tracking", out_frame)

            frame_count += 1
            if frame_count % 30 == 0 and tracker.calibrated:
                print(f"[Frame {frame_count}] pitch={analysis['pitch']:.1f} yaw={analysis['yaw']:.1f} roll={analysis['roll']:.1f} "
                      f"att_score={analysis['attention_score']:.2f} distracted={analysis['is_distracted']} level={analysis['distraction_level']}")

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            if key == ord('r'):  # reset
                tracker.reset_session()
                calibration_phase = True
                print("Session reset - starting calibration again...")

    finally:
        try:
            camera_manager.stop_capture()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Head pose tracking test completed")
