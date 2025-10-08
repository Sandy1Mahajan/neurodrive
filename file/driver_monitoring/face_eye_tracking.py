"""
NeuroDrive Face and Eye Tracking Module
Advanced drowsiness detection using eye closure, yawning, and micro-sleep detection
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy.spatial import distance as dist
import yaml
import logging
from dataclasses import dataclass

@dataclass
class EyeMetrics:
    """Eye tracking metrics"""
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    blink_count: int = 0
    closure_duration: float = 0.0
    is_closed: bool = False

@dataclass  
class FaceMetrics:
    """Face detection metrics"""
    face_detected: bool = False
    face_confidence: float = 0.0
    yawn_detected: bool = False
    mouth_aspect_ratio: float = 0.0
    fatigue_score: float = 0.0

@dataclass
class DrowsinessState:
    """Overall drowsiness state"""
    is_drowsy: bool = False
    drowsiness_level: int = 0  # 0-4 scale
    alert_triggered: bool = False
    micro_sleep_detected: bool = False
    consecutive_frames: int = 0
    total_blinks: int = 0
    yawn_frequency: int = 0

class FaceEyeTracker:
    """Advanced face and eye tracking for drowsiness detection"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config['models']['face_detection']['min_detection_confidence'],
            min_tracking_confidence=self.config['models']['face_detection']['min_tracking_confidence']
        )
        
        # Eye landmark indices (MediaPipe face mesh)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouth landmark indices  
        self.MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 38]
        
        # Thresholds from config
        self.EAR_THRESHOLD = self.config['models']['eye_tracking']['ear_threshold']
        self.CONSECUTIVE_FRAMES = self.config['models']['eye_tracking']['consecutive_frames']
        self.YAWN_THRESHOLD = self.config['models']['eye_tracking']['yawn_threshold']
        
        # State tracking
        self.eye_metrics = EyeMetrics()
        self.face_metrics = FaceMetrics()
        self.drowsiness_state = DrowsinessState()
        
        # Time-based tracking
        self.blink_timestamps = deque(maxlen=50)
        self.yawn_timestamps = deque(maxlen=20)
        self.eye_closure_start = None
        self.last_blink_time = 0
        
        # Moving averages
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        
        # Threading
        self.processing_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame for face and eye tracking"""
        with self.processing_lock:
            start_time = time.time()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            output_frame = frame.copy()
            analysis_data = {}
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye and mouth metrics
                    self._extract_eye_metrics(face_landmarks)
                    self._extract_mouth_metrics(face_landmarks)
                    
                    # Update drowsiness state
                    self._update_drowsiness_state()
                    
                    # Draw landmarks and annotations
                    output_frame = self._draw_annotations(output_frame, face_landmarks)
                    
                    self.face_metrics.face_detected = True
                    self.face_metrics.face_confidence = 1.0
                    
                    break
            else:
                self.face_metrics.face_detected = False
                self.face_metrics.face_confidence = 0.0
                self._handle_no_face_detected()
            
            # Compile analysis data
            analysis_data = self._compile_analysis_data()
            
            # Add performance info
            processing_time = time.time() - start_time
            analysis_data['processing_time'] = processing_time
            
            return output_frame, analysis_data
    
    def _extract_eye_metrics(self, face_landmarks):
        """Extract eye aspect ratio and related metrics"""
        # Get landmark coordinates
        landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
        
        # Calculate EAR for both eyes
        left_eye = landmarks[self.LEFT_EYE_INDICES[:6]]  # Use key points only
        right_eye = landmarks[self.RIGHT_EYE_INDICES[:6]]
        
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        self.eye_metrics.left_ear = left_ear
        self.eye_metrics.right_ear = right_ear
        self.eye_metrics.avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to history
        self.ear_history.append(self.eye_metrics.avg_ear)
        
        # Detect blinks and eye closure
        self._detect_blinks_and_closure()
    
    def _calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_points) < 6:
            return 0.0
        
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # EAR formula
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _extract_mouth_metrics(self, face_landmarks):
        """Extract mouth aspect ratio for yawn detection"""
        landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
        
        # Get key mouth points
        mouth_points = landmarks[[78, 81, 13, 311, 308, 415, 324, 318]]
        
        if len(mouth_points) >= 8:
            # Calculate mouth aspect ratio (MAR)
            # Vertical distances
            A = dist.euclidean(mouth_points[2], mouth_points[6])  # Top to bottom
            B = dist.euclidean(mouth_points[3], mouth_points[7])  # Another vertical
            
            # Horizontal distance
            C = dist.euclidean(mouth_points[0], mouth_points[4])  # Left to right
            
            if C > 0:
                mar = (A + B) / (2.0 * C)
                self.face_metrics.mouth_aspect_ratio = mar
                self.mar_history.append(mar)
                
                # Detect yawn
                self._detect_yawn(mar)
    
    def _detect_blinks_and_closure(self):
        """Detect blinks and prolonged eye closure"""
        current_time = time.time()
        avg_ear = self.eye_metrics.avg_ear
        
        # Check if eyes are closed
        if avg_ear < self.EAR_THRESHOLD:
            self.eye_metrics.is_closed = True
            
            if self.eye_closure_start is None:
                self.eye_closure_start = current_time
            
            self.eye_metrics.closure_duration = current_time - self.eye_closure_start
            self.drowsiness_state.consecutive_frames += 1
            
            # Micro-sleep detection (eyes closed > 0.5 seconds)
            if self.eye_metrics.closure_duration > 0.5:
                self.drowsiness_state.micro_sleep_detected = True
        
        else:
            # Eyes are open
            if self.eye_metrics.is_closed:  # Blink detected
                if self.eye_closure_start and current_time - self.eye_closure_start > 0.1:
                    # Valid blink (> 100ms)
                    self.eye_metrics.blink_count += 1
                    self.drowsiness_state.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                    self.last_blink_time = current_time
            
            self.eye_metrics.is_closed = False
            self.eye_closure_start = None
            self.eye_metrics.closure_duration = 0.0
            self.drowsiness_state.consecutive_frames = 0
            self.drowsiness_state.micro_sleep_detected = False
    
    def _detect_yawn(self, mar: float):
        """Detect yawning based on mouth aspect ratio"""
        current_time = time.time()
        
        if mar > self.YAWN_THRESHOLD:
            if not self.face_metrics.yawn_detected:
                self.face_metrics.yawn_detected = True
                self.yawn_timestamps.append(current_time)
                
                # Calculate yawn frequency (per minute)
                recent_yawns = [t for t in self.yawn_timestamps if current_time - t < 60]
                self.drowsiness_state.yawn_frequency = len(recent_yawns)
        else:
            self.face_metrics.yawn_detected = False
    
    def _update_drowsiness_state(self):
        """Update overall drowsiness state based on all metrics"""
        current_time = time.time()
        
        # Calculate drowsiness factors
        factors = {
            'eye_closure': 0,
            'blink_frequency': 0, 
            'yawn_frequency': 0,
            'micro_sleep': 0,
            'ear_trend': 0
        }
        
        # Eye closure factor
        if self.eye_metrics.closure_duration > 2.0:
            factors['eye_closure'] = 4  # Critical
        elif self.eye_metrics.closure_duration > 1.0:
            factors['eye_closure'] = 3  # High
        elif self.eye_metrics.closure_duration > 0.5:
            factors['eye_closure'] = 2  # Medium
        elif self.drowsiness_state.consecutive_frames > self.CONSECUTIVE_FRAMES:
            factors['eye_closure'] = 1  # Low
        
        # Blink frequency factor (normal: 15-20 blinks/minute)
        recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60]
        blink_rate = len(recent_blinks)
        
        if blink_rate < 5:  # Too few blinks (microsleep)
            factors['blink_frequency'] = 3
        elif blink_rate > 30:  # Too many blinks (fatigue)
            factors['blink_frequency'] = 2
        elif blink_rate < 10 or blink_rate > 25:
            factors['blink_frequency'] = 1
        
        # Yawn frequency factor
        if self.drowsiness_state.yawn_frequency > 5:
            factors['yawn_frequency'] = 3
        elif self.drowsiness_state.yawn_frequency > 3:
            factors['yawn_frequency'] = 2
        elif self.drowsiness_state.yawn_frequency > 1:
            factors['yawn_frequency'] = 1
        
        # Micro-sleep factor
        if self.drowsiness_state.micro_sleep_detected:
            factors['micro_sleep'] = 4
        
        # EAR trend factor (declining trend indicates fatigue)
        if len(self.ear_history) >= 20:
            recent_ear = np.mean(list(self.ear_history)[-10:])
            older_ear = np.mean(list(self.ear_history)[-20:-10])
            
            if recent_ear < older_ear * 0.9:  # 10% decrease
                factors['ear_trend'] = 2
            elif recent_ear < older_ear * 0.95:  # 5% decrease
                factors['ear_trend'] = 1
        
        # Calculate overall drowsiness level
        max_factor = max(factors.values())
        avg_factor = np.mean(list(factors.values()))
        
        self.drowsiness_state.drowsiness_level = min(4, int(max_factor))
        
        # Determine if drowsy
        self.drowsiness_state.is_drowsy = max_factor >= 2
        
        # Calculate fatigue score (0-1)
        self.face_metrics.fatigue_score = min(1.0, avg_factor / 4.0)
        
        # Trigger alert if needed
        if max_factor >= 3 and not self.drowsiness_state.alert_triggered:
            self.drowsiness_state.alert_triggered = True
        elif max_factor < 2:
            self.drowsiness_state.alert_triggered = False
    
    def _handle_no_face_detected(self):
        """Handle case when no face is detected"""
        # Reset metrics but maintain some state
        self.eye_metrics = EyeMetrics()
        self.face_metrics.face_detected = False
        self.face_metrics.yawn_detected = False
        
        # Don't reset drowsiness state immediately - could be temporary
        self.drowsiness_state.consecutive_frames += 1
        
        # If no face for too long, reset drowsiness state
        if self.drowsiness_state.consecutive_frames > 60:  # ~2 seconds at 30fps
            self.drowsiness_state = DrowsinessState()
    
    def _draw_annotations(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw tracking annotations on frame"""
        h, w = frame.shape[:2]
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        
        # Draw eye contours
        left_eye_points = landmarks[self.LEFT_EYE_INDICES[:6]].astype(np.int32)
        right_eye_points = landmarks[self.RIGHT_EYE_INDICES[:6]].astype(np.int32)
        
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
        
        # Draw mouth contour
        mouth_points = landmarks[[78, 81, 13, 311, 308, 415, 324, 318]].astype(np.int32)
        cv2.polylines(frame, [mouth_points], True, (255, 0, 0), 1)
        
        # Status indicators
        y_offset = 30
        
        # EAR value
        cv2.putText(frame, f"EAR: {self.eye_metrics.avg_ear:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        
        # Blink count
        cv2.putText(frame, f"Blinks: {self.eye_metrics.blink_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        
        # Drowsiness level
        color = (0, 255, 0)  # Green
        if self.drowsiness_state.drowsiness_level >= 3:
            color = (0, 0, 255)  # Red
        elif self.drowsiness_state.drowsiness_level >= 2:
            color = (0, 165, 255)  # Orange
        elif self.drowsiness_state.drowsiness_level >= 1:
            color = (0, 255, 255)  # Yellow
        
        cv2.putText(frame, f"Drowsiness: {self.drowsiness_state.drowsiness_level}/4", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 25
        
        # Fatigue score
        cv2.putText(frame, f"Fatigue: {self.face_metrics.fatigue_score:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        
        # Alert indicators
        if self.drowsiness_state.alert_triggered:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        if self.drowsiness_state.micro_sleep_detected:
            cv2.putText(frame, "MICRO-SLEEP DETECTED!", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        if self.face_metrics.yawn_detected:
            cv2.putText(frame, "YAWNING", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def _compile_analysis_data(self) -> Dict:
        """Compile all analysis data into a dictionary"""
        current_time = time.time()
        
        # Calculate blink rate (per minute)
        recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60]
        blink_rate = len(recent_blinks)
        
        return {
            'timestamp': current_time,
            'face_detected': self.face_metrics.face_detected,
            'face_confidence': self.face_metrics.face_confidence,
            
            # Eye metrics
            'left_ear': self.eye_metrics.left_ear,
            'right_ear': self.eye_metrics.right_ear,
            'avg_ear': self.eye_metrics.avg_ear,
            'eyes_closed': self.eye_metrics.is_closed,
            'eye_closure_duration': self.eye_metrics.closure_duration,
            'blink_count': self.eye_metrics.blink_count,
            'blink_rate_per_minute': blink_rate,
            
            # Mouth metrics
            'mouth_aspect_ratio': self.face_metrics.mouth_aspect_ratio,
            'yawn_detected': self.face_metrics.yawn_detected,
            'yawn_frequency': self.drowsiness_state.yawn_frequency,
            
            # Drowsiness metrics
            'is_drowsy': self.drowsiness_state.is_drowsy,
            'drowsiness_level': self.drowsiness_state.drowsiness_level,
            'fatigue_score': self.face_metrics.fatigue_score,
            'alert_triggered': self.drowsiness_state.alert_triggered,
            'micro_sleep_detected': self.drowsiness_state.micro_sleep_detected,
            'consecutive_closed_frames': self.drowsiness_state.consecutive_frames,
            
            # Trends
            'ear_trend': np.mean(list(self.ear_history)[-10:]) if len(self.ear_history) >= 10 else 0,
            'mar_trend': np.mean(list(self.mar_history)[-10:]) if len(self.mar_history) >= 10 else 0
        }
    
    def get_drowsiness_summary(self) -> Dict:
        """Get summarized drowsiness assessment"""
        return {
            'overall_state': 'ALERT' if not self.drowsiness_state.is_drowsy else 
                           'DROWSY' if self.drowsiness_state.drowsiness_level < 3 else 'CRITICAL',
            'level': self.drowsiness_state.drowsiness_level,
            'fatigue_score': self.face_metrics.fatigue_score,
            'primary_indicators': self._get_primary_indicators(),
            'recommendations': self._get_recommendations()
        }
    
    def _get_primary_indicators(self) -> List[str]:
        """Get list of primary drowsiness indicators"""
        indicators = []
        
        if self.drowsiness_state.micro_sleep_detected:
            indicators.append("Micro-sleep detected")
        
        if self.eye_metrics.closure_duration > 1.0:
            indicators.append(f"Prolonged eye closure ({self.eye_metrics.closure_duration:.1f}s)")
        
        if self.drowsiness_state.yawn_frequency > 3:
            indicators.append(f"Frequent yawning ({self.drowsiness_state.yawn_frequency}/min)")
        
        current_time = time.time()
        recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60]
        blink_rate = len(recent_blinks)
        
        if blink_rate < 10:
            indicators.append(f"Low blink rate ({blink_rate}/min)")
        elif blink_rate > 25:
            indicators.append(f"High blink rate ({blink_rate}/min)")
        
        if self.eye_metrics.avg_ear < self.EAR_THRESHOLD * 1.2:
            indicators.append("Low eye aspect ratio")
        
        return indicators
    
    def _get_recommendations(self) -> List[str]:
        """Get safety recommendations based on current state"""
        recommendations = []
        
        if self.drowsiness_state.drowsiness_level >= 3:
            recommendations.extend([
                "PULL OVER IMMEDIATELY",
                "Take a 15-20 minute power nap",
                "Do not continue driving"
            ])
        elif self.drowsiness_state.drowsiness_level >= 2:
            recommendations.extend([
                "Find a safe place to rest soon",
                "Increase cabin ventilation",
                "Consider caffeinated beverage"
            ])
        elif self.drowsiness_state.drowsiness_level >= 1:
            recommendations.extend([
                "Stay alert and monitor condition",
                "Ensure adequate sleep before next trip"
            ])
        
        return recommendations
    
    def reset_session(self):
        """Reset tracking for new session"""
        self.eye_metrics = EyeMetrics()
        self.face_metrics = FaceMetrics()
        self.drowsiness_state = DrowsinessState()
        
        self.blink_timestamps.clear()
        self.yawn_timestamps.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        
        self.eye_closure_start = None
        self.last_blink_time = 0
        
        self.logger.info("Face/Eye tracking session reset")


if __name__ == "__main__":
    # Test face/eye tracker
    import sys
    sys.path.append('..')
    from utils.camera_utils import CameraManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    tracker = FaceEyeTracker()
    camera_manager = CameraManager()
    
    if not camera_manager.initialize_cameras():
        print("Failed to initialize cameras")
        exit(1)
    
    camera_manager.start_capture()
    
    try:
        frame_count = 0
        while True:
            # Get frame from driver camera
            frame = camera_manager.get_frame('driver_cam')
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame
            output_frame, analysis = tracker.process_frame(frame)
            
            # Display results
            cv2.imshow('Face/Eye Tracking', output_frame)
            
            # Print analysis every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}:")
                print(f"  Drowsiness Level: {analysis['drowsiness_level']}/4")
                print(f"  EAR: {analysis['avg_ear']:.3f}")
                print(f"  Fatigue Score: {analysis['fatigue_score']:.2f}")
                print(f"  Blink Rate: {analysis['blink_rate_per_minute']}/min")
                
                if analysis['alert_triggered']:
                    print("  *** DROWSINESS ALERT ***")
                
                summary = tracker.get_drowsiness_summary()
                if summary['primary_indicators']:
                    print(f"  Indicators: {', '.join(summary['primary_indicators'])}")
                
                print()
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera_manager.stop_capture()
        cv2.destroyAllWindows()
        print("Face/Eye tracking test completed")