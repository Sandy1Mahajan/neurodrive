"""
NeuroDrive Hand Gesture Detection Module
Advanced gesture recognition for driver interactions and safety signals
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from enum import Enum
import yaml
import logging
from dataclasses import dataclass

class GestureType(Enum):
    """Supported gesture types"""
    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down" 
    PEACE = "peace"
    OK = "ok"
    STOP = "stop"
    POINTING = "pointing"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    SOS = "sos"
    PHONE_GESTURE = "phone_gesture"
    SMOKING_GESTURE = "smoking_gesture"
    EATING_GESTURE = "eating_gesture"
    DRINKING_GESTURE = "drinking_gesture"
    WAVING = "waving"

@dataclass
class HandData:
    """Hand detection data"""
    landmarks: np.ndarray
    handedness: str  # "Left" or "Right"
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]

@dataclass
class GestureDetection:
    """Gesture detection result"""
    gesture_type: GestureType
    confidence: float
    hand_data: HandData
    duration: float = 0.0
    timestamp: float = 0.0
    description: str = ""

@dataclass
class GestureState:
    """Gesture tracking state"""
    current_gestures: List[GestureDetection]
    gesture_history: deque
    total_gestures: int = 0
    sos_count: int = 0
    distraction_gestures: int = 0
    last_gesture_time: float = 0.0

class HandGestureDetector:
    """Advanced hand gesture detection and recognition"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        gesture_config = self.config.get('models', {}).get('gesture_detection', {})
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=gesture_config.get('max_hands', 2),
            min_detection_confidence=gesture_config.get('confidence', 0.7),
            min_tracking_confidence=0.5
        )
        
        # Gesture detection thresholds
        self.confidence_threshold = 0.7
        self.gesture_stability_frames = 5
        self.sos_pattern_window = 10.0  # seconds
        
        # State tracking
        self.gesture_state = GestureState(
            current_gestures=[],
            gesture_history=deque(maxlen=100)
        )
        
        # Gesture pattern buffers
        self.gesture_buffer = deque(maxlen=30)  # 1 second at 30fps
        self.sos_pattern_buffer = deque(maxlen=300)  # 10 seconds for SOS pattern
        
        # Timing
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Gesture callbacks
        self.gesture_callbacks: Dict[GestureType, List[Callable]] = {}
        
        # Threading
        self.processing_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame for hand gesture detection"""
        with self.processing_lock:
            start_time = time.time()
            self.frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            output_frame = frame.copy()
            self.gesture_state.current_gestures.clear()
            
            if results.multi_hand_landmarks:
                # Process each detected hand
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Extract hand data
                    hand_data = self._extract_hand_data(hand_landmarks, handedness, frame.shape)
                    
                    # Recognize gesture
                    gesture = self._recognize_gesture(hand_data)
                    
                    if gesture:
                        self.gesture_state.current_gestures.append(gesture)
                        
                        # Add to history
                        self.gesture_state.gesture_history.append(gesture)
                        self.gesture_state.total_gestures += 1
                        self.gesture_state.last_gesture_time = start_time
                        
                        # Check for special patterns
                        self._check_sos_pattern(gesture)
                        self._check_distraction_patterns(gesture)
                        
                        # Trigger callbacks
                        self._trigger_gesture_callbacks(gesture)
                    
                    # Draw hand landmarks and gesture
                    output_frame = self._draw_hand_annotations(
                        output_frame, hand_landmarks, hand_data, gesture
                    )
            
            # Update gesture buffer
            self.gesture_buffer.append({
                'timestamp': start_time,
                'gestures': self.gesture_state.current_gestures.copy()
            })
            
            # Compile analysis data
            analysis_data = self._compile_analysis_data()
            
            # Add performance info
            processing_time = time.time() - start_time
            analysis_data['processing_time'] = processing_time
            
            return output_frame, analysis_data
    
    def _extract_hand_data(self, hand_landmarks, handedness, frame_shape) -> HandData:
        """Extract hand data from MediaPipe landmarks"""
        h, w = frame_shape[:2]
        
        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        
        # Calculate bounding box
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
        
        # Get handedness
        hand_label = handedness.classification[0].label  # "Left" or "Right"
        hand_confidence = handedness.classification[0].score
        
        return HandData(
            landmarks=landmarks,
            handedness=hand_label,
            confidence=hand_confidence,
            bbox=bbox,
            center=center
        )
    
    def _recognize_gesture(self, hand_data: HandData) -> Optional[GestureDetection]:
        """Recognize gesture from hand landmarks"""
        landmarks = hand_data.landmarks
        
        if len(landmarks) != 21:  # MediaPipe hand has 21 landmarks
            return None
        
        # Calculate finger positions and angles
        finger_states = self._analyze_finger_states(landmarks)
        hand_orientation = self._calculate_hand_orientation(landmarks)
        finger_distances = self._calculate_finger_distances(landmarks)
        
        # Gesture recognition logic
        gesture_type, confidence, description = self._classify_gesture(
            finger_states, hand_orientation, finger_distances, hand_data
        )
        
        if confidence > self.confidence_threshold:
            return GestureDetection(
                gesture_type=gesture_type,
                confidence=confidence,
                hand_data=hand_data,
                timestamp=time.time(),
                description=description
            )
        
        return None
    
    def _analyze_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """Analyze if fingers are extended or folded"""
        # MediaPipe hand landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        finger_mcps = [2, 5, 9, 13, 17]   # MCP joints
        
        finger_states = {}
        
        # Thumb (special case - compare x-coordinate)
        thumb_extended = abs(landmarks[4][0] - landmarks[3][0]) > abs(landmarks[3][0] - landmarks[2][0])
        finger_states['thumb'] = thumb_extended
        
        # Other fingers (compare y-coordinates)
        fingers = ['index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(fingers):
            tip_idx = finger_tips[i + 1]  # Skip thumb
            pip_idx = finger_pips[i + 1]
            
            # Finger is extended if tip is higher than PIP joint
            extended = landmarks[tip_idx][1] < landmarks[pip_idx][1]
            finger_states[finger] = extended
        
        return finger_states
    
    def _calculate_hand_orientation(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate hand orientation and palm direction"""
        # Wrist to middle finger MCP
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate angle
        direction_vector = middle_mcp - wrist
        angle = math.atan2(direction_vector[1], direction_vector[0])
        angle_degrees = math.degrees(angle)
        
        # Palm center (approximate)
        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)
        
        return {
            'angle': angle_degrees,
            'palm_center': palm_center,
            'direction_vector': direction_vector
        }
    
    def _calculate_finger_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate distances between key finger points"""
        distances = {}
        
        # Thumb to index distance
        distances['thumb_index'] = np.linalg.norm(landmarks[4] - landmarks[8])
        
        # Index to middle distance
        distances['index_middle'] = np.linalg.norm(landmarks[8] - landmarks[12])
        
        # Middle to ring distance
        distances['middle_ring'] = np.linalg.norm(landmarks[12] - landmarks[16])
        
        # Ring to pinky distance
        distances['ring_pinky'] = np.linalg.norm(landmarks[16] - landmarks[20])
        
        # Palm width (approximate)
        distances['palm_width'] = np.linalg.norm(landmarks[5] - landmarks[17])
        
        return distances
    
    def _classify_gesture(self, finger_states: Dict[str, bool], 
                         hand_orientation: Dict[str, float], 
                         finger_distances: Dict[str, float],
                         hand_data: HandData) -> Tuple[GestureType, float, str]:
        """Classify gesture based on finger states and hand features"""
        
        # Count extended fingers
        extended_count = sum(finger_states.values())
        extended_fingers = [name for name, extended in finger_states.items() if extended]
        
        confidence = 0.0
        gesture_type = GestureType.NONE
        description = ""
        
        # Gesture classification logic
        
        # Fist - no fingers extended
        if extended_count == 0:
            gesture_type = GestureType.FIST
            confidence = 0.9
            description = "Closed fist"
        
        # Thumbs up - only thumb extended
        elif extended_count == 1 and finger_states['thumb']:
            gesture_type = GestureType.THUMBS_UP
            confidence = 0.9
            description = "Thumbs up"
        
        # Pointing - only index finger extended
        elif extended_count == 1 and finger_states['index']:
            gesture_type = GestureType.POINTING
            confidence = 0.85
            description = "Pointing gesture"
        
        # Peace sign - index and middle fingers extended
        elif (extended_count == 2 and 
              finger_states['index'] and finger_states['middle'] and
              finger_distances['index_middle'] > finger_distances['palm_width'] * 0.3):
            gesture_type = GestureType.PEACE
            confidence = 0.85
            description = "Peace sign"
        
        # OK sign - thumb and index form circle
        elif (finger_states['thumb'] and finger_states['index'] and
              finger_distances['thumb_index'] < finger_distances['palm_width'] * 0.4):
            gesture_type = GestureType.OK
            confidence = 0.8
            description = "OK sign"
        
        # Stop - all fingers extended, palm facing forward
        elif extended_count == 5:
            # Check if palm is facing camera (simplified check)
            if abs(hand_orientation['angle']) < 45 or abs(hand_orientation['angle']) > 135:
                gesture_type = GestureType.STOP
                confidence = 0.8
                description = "Stop/Open palm"
            else:
                gesture_type = GestureType.OPEN_PALM
                confidence = 0.7
                description = "Open palm"
        
        # Phone gesture - thumb and pinky extended (like "call me")
        elif (extended_count == 2 and 
              finger_states['thumb'] and finger_states['pinky']):
            gesture_type = GestureType.PHONE_GESTURE
            confidence = 0.75
            description = "Phone call gesture"
        
        # Waving - detect motion pattern if available
        elif extended_count >= 4:
            # Check for waving motion in recent history
            if self._detect_waving_motion(hand_data):
                gesture_type = GestureType.WAVING
                confidence = 0.8
                description = "Waving gesture"
            else:
                gesture_type = GestureType.OPEN_PALM
                confidence = 0.6
                description = "Open palm"
        
        # Smoking gesture - index and middle extended, others folded
        elif (extended_count == 2 and 
              finger_states['index'] and finger_states['middle'] and
              not finger_states['ring'] and not finger_states['pinky']):
            # Additional check: fingers close together
            if finger_distances['index_middle'] < finger_distances['palm_width'] * 0.2:
                gesture_type = GestureType.SMOKING_GESTURE
                confidence = 0.7
                description = "Potential smoking gesture"
        
        # Drinking gesture - check hand position and orientation
        elif self._detect_drinking_gesture(hand_data, finger_states):
            gesture_type = GestureType.DRINKING_GESTURE
            confidence = 0.65
            description = "Drinking gesture"
        
        # Eating gesture - similar to drinking but different hand position
        elif self._detect_eating_gesture(hand_data, finger_states):
            gesture_type = GestureType.EATING_GESTURE
            confidence = 0.6
            description = "Eating gesture"
        
        else:
            # Default classification
            gesture_type = GestureType.NONE
            confidence = 0.0
            description = f"Unrecognized gesture ({extended_count} fingers extended)"
        
        return gesture_type, confidence, description
    
    def _detect_waving_motion(self, hand_data: HandData) -> bool:
        """Detect waving motion from recent hand positions"""
        if len(self.gesture_buffer) < 10:
            return False
        
        # Get recent hand centers
        recent_centers = []
        for frame_data in list(self.gesture_buffer)[-10:]:
            for gesture in frame_data['gestures']:
                if gesture.hand_data.handedness == hand_data.handedness:
                    recent_centers.append(gesture.hand_data.center)
                    break
        
        if len(recent_centers) < 5:
            return False
        
        # Check for oscillating motion
        x_positions = [center[0] for center in recent_centers]
        
        # Simple oscillation detection
        direction_changes = 0
        for i in range(2, len(x_positions)):
            prev_diff = x_positions[i-1] - x_positions[i-2]
            curr_diff = x_positions[i] - x_positions[i-1]
            
            if prev_diff * curr_diff < 0:  # Sign change
                direction_changes += 1
        
        return direction_changes >= 2
    
    def _detect_drinking_gesture(self, hand_data: HandData, finger_states: Dict[str, bool]) -> bool:
        """Detect drinking gesture based on hand position and orientation"""
        # Check if hand is near mouth level (upper part of frame)
        frame_height = 480  # Assumed frame height
        hand_y = hand_data.center[1]
        
        # Hand should be in upper portion of frame
        if hand_y > frame_height * 0.6:
            return False
        
        # Check finger configuration (partial fist, like holding a cup)
        extended_count = sum(finger_states.values())
        
        # Typical drinking: thumb extended, 1-2 other fingers partially extended
        return (finger_states['thumb'] and 
                1 <= extended_count <= 3 and
                hand_data.handedness == "Right")  # Assuming right-hand drinking
    
    def _detect_eating_gesture(self, hand_data: HandData, finger_states: Dict[str, bool]) -> bool:
        """Detect eating gesture"""
        # Similar to drinking but different position and finger configuration
        frame_height = 480
        hand_y = hand_data.center[1]
        
        # Hand near mouth level
        if hand_y > frame_height * 0.7:
            return False
        
        # Eating typically involves pinched fingers or partial fist
        extended_count = sum(finger_states.values())
        
        return (extended_count <= 2 and 
                (finger_states['thumb'] or finger_states['index']) and
                hand_data.center[0] < 400)  # Approximate mouth region
    
    def _check_sos_pattern(self, gesture: GestureDetection):
        """Check for SOS pattern (3 short, 3 long, 3 short gestures)"""
        if gesture.gesture_type in [GestureType.FIST, GestureType.STOP, GestureType.POINTING]:
            current_time = time.time()
            
            # Add to SOS pattern buffer
            self.sos_pattern_buffer.append({
                'timestamp': current_time,
                'gesture': gesture.gesture_type,
                'duration': gesture.duration
            })
            
            # Check for SOS pattern in recent gestures
            recent_gestures = [g for g in self.sos_pattern_buffer 
                             if current_time - g['timestamp'] < self.sos_pattern_window]
            
            if self._is_sos_pattern(recent_gestures):
                self.gesture_state.sos_count += 1
                
                # Create SOS gesture detection
                sos_gesture = GestureDetection(
                    gesture_type=GestureType.SOS,
                    confidence=0.9,
                    hand_data=gesture.hand_data,
                    timestamp=current_time,
                    description="SOS distress signal detected"
                )
                
                self.gesture_state.current_gestures.append(sos_gesture)
                self._trigger_gesture_callbacks(sos_gesture)
    
    def _is_sos_pattern(self, gestures: List[Dict]) -> bool:
        """Check if gesture sequence matches SOS pattern"""
        if len(gestures) < 9:  # Minimum for SOS pattern
            return False
        
        # Simplified SOS pattern detection
        # Look for alternating gesture types in groups of 3
        gesture_types = [g['gesture'] for g in gestures[-9:]]
        
        # Basic pattern matching (can be enhanced)
        pattern_matches = 0
        for i in range(0, 9, 3):
            if i + 2 < len(gesture_types):
                group = gesture_types[i:i+3]
                if len(set(group)) == 1:  # Same gesture repeated 3 times
                    pattern_matches += 1
        
        return pattern_matches >= 2  # At least 2 consistent groups
    
    def _check_distraction_patterns(self, gesture: GestureDetection):
        """Check for patterns indicating driver distraction"""
        distraction_gestures = [
            GestureType.PHONE_GESTURE,
            GestureType.SMOKING_GESTURE,
            GestureType.EATING_GESTURE,
            GestureType.DRINKING_GESTURE
        ]
        
        if gesture.gesture_type in distraction_gestures:
            self.gesture_state.distraction_gestures += 1
    
    def _trigger_gesture_callbacks(self, gesture: GestureDetection):
        """Trigger registered callbacks for gesture"""
        callbacks = self.gesture_callbacks.get(gesture.gesture_type, [])
        
        for callback in callbacks:
            try:
                callback(gesture)
            except Exception as e:
                self.logger.error(f"Gesture callback error: {e}")
    
    def _draw_hand_annotations(self, frame: np.ndarray, hand_landmarks, 
                             hand_data: HandData, gesture: Optional[GestureDetection]) -> np.ndarray:
        """Draw hand landmarks and gesture annotations"""
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Draw bounding box
        x, y, w, h = hand_data.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw handedness label
        cv2.putText(frame, hand_data.handedness, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw gesture information
        if gesture and gesture.gesture_type != GestureType.NONE:
            # Gesture label
            gesture_text = f"{gesture.gesture_type.value} ({gesture.confidence:.2f})"
            
            # Color based on gesture type
            if gesture.gesture_type == GestureType.SOS:
                color = (0, 0, 255)  # Red for SOS
            elif gesture.gesture_type in [GestureType.PHONE_GESTURE, GestureType.SMOKING_GESTURE, 
                                        GestureType.EATING_GESTURE, GestureType.DRINKING_GESTURE]:
                color = (0, 165, 255)  # Orange for distraction gestures
            else:
                color = (255, 255, 0)  # Cyan for other gestures
            
            cv2.putText(frame, gesture_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Description
            if gesture.description:
                cv2.putText(frame, gesture.description, (x, y + h + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _compile_analysis_data(self) -> Dict:
        """Compile all analysis data into a dictionary"""
        current_time = time.time()
        
        # Count gesture types in recent history
        recent_gestures = [g for g in self.gesture_state.gesture_history 
                          if current_time - g.timestamp < 60]  # Last minute
        
        gesture_counts = {}
        for gesture in recent_gestures:
            gesture_type = gesture.gesture_type.value
            gesture_counts[gesture_type] = gesture_counts.get(gesture_type, 0) + 1
        
        # Analyze current gestures
        current_gesture_types = [g.gesture_type.value for g in self.gesture_state.current_gestures]
        max_confidence = max([g.confidence for g in self.gesture_state.current_gestures], default=0.0)
        
        return {
            'timestamp': current_time,
            'frame_count': self.frame_count,
            
            # Current detection
            'hands_detected': len(self.gesture_state.current_gestures),
            'current_gestures': current_gesture_types,
            'max_confidence': max_confidence,
            'gestures_detail': [
                {
                    'type': g.gesture_type.value,
                    'confidence': g.confidence,
                    'handedness': g.hand_data.handedness,
                    'description': g.description
                } for g in self.gesture_state.current_gestures
            ],
            
            # Statistics
            'total_gestures': self.gesture_state.total_gestures,
            'sos_count': self.gesture_state.sos_count,
            'distraction_gestures': self.gesture_state.distraction_gestures,
            'recent_gesture_counts': gesture_counts,
            'last_gesture_time': self.gesture_state.last_gesture_time,
            
            # Alerts
            'sos_detected': any(g.gesture_type == GestureType.SOS for g in self.gesture_state.current_gestures),
            'distraction_detected': any(g.gesture_type in [
                GestureType.PHONE_GESTURE, GestureType.SMOKING_GESTURE,
                GestureType.EATING_GESTURE, GestureType.DRINKING_GESTURE
            ] for g in self.gesture_state.current_gestures),
            
            # Timing
            'time_since_last_gesture': current_time - self.gesture_state.last_gesture_time if self.gesture_state.last_gesture_time > 0 else 0
        }
    
    def get_gesture_summary(self) -> Dict:
        """Get summarized gesture analysis"""
        current_time = time.time()
        
        # Analyze gesture patterns
        recent_gestures = [g for g in self.gesture_state.gesture_history 
                          if current_time - g.timestamp < 300]  # Last 5 minutes
        
        # Count distraction gestures
        distraction_count = sum(1 for g in recent_gestures 
                              if g.gesture_type in [GestureType.PHONE_GESTURE, GestureType.SMOKING_GESTURE,
                                                  GestureType.EATING_GESTURE, GestureType.DRINKING_GESTURE])
        
        # Determine alert level
        if any(g.gesture_type == GestureType.SOS for g in self.gesture_state.current_gestures):
            alert_level = "EMERGENCY"
        elif distraction_count > 5:
            alert_level = "HIGH_DISTRACTION"
        elif distraction_count > 2:
            alert_level = "MODERATE_DISTRACTION"
        elif any(g.gesture_type in [GestureType.PHONE_GESTURE, GestureType.SMOKING_GESTURE] 
                for g in self.gesture_state.current_gestures):
            alert_level = "CURRENT_DISTRACTION"
        else:
            alert_level = "NORMAL"
        
        # Most frequent gestures
        gesture_frequency = {}
        for gesture in recent_gestures:
            gesture_type = gesture.gesture_type.value
            gesture_frequency[gesture_type] = gesture_frequency.get(gesture_type, 0) + 1
        
        return {
            'alert_level': alert_level,
            'current_gestures': [g.gesture_type.value for g in self.gesture_state.current_gestures],
            'sos_count': self.gesture_state.sos_count,
            'recent_distraction_count': distraction_count,
            'total_gestures': self.gesture_state.total_gestures,
            'frequent_gestures': dict(sorted(gesture_frequency.items(), key=lambda x: x[1], reverse=True)[:5]),
            'time_since_last_gesture': current_time - self.gesture_state.last_gesture_time if self.gesture_state.last_gesture_time > 0 else 0,
            'recommendations': self._get_gesture_recommendations()
        }
    
    def _get_gesture_recommendations(self) -> List[str]:
        """Get recommendations based on detected gestures"""
        recommendations = []
        
        current_gesture_types = [g.gesture_type for g in self.gesture_state.current_gestures]
        
        if GestureType.SOS in current_gesture_types:
            recommendations.extend([
                "EMERGENCY SOS SIGNAL DETECTED",
                "Contact emergency services immediately",
                "Check driver status and safety"
            ])
        
        elif GestureType.PHONE_GESTURE in current_gesture_types:
            recommendations.extend([
                "Phone use detected while driving",
                "Use hands-free mode for calls",
                "Pull over safely if call is urgent"
            ])
        
        elif GestureType.SMOKING_GESTURE in current_gesture_types:
            recommendations.extend([
                "Smoking detected while driving",
                "Smoking impairs driving ability",
                "Consider stopping in a safe location"
            ])
        
        elif GestureType.EATING_GESTURE in current_gesture_types:
            recommendations.extend([
                "Eating while driving detected",
                "Keep both hands on steering wheel",
                "Eat before or after driving"
            ])
        
        elif GestureType.DRINKING_GESTURE in current_gesture_types:
            recommendations.extend([
                "Drinking while driving detected",
                "Use hands-free drinking systems",
                "Stay hydrated before driving"
            ])
        
        elif self.gesture_state.distraction_gestures > 10:
            recommendations.extend([
                "Frequent hand gestures detected",
                "Minimize non-driving hand activities",
                "Focus on road and steering control"
            ])
        
        return recommendations
    
    def add_gesture_callback(self, gesture_type: GestureType, callback: Callable):
        """Add callback for specific gesture type"""
        if gesture_type not in self.gesture_callbacks:
            self.gesture_callbacks[gesture_type] = []
        
        self.gesture_callbacks[gesture_type].append(callback)
    
    def remove_gesture_callback(self, gesture_type: GestureType, callback: Callable):
        """Remove callback for gesture type"""
        if gesture_type in self.gesture_callbacks and callback in self.gesture_callbacks[gesture_type]:
            self.gesture_callbacks[gesture_type].remove(callback)
    
    def reset_session(self):
        """Reset gesture tracking for new session"""
        self.gesture_state = GestureState(
            current_gestures=[],
            gesture_history=deque(maxlen=100)
        )
        
        self.gesture_buffer.clear()
        self.sos_pattern_buffer.clear()
        
        self.frame_count = 0
        self.last_process_time = time.time()
        
        self.logger.info("Gesture detection session reset")
    
    def get_gesture_statistics(self) -> Dict:
        """Get detailed gesture statistics"""
        current_time = time.time()
        
        # Time-based analysis
        hourly_gestures = {}
        daily_gestures = {}
        
        for gesture in self.gesture_state.gesture_history:
            gesture_hour = time.strftime('%H', time.localtime(gesture.timestamp))
            gesture_day = time.strftime('%Y-%m-%d', time.localtime(gesture.timestamp))
            
            hourly_gestures[gesture_hour] = hourly_gestures.get(gesture_hour, 0) + 1
            daily_gestures[gesture_day] = daily_gestures.get(gesture_day, 0) + 1
        
        # Confidence analysis
        confidences = [g.confidence for g in self.gesture_state.gesture_history]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Hand preference analysis
        hand_usage = {'Left': 0, 'Right': 0}
        for gesture in self.gesture_state.gesture_history:
            hand_usage[gesture.hand_data.handedness] += 1
        
        return {
            'total_gestures': len(self.gesture_state.gesture_history),
            'average_confidence': avg_confidence,
            'hand_preference': hand_usage,
            'hourly_distribution': hourly_gestures,
            'daily_distribution': daily_gestures,
            'sos_incidents': self.gesture_state.sos_count,
            'distraction_incidents': self.gesture_state.distraction_gestures,
            'session_duration': current_time - (self.gesture_state.gesture_history[0].timestamp 
                                               if self.gesture_state.gesture_history else current_time)
        }


if __name__ == "__main__":
    # Test gesture detector
    import sys
    sys.path.append('..')
    from utils.camera_utils import CameraManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    detector = HandGestureDetector()
    camera_manager = CameraManager()
    
    # Add test callbacks
    def on_sos(gesture):
        print(f"🚨 SOS DETECTED: {gesture.description}")
    
    def on_phone(gesture):
        print(f"📱 Phone gesture: {gesture.description}")
    
    def on_smoking(gesture):
        print(f"🚬 Smoking gesture: {gesture.description}")
    
    detector.add_gesture_callback(GestureType.SOS, on_sos)
    detector.add_gesture_callback(GestureType.PHONE_GESTURE, on_phone)
    detector.add_gesture_callback(GestureType.SMOKING_GESTURE, on_smoking)
    
    if not camera_manager.initialize_cameras():
        print("Failed to initialize cameras")
        exit(1)
    
    camera_manager.start_capture()
    
    try:
        frame_count = 0
        
        print("Hand Gesture Detection Test")
        print("Try different gestures:")
        print("  - Thumbs up/down")
        print("  - Peace sign")
        print("  - OK sign")
        print("  - Stop/Open palm")
        print("  - Pointing")
        print("  - Phone gesture (thumb + pinky)")
        print("  - SOS pattern")
        print("Press 'q' to quit, 'r' to reset, 's' for statistics")
        print()
        
        while True:
            # Get frame from cabin camera
            frame = camera_manager.get_frame('cabin_cam')
            if frame is None:
                # Fallback to driver camera
                frame = camera_manager.get_frame('driver_cam')
                if frame is None:
                    time.sleep(0.01)
                    continue
            
            # Process frame
            output_frame, analysis = detector.process_frame(frame)
            
            # Display results
            cv2.imshow('Hand Gesture Detection', output_frame)
            
            # Print analysis every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}:")
                print(f"  Hands Detected: {analysis['hands_detected']}")
                print(f"  Current Gestures: {analysis['current_gestures']}")
                print(f"  Total Gestures: {analysis['total_gestures']}")
                
                if analysis['sos_detected']:
                    print("  🚨 SOS DETECTED!")
                
                if analysis['distraction_detected']:
                    print("  ⚠️ DISTRACTION GESTURE DETECTED!")
                
                if analysis['recent_gesture_counts']:
                    print(f"  Recent Gestures: {analysis['recent_gesture_counts']}")
                
                summary = detector.get_gesture_summary()
                print(f"  Alert Level: {summary['alert_level']}")
                
                if summary['recommendations']:
                    print(f"  Recommendation: {summary['recommendations'][0]}")
                
                print()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_session()
                frame_count = 0
                print("Session reset")
            elif key == ord('s'):
                stats = detector.get_gesture_statistics()
                print("\n=== Gesture Statistics ===")
                print(f"Total Gestures: {stats['total_gestures']}")
                print(f"Average Confidence: {stats['average_confidence']:.3f}")
                print(f"Hand Preference: {stats['hand_preference']}")
                print(f"SOS Incidents: {stats['sos_incidents']}")
                print(f"Distraction Incidents: {stats['distraction_incidents']}")
                print(f"Session Duration: {stats['session_duration']:.1f}s")
                print("========================\n")
    
    finally:
        camera_manager.stop_capture()
        cv2.destroyAllWindows()
        print("Hand gesture detection test completed")