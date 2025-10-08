"""
NeuroDrive Biometric Authentication Module
Continuous driver verification using face recognition and behavioral biometrics
"""

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
import threading
import pickle
import os
from typing import Dict, List, Tuple, Optional
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import yaml
import logging
from dataclasses import dataclass
import hashlib
import base64

@dataclass
class DriverProfile:
    """Driver profile data structure"""
    driver_id: str
    name: str
    face_encodings: List[np.ndarray]
    behavioral_model: Optional[OneClassSVM] = None
    enrollment_date: float = 0.0
    last_verified: float = 0.0
    verification_count: int = 0
    confidence_threshold: float = 0.6

@dataclass
class AuthenticationState:
    """Authentication state tracking"""
    is_authenticated: bool = False
    current_driver_id: Optional[str] = None
    confidence_score: float = 0.0
    verification_failures: int = 0
    last_verification: float = 0.0
    unauthorized_duration: float = 0.0
    behavioral_anomaly: bool = False

class BiometricAuthenticator:
    """Advanced biometric authentication system"""
    
    def __init__(self, config_path: str = "config.yaml", profiles_dir: str = "driver_profiles"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Face recognition settings
        self.face_threshold = self.config.get('models', {}).get('face_recognition', {}).get('threshold', 0.6)
        self.verification_interval = 5.0  # Seconds between verifications
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.7
        )
        
        # Driver profiles storage
        self.driver_profiles: Dict[str, DriverProfile] = {}
        self.load_driver_profiles()
        
        # Authentication state
        self.auth_state = AuthenticationState()
        
        # Behavioral feature tracking
        self.behavioral_features = deque(maxlen=300)  # 10 seconds at 30fps
        self.feature_scaler = StandardScaler()
        self.behavioral_buffer = deque(maxlen=100)
        
        # Face recognition history
        self.face_encoding_history = deque(maxlen=50)
        self.verification_history = deque(maxlen=100)
        
        # Timing
        self.last_face_encoding = None
        self.unauthorized_start_time = None
        
        # Threading
        self.processing_lock = threading.Lock()
        self.enrollment_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def enroll_driver(self, name: str, frames: List[np.ndarray], 
                     driver_id: Optional[str] = None) -> Tuple[bool, str]:
        """Enroll a new driver with multiple face samples"""
        with self.enrollment_lock:
            if driver_id is None:
                # Generate unique driver ID
                driver_id = self._generate_driver_id(name)
            
            if driver_id in self.driver_profiles:
                return False, f"Driver {driver_id} already exists"
            
            # Extract face encodings from frames
            face_encodings = []
            
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if len(face_locations) == 1:  # Ensure exactly one face
                    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if encodings:
                        face_encodings.append(encodings[0])
                else:
                    self.logger.warning(f"Frame skipped: {len(face_locations)} faces detected")
            
            if len(face_encodings) < 3:
                return False, f"Insufficient face samples. Need at least 3, got {len(face_encodings)}"
            
            # Create driver profile
            profile = DriverProfile(
                driver_id=driver_id,
                name=name,
                face_encodings=face_encodings,
                enrollment_date=time.time(),
                confidence_threshold=self.face_threshold
            )
            
            self.driver_profiles[driver_id] = profile
            
            # Save profile to disk
            self._save_driver_profile(profile)
            
            self.logger.info(f"Driver {name} ({driver_id}) enrolled with {len(face_encodings)} face samples")
            
            return True, driver_id
    
    def verify_driver(self, frame: np.ndarray, expected_driver_id: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """Verify driver identity from frame"""
        with self.processing_lock:
            start_time = time.time()
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            output_frame = frame.copy()
            analysis_data = {}
            
            if results.detections:
                # Use the largest face detection
                detection = max(results.detections, key=lambda d: self._get_detection_area(d))
                
                # Extract face region for recognition
                face_frame = self._extract_face_region(rgb_frame, detection)
                
                if face_frame is not None:
                    # Perform face recognition
                    verification_result = self._perform_face_recognition(face_frame, expected_driver_id)
                    
                    # Update authentication state
                    self._update_authentication_state(verification_result)
                    
                    # Extract behavioral features
                    behavioral_features = self._extract_behavioral_features(frame, detection)
                    self._update_behavioral_model(behavioral_features)
                    
                    # Draw detection visualization
                    output_frame = self._draw_verification_overlay(output_frame, detection, verification_result)
                else:
                    self._handle_no_valid_face()
            else:
                self._handle_no_face_detection()
            
            # Compile analysis data
            analysis_data = self._compile_analysis_data()
            
            # Add performance info
            processing_time = time.time() - start_time
            analysis_data['processing_time'] = processing_time
            
            return output_frame, analysis_data
    
    def _generate_driver_id(self, name: str) -> str:
        """Generate unique driver ID"""
        # Create hash from name and timestamp
        data = f"{name}_{time.time()}".encode('utf-8')
        hash_object = hashlib.sha256(data)
        return base64.urlsafe_b64encode(hash_object.digest())[:12].decode('utf-8')
    
    def _get_detection_area(self, detection) -> float:
        """Calculate area of face detection bounding box"""
        bbox = detection.location_data.relative_bounding_box
        return bbox.width * bbox.height
    
    def _extract_face_region(self, rgb_frame: np.ndarray, detection) -> Optional[np.ndarray]:
        """Extract face region from frame using detection"""
        h, w = rgb_frame.shape[:2]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        if width > 50 and height > 50:  # Minimum face size
            face_region = rgb_frame[y:y+height, x:x+width]
            return face_region
        
        return None
    
    def _perform_face_recognition(self, face_frame: np.ndarray, 
                                expected_driver_id: Optional[str] = None) -> Dict:
        """Perform face recognition against enrolled drivers"""
        # Get face encoding
        face_locations = face_recognition.face_locations(face_frame, model="hog")
        
        if not face_locations:
            return {
                'recognized': False,
                'driver_id': None,
                'confidence': 0.0,
                'distance': float('inf')
            }
        
        face_encodings = face_recognition.face_encodings(face_frame, face_locations)
        
        if not face_encodings:
            return {
                'recognized': False,
                'driver_id': None,
                'confidence': 0.0,
                'distance': float('inf')
            }
        
        current_encoding = face_encodings[0]
        self.last_face_encoding = current_encoding
        self.face_encoding_history.append(current_encoding)
        
        # Compare against enrolled drivers
        best_match = None
        best_distance = float('inf')
        best_driver_id = None
        
        # If expected driver specified, check them first
        if expected_driver_id and expected_driver_id in self.driver_profiles:
            profile = self.driver_profiles[expected_driver_id]
            distances = face_recognition.face_distance(profile.face_encodings, current_encoding)
            min_distance = np.min(distances)
            
            if min_distance < profile.confidence_threshold:
                best_match = profile
                best_distance = min_distance
                best_driver_id = expected_driver_id
        
        # If no expected match or expected didn't match, check all drivers
        if best_match is None:
            for driver_id, profile in self.driver_profiles.items():
                distances = face_recognition.face_distance(profile.face_encodings, current_encoding)
                min_distance = np.min(distances)
                
                if min_distance < profile.confidence_threshold and min_distance < best_distance:
                    best_match = profile
                    best_distance = min_distance
                    best_driver_id = driver_id
        
        # Calculate confidence score (inverse of distance)
        confidence = max(0.0, 1.0 - best_distance) if best_distance != float('inf') else 0.0
        
        return {
            'recognized': best_match is not None,
            'driver_id': best_driver_id,
            'driver_name': best_match.name if best_match else None,
            'confidence': confidence,
            'distance': best_distance,
            'profile': best_match
        }
    
    def _extract_behavioral_features(self, frame: np.ndarray, detection) -> np.ndarray:
        """Extract behavioral biometric features"""
        features = []
        
        # Face position and size features
        bbox = detection.location_data.relative_bounding_box
        features.extend([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
        
        # Face region analysis
        h, w = frame.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        if width > 0 and height > 0:
            face_region = frame[y:y+height, x:x+width]
            
            # Color distribution features
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            features.extend([
                np.mean(face_gray),      # Average brightness
                np.std(face_gray),       # Brightness variation
                np.mean(face_region[:,:,0]),  # Blue channel mean
                np.mean(face_region[:,:,1]),  # Green channel mean
                np.mean(face_region[:,:,2])   # Red channel mean
            ])
            
            # Texture features (simplified)
            if face_gray.size > 100:
                # Calculate local binary pattern approximation
                kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
                edges = cv2.filter2D(face_gray, -1, kernel)
                features.append(np.mean(np.abs(edges)))
                features.append(np.std(edges))
        else:
            # Default values if face region is invalid
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Detection confidence
        features.append(detection.score[0] if detection.score else 0.0)
        
        # Temporal features (if we have history)
        if len(self.behavioral_features) > 0:
            prev_features = self.behavioral_features[-1]
            if len(prev_features) == len(features):
                # Feature velocity (change from previous frame)
                velocity = np.array(features) - np.array(prev_features)
                features.extend(velocity[:4])  # Only position/size velocities
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _update_behavioral_model(self, features: np.ndarray):
        """Update behavioral biometric model"""
        if len(features) == 0:
            return
        
        self.behavioral_features.append(features)
        
        # Only update model if we have authenticated driver
        if (self.auth_state.is_authenticated and 
            self.auth_state.current_driver_id and
            len(self.behavioral_features) >= 50):
            
            profile = self.driver_profiles.get(self.auth_state.current_driver_id)
            if profile:
                # Convert features to numpy array
                feature_matrix = np.array(list(self.behavioral_features))
                
                if profile.behavioral_model is None:
                    # Create new model
                    if len(self.behavioral_features) >= 100:
                        # Fit scaler and model
                        self.feature_scaler.fit(feature_matrix)
                        scaled_features = self.feature_scaler.transform(feature_matrix)
                        
                        profile.behavioral_model = OneClassSVM(
                            kernel='rbf',
                            gamma='scale',
                            nu=0.1  # Expected outlier fraction
                        )
                        profile.behavioral_model.fit(scaled_features)
                        
                        self.logger.info(f"Behavioral model created for driver {profile.name}")
                
                else:
                    # Check for anomalies with existing model
                    if hasattr(self.feature_scaler, 'mean_'):
                        try:
                            current_features = features.reshape(1, -1)
                            scaled_current = self.feature_scaler.transform(current_features)
                            anomaly_score = profile.behavioral_model.decision_function(scaled_current)[0]
                            
                            # Update anomaly state
                            self.auth_state.behavioral_anomaly = anomaly_score < -0.5
                            
                        except Exception as e:
                            self.logger.debug(f"Behavioral analysis error: {e}")
    
    def _update_authentication_state(self, verification_result: Dict):
        """Update authentication state based on verification result"""
        current_time = time.time()
        
        if verification_result['recognized'] and verification_result['confidence'] > 0.7:
            # Successful verification
            if not self.auth_state.is_authenticated:
                self.logger.info(f"Driver authenticated: {verification_result['driver_name']}")
            
            self.auth_state.is_authenticated = True
            self.auth_state.current_driver_id = verification_result['driver_id']
            self.auth_state.confidence_score = verification_result['confidence']
            self.auth_state.verification_failures = 0
            self.auth_state.last_verification = current_time
            self.auth_state.unauthorized_duration = 0.0
            
            # Update profile
            if verification_result['profile']:
                verification_result['profile'].last_verified = current_time
                verification_result['profile'].verification_count += 1
            
            # Reset unauthorized timer
            self.unauthorized_start_time = None
            
        else:
            # Failed verification
            self.auth_state.verification_failures += 1
            
            # Start unauthorized timer
            if self.unauthorized_start_time is None:
                self.unauthorized_start_time = current_time
            
            self.auth_state.unauthorized_duration = current_time - self.unauthorized_start_time
            
            # Threshold for considering driver unauthorized
            if (self.auth_state.verification_failures > 10 or 
                self.auth_state.unauthorized_duration > 15.0):
                
                if self.auth_state.is_authenticated:
                    self.logger.warning("Driver authentication lost")
                
                self.auth_state.is_authenticated = False
                self.auth_state.current_driver_id = None
                self.auth_state.confidence_score = 0.0
        
        # Store verification history
        self.verification_history.append({
            'timestamp': current_time,
            'recognized': verification_result['recognized'],
            'confidence': verification_result['confidence'],
            'driver_id': verification_result.get('driver_id')
        })
    
    def _handle_no_valid_face(self):
        """Handle case when face is detected but not valid for recognition"""
        self.auth_state.verification_failures += 1
        
        if self.unauthorized_start_time is None:
            self.unauthorized_start_time = time.time()
        
        self.auth_state.unauthorized_duration = time.time() - self.unauthorized_start_time
    
    def _handle_no_face_detection(self):
        """Handle case when no face is detected"""
        current_time = time.time()
        
        # More lenient handling - could be temporary
        if current_time - self.auth_state.last_verification > 10.0:
            self.auth_state.verification_failures += 1
            
            if self.unauthorized_start_time is None:
                self.unauthorized_start_time = current_time
            
            self.auth_state.unauthorized_duration = current_time - self.unauthorized_start_time
            
            # Only revoke authentication after extended period
            if self.auth_state.unauthorized_duration > 30.0:
                self.auth_state.is_authenticated = False
                self.auth_state.current_driver_id = None
    
    def _draw_verification_overlay(self, frame: np.ndarray, detection, 
                                 verification_result: Dict) -> np.ndarray:
        """Draw verification overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw face detection box
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Color based on verification status
        if verification_result['recognized']:
            color = (0, 255, 0) if verification_result['confidence'] > 0.8 else (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        # Draw verification info
        if verification_result['recognized']:
            label = f"{verification_result['driver_name']} ({verification_result['confidence']:.2f})"
        else:
            label = "UNAUTHORIZED"
        
        # Background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Authentication status
        status_y = 30
        if self.auth_state.is_authenticated:
            status_text = f"AUTHENTICATED: {self.driver_profiles[self.auth_state.current_driver_id].name if self.auth_state.current_driver_id else 'Unknown'}"
            status_color = (0, 255, 0)
        else:
            status_text = "NOT AUTHENTICATED"
            status_color = (0, 0, 255)
            
            if self.auth_state.unauthorized_duration > 0:
                status_text += f" ({self.auth_state.unauthorized_duration:.1f}s)"
        
        cv2.putText(frame, status_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Verification failures
        if self.auth_state.verification_failures > 0:
            fail_text = f"Verification failures: {self.auth_state.verification_failures}"
            cv2.putText(frame, fail_text, (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Behavioral anomaly indicator
        if self.auth_state.behavioral_anomaly:
            cv2.putText(frame, "BEHAVIORAL ANOMALY", (10, status_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame
    
    def _compile_analysis_data(self) -> Dict:
        """Compile all analysis data into a dictionary"""
        current_time = time.time()
        
        # Calculate recent verification statistics
        recent_verifications = [v for v in self.verification_history if current_time - v['timestamp'] < 300]
        success_rate = np.mean([v['recognized'] for v in recent_verifications]) if recent_verifications else 0.0
        
        return {
            'timestamp': current_time,
            
            # Authentication state
            'is_authenticated': self.auth_state.is_authenticated,
            'current_driver_id': self.auth_state.current_driver_id,
            'current_driver_name': self.driver_profiles[self.auth_state.current_driver_id].name if self.auth_state.current_driver_id else None,
            'confidence_score': self.auth_state.confidence_score,
            'verification_failures': self.auth_state.verification_failures,
            'last_verification': self.auth_state.last_verification,
            'unauthorized_duration': self.auth_state.unauthorized_duration,
            'behavioral_anomaly': self.auth_state.behavioral_anomaly,
            
            # Statistics
            'total_enrolled_drivers': len(self.driver_profiles),
            'recent_success_rate': success_rate,
            'recent_verification_count': len(recent_verifications),
            
            # Timing
            'time_since_last_verification': current_time - self.auth_state.last_verification
        }
    
    def get_authentication_summary(self) -> Dict:
        """Get summarized authentication status"""
        current_time = time.time()
        
        # Determine security level
        if not self.auth_state.is_authenticated:
            security_level = "UNAUTHORIZED"
        elif self.auth_state.behavioral_anomaly:
            security_level = "ANOMALOUS_BEHAVIOR"
        elif self.auth_state.confidence_score > 0.9:
            security_level = "HIGH_CONFIDENCE"
        elif self.auth_state.confidence_score > 0.7:
            security_level = "MEDIUM_CONFIDENCE"
        else:
            security_level = "LOW_CONFIDENCE"
        
        return {
            'security_level': security_level,
            'authenticated': self.auth_state.is_authenticated,
            'driver_name': self.driver_profiles[self.auth_state.current_driver_id].name if self.auth_state.current_driver_id else None,
            'confidence': self.auth_state.confidence_score,
            'unauthorized_duration': self.auth_state.unauthorized_duration,
            'verification_failures': self.auth_state.verification_failures,
            'behavioral_anomaly': self.auth_state.behavioral_anomaly,
            'time_since_verification': current_time - self.auth_state.last_verification,
            'recommendations': self._get_security_recommendations()
        }
    
    def _get_security_recommendations(self) -> List[str]:
        """Get security recommendations based on current state"""
        recommendations = []
        
        if not self.auth_state.is_authenticated:
            if self.auth_state.unauthorized_duration > 30:
                recommendations.extend([
                    "UNAUTHORIZED DRIVER DETECTED",
                    "Vehicle should be immobilized",
                    "Alert fleet management immediately"
                ])
            else:
                recommendations.extend([
                    "Driver verification required",
                    "Position face clearly in camera view",
                    "Ensure adequate lighting"
                ])
        
        elif self.auth_state.behavioral_anomaly:
            recommendations.extend([
                "Unusual behavioral patterns detected",
                "Monitor driver closely",
                "Consider additional verification"
            ])
        
        elif self.auth_state.confidence_score < 0.7:
            recommendations.extend([
                "Low confidence in driver identity",
                "Improve camera positioning",
                "Re-enroll driver if issues persist"
            ])
        
        return recommendations
    
    def save_driver_profiles(self):
        """Save all driver profiles to disk"""
        for profile in self.driver_profiles.values():
            self._save_driver_profile(profile)
    
    def _save_driver_profile(self, profile: DriverProfile):
        """Save individual driver profile"""
        profile_path = os.path.join(self.profiles_dir, f"{profile.driver_id}.pkl")
        
        try:
            with open(profile_path, 'wb') as f:
                pickle.dump(profile, f)
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.driver_id}: {e}")
    
    def load_driver_profiles(self):
        """Load all driver profiles from disk"""
        if not os.path.exists(self.profiles_dir):
            return
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.pkl'):
                profile_path = os.path.join(self.profiles_dir, filename)
                
                try:
                    with open(profile_path, 'rb') as f:
                        profile = pickle.load(f)
                        self.driver_profiles[profile.driver_id] = profile
                        
                    self.logger.info(f"Loaded profile: {profile.name} ({profile.driver_id})")
                except Exception as e:
                    self.logger.error(f"Failed to load profile {filename}: {e}")
    
    def delete_driver_profile(self, driver_id: str) -> bool:
        """Delete a driver profile"""
        if driver_id not in self.driver_profiles:
            return False
        
        # Remove from memory
        del self.driver_profiles[driver_id]
        
        # Remove from disk
        profile_path = os.path.join(self.profiles_dir, f"{driver_id}.pkl")
        try:
            if os.path.exists(profile_path):
                os.remove(profile_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete profile file {driver_id}: {e}")
            return False
    
    def reset_session(self):
        """Reset authentication session"""
        self.auth_state = AuthenticationState()
        self.behavioral_features.clear()
        self.face_encoding_history.clear()
        self.verification_history.clear()
        self.unauthorized_start_time = None
        
        self.logger.info("Biometric authentication session reset")


if __name__ == "__main__":
    # Test biometric authenticator
    import sys
    sys.path.append('..')
    from utils.camera_utils import CameraManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    auth = BiometricAuthenticator()
    camera_manager = CameraManager()
    
    if not camera_manager.initialize_cameras():
        print("Failed to initialize cameras")
        exit(1)
    
    camera_manager.start_capture()
    
    try:
        enrollment_mode = False
        enrollment_frames = []
        frame_count = 0
        
        print("Biometric Authentication Test")
        print("Commands:")
        print("  'e' - Enter enrollment mode")
        print("  'r' - Reset session")
        print("  's' - Show driver profiles")
        print("  'q' - Quit")
        print()
        
        while True:
            # Get frame from driver camera
            frame = camera_manager.get_frame('driver_cam')
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame
            if enrollment_mode:
                # Enrollment mode - collect frames
                enrollment_frames.append(frame.copy())
                
                cv2.putText(frame, f"ENROLLMENT MODE - Frame {len(enrollment_frames)}/20", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Look directly at camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if len(enrollment_frames) >= 20:
                    # Complete enrollment
                    name = input("Enter driver name: ")
                    success, driver_id = auth.enroll_driver(name, enrollment_frames)
                    
                    if success:
                        print(f"Driver {name} enrolled successfully with ID: {driver_id}")
                    else:
                        print(f"Enrollment failed: {driver_id}")
                    
                    enrollment_mode = False
                    enrollment_frames = []
            
            else:
                # Normal verification mode
                output_frame, analysis = auth.verify_driver(frame)
                frame = output_frame
                
                # Print analysis every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}:")
                    print(f"  Authenticated: {analysis['is_authenticated']}")
                    if analysis['current_driver_name']:
                        print(f"  Driver: {analysis['current_driver_name']}")
                        print(f"  Confidence: {analysis['confidence_score']:.3f}")
                    
                    if analysis['verification_failures'] > 0:
                        print(f"  Verification Failures: {analysis['verification_failures']}")
                    
                    if analysis['unauthorized_duration'] > 0:
                        print(f"  Unauthorized Duration: {analysis['unauthorized_duration']:.1f}s")
                    
                    if analysis['behavioral_anomaly']:
                        print("  *** BEHAVIORAL ANOMALY DETECTED ***")
                    
                    summary = auth.get_authentication_summary()
                    print(f"  Security Level: {summary['security_level']}")
                    
                    if summary['recommendations']:
                        print(f"  Recommendation: {summary['recommendations'][0]}")
                    
                    print()
            
            # Display frame
            cv2.imshow('Biometric Authentication', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e') and not enrollment_mode:
                enrollment_mode = True
                enrollment_frames = []
                print("Entering enrollment mode - collect 20 frames...")
            elif key == ord('r'):
                auth.reset_session()
                print("Session reset")
            elif key == ord('s'):
                print("\nEnrolled Drivers:")
                for driver_id, profile in auth.driver_profiles.items():
                    print(f"  {profile.name} ({driver_id}) - {len(profile.face_encodings)} encodings")
                print()
    
    finally:
        # Save profiles before exit
        auth.save_driver_profiles()
        camera_manager.stop_capture()
        cv2.destroyAllWindows()
        print("Biometric authentication test completed")