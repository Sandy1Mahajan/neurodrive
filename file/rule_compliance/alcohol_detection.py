"""
Alcohol Detection Module for NeuroDrive
Multi-modal alcohol detection using sensor fusion and behavioral analysis
"""

import cv2
import numpy as np
import time
import yaml
import logging
import threading
from collections import deque
import mediapipe as mp

class AlcoholDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Alcohol detection thresholds
        self.bac_threshold = 0.08  # Legal BAC limit (can be configured)
        self.face_redness_threshold = 0.6
        self.eye_redness_threshold = 0.7
        self.behavioral_threshold = 0.65
        
        # MediaPipe for facial analysis
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection state
        self.current_bac = 0.0
        self.alcohol_probability = 0.0
        self.detection_history = deque(maxlen=100)
        
        # Sensor data (simulated - in production would be real sensors)
        self.breathalyzer_reading = 0.0
        self.last_breathalyzer_time = 0
        
        # Behavioral indicators
        self.behavioral_score = 0.0
        self.eye_tracking_data = deque(maxlen=50)
        self.facial_analysis_data = deque(maxlen=30)
        
        # Analysis results
        self.detection_results = {
            'is_detected': False,
            'confidence': 0.0,
            'bac_estimate': 0.0,
            'detection_methods': [],
            'behavioral_indicators': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def simulate_breathalyzer_reading(self):
        """Simulate breathalyzer sensor reading (replace with actual sensor in production)"""
        # In production, this would interface with actual breathalyzer hardware
        # For demo purposes, we simulate readings
        current_time = time.time()
        
        # Simulate periodic readings (every 30 seconds)
        if current_time - self.last_breathalyzer_time > 30:
            # Simulate BAC reading with some noise
            base_bac = 0.02  # Baseline (can be adjusted for testing)
            noise = np.random.normal(0, 0.005)  # Small amount of sensor noise
            self.breathalyzer_reading = max(0, base_bac + noise)
            self.last_breathalyzer_time = current_time
            
            self.logger.debug(f"Simulated breathalyzer reading: {self.breathalyzer_reading:.4f}")
        
        return self.breathalyzer_reading

    def analyze_facial_redness(self, frame):
        """Analyze facial redness as indicator of alcohol consumption"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                return 0.0
            
            h, w = frame.shape[:2]
            face_redness_scores = []
            
            for detection in results.detections:
                # Extract face region
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_region = frame[y1:y2, x1:x2]
                    
                    # Analyze redness in face region
                    redness_score = self._calculate_redness_score(face_region)
                    face_redness_scores.append(redness_score)
            
            # Return average redness score
            return np.mean(face_redness_scores) if face_redness_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Facial redness analysis error: {e}")
            return 0.0

    def _calculate_redness_score(self, face_region):
        """Calculate redness score from face region"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Define red color ranges in HSV
            # Red hue ranges: 0-10 and 160-180
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            # Create masks for red regions
            mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Calculate percentage of red pixels
            total_pixels = face_region.shape[0] * face_region.shape[1]
            red_pixels = cv2.countNonZero(red_mask)
            redness_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
            
            # Also analyze RGB channels directly
            b, g, r = cv2.split(face_region)
            avg_red = np.mean(r)
            avg_green = np.mean(g)
            avg_blue = np.mean(b)
            
            # Calculate red dominance
            total_intensity = avg_red + avg_green + avg_blue
            red_dominance = avg_red / total_intensity if total_intensity > 0 else 0
            
            # Combine metrics
            redness_score = (redness_ratio * 0.6) + (red_dominance * 0.4)
            
            return min(redness_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Redness calculation error: {e}")
            return 0.0

    def analyze_eye_characteristics(self, frame):
        """Analyze eye characteristics for alcohol indicators"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {'redness': 0.0, 'drooping': 0.0, 'dilation': 0.0}
            
            h, w = frame.shape[:2]
            landmarks = results.multi_face_landmarks[0]
            
            # Eye landmark indices (MediaPipe)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            eye_analysis = {}
            
            for eye_name, eye_indices in [('left', left_eye_indices), ('right', right_eye_indices)]:
                eye_points = []
                for idx in eye_indices:
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    eye_points.append([x, y])
                
                eye_points = np.array(eye_points)
                
                # Extract eye region
                x_min, y_min = np.min(eye_points, axis=0)
                x_max, y_max = np.max(eye_points, axis=0)
                
                # Add padding
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                if x_max > x_min and y_max > y_min:
                    eye_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Analyze eye characteristics
                    redness = self._analyze_eye_redness(eye_region)
                    drooping = self._analyze_eye_drooping(eye_points)
                    
                    eye_analysis[eye_name] = {
                        'redness': redness,
                        'drooping': drooping
                    }
            
            # Average results from both eyes
            if eye_analysis:
                avg_redness = np.mean([eye['redness'] for eye in eye_analysis.values()])
                avg_drooping = np.mean([eye['drooping'] for eye in eye_analysis.values()])
                
                return {
                    'redness': avg_redness,
                    'drooping': avg_drooping,
                    'dilation': 0.0  # Placeholder - requires specialized hardware
                }
            
            return {'redness': 0.0, 'drooping': 0.0, 'dilation': 0.0}
            
        except Exception as e:
            self.logger.error(f"Eye analysis error: {e}")
            return {'redness': 0.0, 'drooping': 0.0, 'dilation': 0.0}

    def _analyze_eye_redness(self, eye_region):
        """Analyze redness in eye region"""
        # Similar to facial redness but focused on eye area
        return self._calculate_redness_score(eye_region)

    def _analyze_eye_drooping(self, eye_points):
        """Analyze eye drooping/ptosis"""
        try:
            if len(eye_points) < 6:
                return 0.0
            
            # Calculate eye opening ratio
            # Top and bottom points of eye
            eye_top = np.min(eye_points[:, 1])
            eye_bottom = np.max(eye_points[:, 1])
            eye_left = np.min(eye_points[:, 0])
            eye_right = np.max(eye_points[:, 0])
            
            eye_height = eye_bottom - eye_top
            eye_width = eye_right - eye_left
            
            if eye_width > 0:
                aspect_ratio = eye_height / eye_width
                # Lower aspect ratio indicates more drooping
                # Normal ratio is around 0.3, droopy eyes have lower ratios
                drooping_score = max(0, (0.3 - aspect_ratio) / 0.3)
                return min(drooping_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Eye drooping analysis error: {e}")
            return 0.0

    def analyze_behavioral_indicators(self, head_pose_data=None, eye_tracking_data=None):
        """Analyze behavioral indicators of alcohol impairment"""
        try:
            behavioral_score = 0.0
            indicators = []
            
            # Head movement analysis
            if head_pose_data:
                # Excessive head movement/instability
                if len(head_pose_data.get('pose_history', [])) > 10:
                    recent_poses = head_pose_data['pose_history'][-10:]
                    yaw_variance = np.var([p.get('yaw', 0) for p in recent_poses])
                    pitch_variance = np.var([p.get('pitch', 0) for p in recent_poses])
                    
                    movement_instability = (yaw_variance + pitch_variance) / 200.0  # Normalize
                    
                    if movement_instability > 0.5:
                        behavioral_score += 0.3
                        indicators.append('head_movement_instability')
            
            # Eye tracking analysis
            if eye_tracking_data:
                # Analyze gaze patterns
                gaze_instability = self._analyze_gaze_patterns(eye_tracking_data)
                if gaze_instability > 0.6:
                    behavioral_score += 0.4
                    indicators.append('gaze_instability')
            
            # Reaction time analysis (would need additional input)
            # This would typically come from steering wheel sensors or response tests
            
            self.behavioral_score = min(behavioral_score, 1.0)
            
            return {
                'score': self.behavioral_score,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Behavioral analysis error: {e}")
            return {'score': 0.0, 'indicators': []}

    def _analyze_gaze_patterns(self, eye_tracking_data):
        """Analyze gaze patterns for signs of impairment"""
        try:
            if len(eye_tracking_data) < 10:
                return 0.0
            
            # Calculate gaze stability metrics
            gaze_points = [(d.get('x', 0), d.get('y', 0)) for d in eye_tracking_data[-20:]]
            
            if not gaze_points:
                return 0.0
            
            # Calculate variance in gaze positions
            x_positions = [p[0] for p in gaze_points]
            y_positions = [p[1] for p in gaze_points]
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # High variance indicates unstable gaze
            instability = (x_variance + y_variance) / 10000.0  # Normalize
            
            return min(instability, 1.0)
            
        except Exception as e:
            self.logger.error(f"Gaze pattern analysis error: {e}")
            return 0.0

    def detect_alcohol_impairment(self, frame, head_pose_data=None, eye_tracking_data=None):
        """Main alcohol detection function combining all methods"""
        try:
            detection_methods = []
            confidence_scores = []
            
            # Method 1: Breathalyzer reading (if available)
            bac_reading = self.simulate_breathalyzer_reading()
            if bac_reading > 0:
                bac_confidence = min(bac_reading / self.bac_threshold, 1.0)
                confidence_scores.append(bac_confidence * 0.6)  # High weight for direct measurement
                detection_methods.append(f'breathalyzer_bac_{bac_reading:.4f}')
            
            # Method 2: Facial redness analysis
            face_redness = self.analyze_facial_redness(frame)
            if face_redness > self.face_redness_threshold:
                confidence_scores.append(face_redness * 0.3)
                detection_methods.append(f'facial_redness_{face_redness:.3f}')
            
            # Method 3: Eye characteristics
            eye_analysis = self.analyze_eye_characteristics(frame)
            eye_score = (eye_analysis['redness'] * 0.4 + 
                        eye_analysis['drooping'] * 0.6)
            
            if eye_score > self.eye_redness_threshold:
                confidence_scores.append(eye_score * 0.2)
                detection_methods.append(f'eye_analysis_{eye_score:.3f}')
            
            # Method 4: Behavioral indicators
            behavioral_analysis = self.analyze_behavioral_indicators(head_pose_data, eye_tracking_data)
            if behavioral_analysis['score'] > self.behavioral_threshold:
                confidence_scores.append(behavioral_analysis['score'] * 0.3)
                detection_methods.append('behavioral_indicators')
            
            # Calculate combined confidence
            total_confidence = sum(confidence_scores) if confidence_scores else 0.0
            is_detected = total_confidence > 0.5
            
            # Estimate BAC
            estimated_bac = bac_reading if bac_reading > 0 else (total_confidence * 0.15)
            
            # Update detection results
            self.detection_results = {
                'is_detected': is_detected,
                'confidence': total_confidence,
                'bac_estimate': estimated_bac,
                'bac_reading': bac_reading,
                'detection_methods': detection_methods,
                'behavioral_indicators': behavioral_analysis.get('indicators', []),
                'face_redness': face_redness,
                'eye_analysis': eye_analysis,
                'timestamp': time.time()
            }
            
            # Store in history
            self.detection_history.append(self.detection_results.copy())
            
            return self.detection_results
            
        except Exception as e:
            self.logger.error(f"Alcohol detection error: {e}")
            return {
                'is_detected': False,
                'confidence': 0.0,
                'bac_estimate': 0.0,
                'detection_methods': [],
                'behavioral_indicators': [],
                'timestamp': time.time()
            }

    def get_detection_results(self):
        """Get current alcohol detection results"""
        return self.detection_results.copy()

    def get_risk_score(self):
        """Get alcohol-based risk score (0-1)"""
        return self.detection_results.get('confidence', 0.0)

    def draw_detection_info(self, frame):
        """Draw alcohol detection information on frame"""
        annotated_frame = frame.copy()
        
        # Detection status
        status_text = "ALCOHOL DETECTED" if self.detection_results['is_detected'] else "NO ALCOHOL"
        status_color = (0, 0, 255) if self.detection_results['is_detected'] else (0, 255, 0)
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Confidence and BAC
        conf_text = f"Confidence: {self.detection_results['confidence']:.2f}"
        cv2.putText(annotated_frame, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        bac_text = f"Est. BAC: {self.detection_results['bac_estimate']:.4f}"
        cv2.putText(annotated_frame, bac_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection methods
        if self.detection_results['detection_methods']:
            methods_text = f"Methods: {len(self.detection_results['detection_methods'])}"
            cv2.putText(annotated_frame, methods_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated_frame

    def calibrate_baseline(self, frames_for_calibration=100):
        """Calibrate baseline measurements for the driver"""
        self.logger.info("Starting alcohol detection baseline calibration...")
        
        # This would collect baseline measurements when the driver is known to be sober
        # For production use, this should be done during initial setup
        
        baseline_data = {
            'face_redness_baseline': 0.2,  # Typical baseline
            'eye_redness_baseline': 0.15,
            'behavioral_baseline': 0.1
        }
        
        # Adjust thresholds based on baseline
        self.face_redness_threshold = baseline_data['face_redness_baseline'] + 0.3
        self.eye_redness_threshold = baseline_data['eye_redness_baseline'] + 0.4
        self.behavioral_threshold = baseline_data['behavioral_baseline'] + 0.4
        
        self.logger.info("Alcohol detection baseline calibrated")
        return baseline_data


def test_alcohol_detector():
    """Test the alcohol detector"""
    detector = AlcoholDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("Starting alcohol detection test...")
    print("Press 'q' to quit, 'c' to calibrate baseline")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect alcohol impairment
            results = detector.detect_alcohol_impairment(frame)
            
            # Draw detection info
            annotated_frame = detector.draw_detection_info(frame)
            
            # Display results
            print(f"Alcohol Detection - Detected: {results['is_detected']}, "
                  f"Confidence: {results['confidence']:.3f}, "
                  f"BAC Est: {results['bac_estimate']:.4f}, "
                  f"Methods: {len(results['detection_methods'])}")
            
            cv2.imshow('Alcohol Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.calibrate_baseline()
                print("Baseline calibrated")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_alcohol_detector()