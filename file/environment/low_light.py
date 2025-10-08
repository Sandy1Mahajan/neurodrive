"""
Low Light Face Detection Module for NeuroDrive
Uses IR sensors and advanced CNN models for face detection in dark conditions
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from retinaface import RetinaFace
import mediapipe as mp
import yaml
import logging
import time
from collections import deque
import threading

class LowLightDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.env_config = self.config['environment']
        self.low_light_threshold = self.env_config['low_light_threshold']
        
        # Face detection models
        self.retinaface_model = None
        self.mediapipe_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Low light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        # Thermal/IR processing
        self.ir_processor = IRProcessor()
        
        # Results storage
        self.detection_buffer = deque(maxlen=10)
        self.current_light_level = 100  # lux
        self.is_low_light = False
        
        # Performance tracking
        self.fps_counter = FPSCounter()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_models(self):
        """Initialize face detection models"""
        try:
            # RetinaFace for robust detection
            self.logger.info("Initializing RetinaFace model...")
            # RetinaFace will auto-download if not present
            
            self.logger.info("Low light detection models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            return False

    def detect_light_level(self, frame):
        """Estimate ambient light level from frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness
            mean_brightness = np.mean(gray)
            
            # Estimate lux (rough approximation)
            # This is a simplified mapping - in production, use a light sensor
            estimated_lux = (mean_brightness / 255.0) * 500
            
            self.current_light_level = estimated_lux
            self.is_low_light = estimated_lux < self.low_light_threshold
            
            return estimated_lux
            
        except Exception as e:
            self.logger.error(f"Light level detection error: {e}")
            return 100  # Default to normal light

    def enhance_low_light_image(self, frame):
        """Enhance image for better face detection in low light"""
        try:
            # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            enhanced_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Method 2: Gamma correction
            gamma = 1.5  # Increase for darker images
            enhanced_gamma = np.power(frame / 255.0, 1.0 / gamma)
            enhanced_gamma = (enhanced_gamma * 255).astype(np.uint8)
            
            # Method 3: Bilateral filter for noise reduction
            enhanced_bilateral = cv2.bilateralFilter(enhanced_clahe, 9, 75, 75)
            
            # Method 4: Unsharp masking for detail enhancement
            gaussian = cv2.GaussianBlur(enhanced_bilateral, (0, 0), 2.0)
            enhanced_unsharp = cv2.addWeighted(enhanced_bilateral, 1.5, gaussian, -0.5, 0)
            
            # Combine enhancements
            alpha = 0.7  # Weight for enhanced image
            beta = 0.3   # Weight for original image
            final_enhanced = cv2.addWeighted(enhanced_unsharp, alpha, frame, beta, 0)
            
            return final_enhanced
            
        except Exception as e:
            self.logger.error(f"Image enhancement error: {e}")
            return frame

    def detect_faces_low_light(self, frame):
        """Detect faces in low light conditions"""
        faces = []
        
        try:
            light_level = self.detect_light_level(frame)
            
            if self.is_low_light:
                # Enhance image for low light
                enhanced_frame = self.enhance_low_light_image(frame)
                detection_frame = enhanced_frame
                self.logger.debug(f"Low light detected ({light_level:.1f} lux), using enhanced image")
            else:
                detection_frame = frame
            
            # Method 1: RetinaFace (most robust)
            retinaface_detections = self._detect_with_retinaface(detection_frame)
            
            # Method 2: MediaPipe (backup)
            mediapipe_detections = self._detect_with_mediapipe(detection_frame)
            
            # Combine and filter detections
            faces = self._combine_detections(
                retinaface_detections, 
                mediapipe_detections, 
                frame.shape
            )
            
            # Store detection results
            self.detection_buffer.append({
                'timestamp': time.time(),
                'faces': faces,
                'light_level': light_level,
                'is_low_light': self.is_low_light
            })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []

    def _detect_with_retinaface(self, frame):
        """Detect faces using RetinaFace"""
        faces = []
        
        try:
            # RetinaFace detection
            detections = RetinaFace.detect_faces(frame)
            
            if isinstance(detections, dict):
                for key, face_data in detections.items():
                    # Extract face region
                    facial_area = face_data['facial_area']
                    x1, y1, x2, y2 = facial_area
                    
                    # Extract landmarks
                    landmarks = face_data.get('landmarks', {})
                    
                    face = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': face_data.get('score', 0.9),
                        'landmarks': landmarks,
                        'method': 'retinaface'
                    }
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"RetinaFace detection error: {e}")
            return []

    def _detect_with_mediapipe(self, frame):
        """Detect faces using MediaPipe (backup method)"""
        faces = []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mediapipe_face.process(rgb_frame)
            
            if results.detections:
                h, w = frame.shape[:2]
                
                for detection in results.detections:
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    face = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': detection.score[0],
                        'landmarks': {},
                        'method': 'mediapipe'
                    }
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"MediaPipe detection error: {e}")
            return []

    def _combine_detections(self, retinaface_faces, mediapipe_faces, frame_shape):
        """Combine and filter face detections from multiple methods"""
        all_faces = []
        
        # Add RetinaFace detections (prioritized)
        all_faces.extend(retinaface_faces)
        
        # Add MediaPipe detections if no RetinaFace detections
        if not retinaface_faces:
            all_faces.extend(mediapipe_faces)
        
        # Filter by confidence and size
        filtered_faces = []
        h, w = frame_shape[:2]
        min_face_size = min(w, h) * 0.05  # Minimum 5% of frame
        
        for face in all_faces:
            bbox = face['bbox']
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            
            # Filter by size and confidence
            if (face_width > min_face_size and 
                face_height > min_face_size and 
                face['confidence'] > 0.5):
                filtered_faces.append(face)
        
        # Remove overlapping detections (NMS)
        final_faces = self._non_max_suppression(filtered_faces)
        
        return final_faces

    def _non_max_suppression(self, faces, iou_threshold=0.5):
        """Remove overlapping face detections"""
        if not faces:
            return []
        
        # Sort by confidence
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while faces:
            # Keep the highest confidence face
            current = faces.pop(0)
            keep.append(current)
            
            # Remove overlapping faces
            remaining = []
            for face in faces:
                if self._calculate_iou(current['bbox'], face['bbox']) < iou_threshold:
                    remaining.append(face)
            faces = remaining
        
        return keep

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def process_ir_frame(self, ir_frame):
        """Process infrared/thermal frame for face detection"""
        return self.ir_processor.detect_faces(ir_frame)

    def get_detection_results(self):
        """Get current detection results"""
        if not self.detection_buffer:
            return None
        
        latest = self.detection_buffer[-1]
        return {
            'faces': latest['faces'],
            'light_level': latest['light_level'],
            'is_low_light': latest['is_low_light'],
            'timestamp': latest['timestamp'],
            'fps': self.fps_counter.get_fps()
        }

    def draw_detections(self, frame, faces):
        """Draw face detections on frame"""
        annotated_frame = frame.copy()
        
        for face in faces:
            bbox = face['bbox']
            confidence = face['confidence']
            method = face['method']
            
            # Draw bounding box
            color = (0, 255, 0) if method == 'retinaface' else (0, 0, 255)
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 2)
            
            # Draw confidence and method
            label = f"{method}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw landmarks if available
            if face['landmarks']:
                self._draw_landmarks(annotated_frame, face['landmarks'])
        
        # Draw light level info
        light_color = (0, 0, 255) if self.is_low_light else (0, 255, 0)
        cv2.putText(annotated_frame, f"Light: {self.current_light_level:.1f} lux", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
        
        return annotated_frame

    def _draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks"""
        for landmark_name, (x, y) in landmarks.items():
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)


class IRProcessor:
    """Infrared/Thermal image processor for face detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, ir_frame):
        """Detect faces in infrared/thermal image"""
        try:
            # Convert thermal data to visible spectrum
            normalized_frame = self._normalize_thermal(ir_frame)
            
            # Use temperature gradients for face detection
            faces = self._detect_thermal_faces(normalized_frame)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"IR face detection error: {e}")
            return []
    
    def _normalize_thermal(self, thermal_frame):
        """Normalize thermal image to 8-bit"""
        # Assuming thermal frame is in temperature values
        min_temp = np.min(thermal_frame)
        max_temp = np.max(thermal_frame)
        
        if max_temp > min_temp:
            normalized = ((thermal_frame - min_temp) / (max_temp - min_temp) * 255)
            return normalized.astype(np.uint8)
        else:
            return np.zeros_like(thermal_frame, dtype=np.uint8)
    
    def _detect_thermal_faces(self, thermal_frame):
        """Detect faces using thermal signatures"""
        faces = []
        
        try:
            # Simple thermal face detection based on temperature blobs
            # In production, use specialized thermal face detection models
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(thermal_frame, (11, 11), 0)
            
            # Threshold for face temperature
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum face area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Face-like aspect ratio
                    if 0.5 < aspect_ratio < 2.0:
                        face = {
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.8,
                            'landmarks': {},
                            'method': 'thermal'
                        }
                        faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Thermal face detection error: {e}")
            return []


class FPSCounter:
    """FPS counter for performance monitoring"""
    
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0
        return 1.0 / np.mean(self.frame_times)


def test_low_light_detector():
    """Test the low light detector"""
    detector = LowLightDetector()
    
    if not detector.initialize_models():
        print("Failed to initialize models")
        return
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("Starting low light face detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detector.fps_counter.update()
            
            # Detect faces
            faces = detector.detect_faces_low_light(frame)
            
            # Draw detections
            annotated_frame = detector.draw_detections(frame, faces)
            
            # Display results
            results = detector.get_detection_results()
            if results:
                cv2.putText(annotated_frame, f"FPS: {results['fps']:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print(f"Faces: {len(faces)}, Light: {results['light_level']:.1f} lux, "
                      f"Low light: {results['is_low_light']}, FPS: {results['fps']:.1f}")
            
            cv2.imshow('Low Light Face Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'low_light_test_{int(time.time())}.jpg', annotated_frame)
                print("Frame saved")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_low_light_detector()