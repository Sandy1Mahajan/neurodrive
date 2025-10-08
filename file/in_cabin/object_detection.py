"""
NeuroDrive Object Detection Module
Advanced detection of unauthorized objects in vehicle cabin using YOLOv8
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from enum import Enum
import yaml
import logging
from dataclasses import dataclass
from ultralytics import YOLO
import torch


class ObjectCategory(Enum):
    """Object categories for detection"""
    PHONE = "phone"
    CIGARETTE = "cigarette"
    BOTTLE = "bottle"
    CAN = "can"
    CUP = "cup"
    FOOD = "food"
    LAPTOP = "laptop"
    BOOK = "book"
    WEAPON = "weapon"
    ALCOHOL = "alcohol"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """Detected object data structure"""
    category: ObjectCategory
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: float
    timestamp: float
    class_id: int
    class_name: str
    tracking_id: Optional[int] = None


@dataclass
class DetectionState:
    """Object detection state tracking"""
    current_objects: List[DetectedObject]
    object_history: deque
    total_detections: int = 0
    unauthorized_objects: int = 0
    phone_detections: int = 0
    cigarette_detections: int = 0
    last_detection_time: float = 0.0
    persistent_objects: Dict[int, float] = None  # tracking_id -> first_seen_time

    def __post_init__(self):
        if self.persistent_objects is None:
            self.persistent_objects = {}


class ObjectDetector:
    """Advanced object detection for vehicle cabin monitoring"""

    def __init__(self, config_path: str = "config.yaml", model_path: str = "yolov8n.pt"):
        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        # Load config
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults.")
            self.config = {}

        # Object detection configuration
        obj_config = self.config.get('models', {}).get('object_detection', {})
        self.confidence_threshold = float(obj_config.get('confidence', 0.6))
        self.target_classes = obj_config.get('classes', [67, 39])  # default: phone, bottle etc

        # Initialize YOLOv8 model
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.model.to(self.device)
            except Exception:
                # some ultralytics builds may not expose .to or already handle device internally
                pass
            self.logger.info(f"YOLOv8 model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

        # COCO class names mapping
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

        # Custom object category mapping
        self.category_mapping = {
            67: ObjectCategory.PHONE,      # cell phone
            39: ObjectCategory.BOTTLE,     # bottle
            41: ObjectCategory.CUP,        # cup
            63: ObjectCategory.LAPTOP,     # laptop
            73: ObjectCategory.BOOK,       # book
            46: ObjectCategory.FOOD,       # banana (food example)
            47: ObjectCategory.FOOD,       # apple
            48: ObjectCategory.FOOD,       # sandwich
            52: ObjectCategory.FOOD,       # hot dog
            53: ObjectCategory.FOOD,       # pizza
        }

        # Unauthorized objects list
        self.unauthorized_categories = {
            ObjectCategory.PHONE,
            ObjectCategory.CIGARETTE,
            ObjectCategory.LAPTOP,
            ObjectCategory.ALCOHOL
        }

        # Distraction objects (less severe)
        self.distraction_categories = {
            ObjectCategory.FOOD,
            ObjectCategory.CUP,
            ObjectCategory.BOTTLE,
            ObjectCategory.BOOK
        }

        # Detection state
        self.detection_state = DetectionState(
            current_objects=[],
            object_history=deque(maxlen=200)
        )

        # Object tracking
        self.next_tracking_id = 0
        self.tracking_threshold = int(self.config.get('tracking_threshold', 50))  # pixels

        # Timing and persistence
        self.persistent_detection_time = float(self.config.get('persistent_detection_time', 2.0))  # seconds
        self.alert_cooldown = float(self.config.get('alert_cooldown', 5.0))  # seconds between alerts for same object
        self.last_alerts: Dict[str, float] = {}

        # Callbacks
        self.detection_callbacks: Dict[ObjectCategory, List[Callable]] = {}

        # Threading
        self.processing_lock = threading.Lock()

        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame for object detection"""
        with self.processing_lock:
            start_time = time.time()
            self.frame_count += 1

            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            output_frame = frame.copy()
            # Clear current_objects list for this frame; keep history separate
            self.detection_state.current_objects = []

            # Process detection results
            if results and len(results) > 0:
                detections = results[0]

                if hasattr(detections, 'boxes') and detections.boxes is not None and len(detections.boxes) > 0:
                    # Extract detection data
                    try:
                        boxes = detections.boxes.xyxy.cpu().numpy()
                        confidences = detections.boxes.conf.cpu().numpy()
                        class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        # Some ultralytics versions expose boxes differently
                        boxes = np.array(detections.boxes.xyxy)
                        confidences = np.array(detections.boxes.conf)
                        class_ids = np.array(detections.boxes.cls).astype(int)

                    # Process each detection
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if conf >= self.confidence_threshold:
                            detected_obj = self._create_detected_object(
                                box, float(conf), int(class_id), start_time
                            )

                            if detected_obj:
                                # Update object tracking
                                self._update_object_tracking(detected_obj)

                                self.detection_state.current_objects.append(detected_obj)
                                self.detection_state.object_history.append(detected_obj)
                                self.detection_state.total_detections += 1
                                self.detection_state.last_detection_time = start_time

                                # Check for unauthorized objects
                                self._check_unauthorized_object(detected_obj)

                                # Trigger callbacks
                                self._trigger_detection_callbacks(detected_obj)

            # Update persistent object tracking
            self._update_persistent_objects()

            # Draw detections on frame
            output_frame = self._draw_detections(output_frame)

            # Compile analysis data
            analysis_data = self._compile_analysis_data()

            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            analysis_data['processing_time'] = processing_time
            analysis_data['avg_processing_time'] = float(np.mean(self.processing_times)) if self.processing_times else 0.0
            analysis_data['fps_estimate'] = float(1.0 / analysis_data['avg_processing_time']) if analysis_data['avg_processing_time'] > 0 else 0.0

            return output_frame, analysis_data

    def _create_detected_object(self, box: np.ndarray, confidence: float,
                                class_id: int, timestamp: float) -> Optional[DetectedObject]:
        """Create DetectedObject from YOLO detection"""
        x1, y1, x2, y2 = box

        # Calculate bounding box properties
        width = max(0, int(x2 - x1))
        height = max(0, int(y2 - y1))
        center_x = int(x1 + width / 2)
        center_y = int(y1 + height / 2)
        area = float(width * height)

        # Get class name
        class_name = self.coco_classes.get(class_id, f"class_{class_id}")

        # Map to our object category
        category = self.category_mapping.get(class_id, ObjectCategory.UNKNOWN)

        # Special handling for cigarettes (may need custom detection)
        if "cigarette" in class_name.lower() or class_id == 999:  # Custom cigarette class
            category = ObjectCategory.CIGARETTE

        return DetectedObject(
            category=category,
            confidence=float(confidence),
            bbox=(int(x1), int(y1), width, height),
            center=(center_x, center_y),
            area=area,
            timestamp=float(timestamp),
            class_id=int(class_id),
            class_name=str(class_name)
        )

    def _update_object_tracking(self, detected_obj: DetectedObject):
        """Update object tracking with simple centroid-based tracking"""
        current_center = np.array(detected_obj.center)
        best_match_id = None
        min_distance = float('inf')

        # Compare against previously tracked objects in persistent_objects by center stored in current_objects
        for obj in self.detection_state.current_objects:
            if (obj.category == detected_obj.category and
                    obj.tracking_id is not None):
                existing_center = np.array(obj.center)
                distance = np.linalg.norm(current_center - existing_center)

                if distance < self.tracking_threshold and distance < min_distance:
                    min_distance = distance
                    best_match_id = obj.tracking_id

        # Assign tracking ID
        if best_match_id is not None:
            detected_obj.tracking_id = best_match_id
        else:
            detected_obj.tracking_id = self.next_tracking_id
            self.next_tracking_id += 1

            # Record first detection time
            self.detection_state.persistent_objects[detected_obj.tracking_id] = detected_obj.timestamp

    def _update_persistent_objects(self):
        """Update persistent object tracking and remove old objects"""
        current_time = time.time()
        current_tracking_ids = {obj.tracking_id for obj in self.detection_state.current_objects
                               if obj.tracking_id is not None}

        # Remove objects that haven't been seen recently
        to_remove = []
        for tracking_id, first_seen in list(self.detection_state.persistent_objects.items()):
            if tracking_id not in current_tracking_ids:
                # If object hasn't been seen for 5 seconds, remove it
                if current_time - first_seen > 5.0:
                    to_remove.append(tracking_id)

        for tracking_id in to_remove:
            try:
                del self.detection_state.persistent_objects[tracking_id]
            except KeyError:
                pass

    def _check_unauthorized_object(self, detected_obj: DetectedObject):
        """Check if detected object is unauthorized and handle accordingly"""
        current_time = time.time()

        # Check if object is unauthorized
        if detected_obj.category in self.unauthorized_categories:
            self.detection_state.unauthorized_objects += 1

            # Specific counting
            if detected_obj.category == ObjectCategory.PHONE:
                self.detection_state.phone_detections += 1
            elif detected_obj.category == ObjectCategory.CIGARETTE:
                self.detection_state.cigarette_detections += 1

            # Check for alert cooldown
            alert_key = f"{detected_obj.category.value}_{detected_obj.tracking_id}"
            last_alert_time = self.last_alerts.get(alert_key, 0)

            if current_time - last_alert_time > self.alert_cooldown:
                self.last_alerts[alert_key] = current_time

                # Log unauthorized object detection
                self.logger.warning(
                    f"Unauthorized object detected: {detected_obj.category.value} "
                    f"(confidence: {detected_obj.confidence:.2f})"
                )

        # Check for distraction objects
        elif detected_obj.category in self.distraction_categories:
            # Less severe alert for distraction objects
            if detected_obj.confidence > 0.8:  # Higher threshold for distraction alerts
                alert_key = f"{detected_obj.category.value}_{detected_obj.tracking_id}"
                last_alert_time = self.last_alerts.get(alert_key, 0)

                if current_time - last_alert_time > self.alert_cooldown * 2:  # Longer cooldown
                    self.last_alerts[alert_key] = current_time

                    self.logger.info(
                        f"Distraction object detected: {detected_obj.category.value} "
                        f"(confidence: {detected_obj.confidence:.2f})"
                    )

    def _trigger_detection_callbacks(self, detected_obj: DetectedObject):
        """Trigger registered callbacks for object detection"""
        callbacks = self.detection_callbacks.get(detected_obj.category, [])

        for callback in callbacks:
            try:
                callback(detected_obj)
            except Exception as e:
                self.logger.error(f"Detection callback error: {e}")

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame"""
        for detected_obj in self.detection_state.current_objects:
            x, y, w, h = detected_obj.bbox

            # Determine color based on object category
            if detected_obj.category in self.unauthorized_categories:
                color = (0, 0, 255)  # Red for unauthorized
                severity = "UNAUTHORIZED"
            elif detected_obj.category in self.distraction_categories:
                color = (0, 165, 255)  # Orange for distraction
                severity = "DISTRACTION"
            else:
                color = (0, 255, 0)  # Green for normal objects
                severity = "DETECTED"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Prepare label
            label = f"{detected_obj.class_name} ({detected_obj.confidence:.2f})"
            if detected_obj.tracking_id is not None:
                label += f" ID:{detected_obj.tracking_id}"

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, max(0, y - text_height - 10)), (x + text_width, y), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw severity indicator
            cv2.putText(frame, severity, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw persistence indicator for tracked objects
            if detected_obj.tracking_id in self.detection_state.persistent_objects:
                duration = time.time() - self.detection_state.persistent_objects[detected_obj.tracking_id]
                if duration > self.persistent_detection_time:
                    cv2.putText(frame, f"PERSISTENT ({duration:.1f}s)",
                               (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw detection statistics
        stats_y = 30
        cv2.putText(frame, f"Objects: {len(self.detection_state.current_objects)}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stats_y += 25
        if self.detection_state.unauthorized_objects > 0:
            cv2.putText(frame, f"Unauthorized: {self.detection_state.unauthorized_objects}",
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stats_y += 25
        cv2.putText(frame, f"Phones: {self.detection_state.phone_detections}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stats_y += 25
        cv2.putText(frame, f"Cigarettes: {self.detection_state.cigarette_detections}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stats_y += 25
        cv2.putText(frame, f"Total Detections: {self.detection_state.total_detections}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _compile_analysis_data(self) -> Dict:
        """Compile analysis data for the current detection cycle."""
        # Basic counts and recent top confidences
        data: Dict = {}
        data['timestamp'] = time.time()
        data['num_current_objects'] = len(self.detection_state.current_objects)
        data['total_detections'] = int(self.detection_state.total_detections)
        data['unauthorized_objects'] = int(self.detection_state.unauthorized_objects)
        data['phone_detections'] = int(self.detection_state.phone_detections)
        data['cigarette_detections'] = int(self.detection_state.cigarette_detections)
        data['last_detection_time'] = float(self.detection_state.last_detection_time)

        # Recent confidences and class breakdown
        confidences = [obj.confidence for obj in self.detection_state.current_objects]
        data['max_confidence'] = float(max(confidences)) if confidences else 0.0
        data['min_confidence'] = float(min(confidences)) if confidences else 0.0
        data['avg_confidence'] = float(np.mean(confidences)) if confidences else 0.0

        # Category counts in current frame
        category_counts: Dict[str, int] = {}
        for obj in self.detection_state.current_objects:
            cat = obj.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        data['category_counts'] = category_counts

        # Persistent objects summary
        persistent_summary = []
        current_time = time.time()
        for tid, first_seen in self.detection_state.persistent_objects.items():
            persistent_summary.append({
                'tracking_id': int(tid),
                'first_seen': float(first_seen),
                'duration': float(current_time - first_seen)
            })
        data['persistent_objects'] = persistent_summary

        # Recent object history (ids and classes, up to last 20)
        history_summary = []
        for obj in list(self.detection_state.object_history)[-20:]:
            history_summary.append({
                'timestamp': float(obj.timestamp),
                'tracking_id': obj.tracking_id,
                'class_name': obj.class_name,
                'category': obj.category.value,
                'confidence': float(obj.confidence)
            })
        data['recent_history'] = history_summary

        return data

    # ----- Utility API methods -----
    def register_callback(self, category: ObjectCategory, callback: Callable[[DetectedObject], None]):
        """Register a callback for a category. Callback receives DetectedObject."""
        if category not in self.detection_callbacks:
            self.detection_callbacks[category] = []
        self.detection_callbacks[category].append(callback)
        self.logger.debug(f"Callback registered for {category}")

    def unregister_callback(self, category: ObjectCategory, callback: Callable[[DetectedObject], None]):
        """Unregister a callback."""
        if category in self.detection_callbacks and callback in self.detection_callbacks[category]:
            self.detection_callbacks[category].remove(callback)
            self.logger.debug(f"Callback unregistered for {category}")

    def reset_state(self):
        """Reset detection state and counters (keeps model loaded)."""
        with self.processing_lock:
            self.detection_state = DetectionState(current_objects=[], object_history=deque(maxlen=200))
            self.next_tracking_id = 0
            self.last_alerts.clear()
            self.processing_times.clear()
            self.frame_count = 0
            self.logger.info("Detection state reset.")

    def save_config(self, path: str = "config.yaml"):
        """Persist current config to yaml."""
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(self.config, f)
            self.logger.info(f"Config saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def set_confidence_threshold(self, thresh: float):
        """Update confidence threshold at runtime."""
        self.confidence_threshold = float(thresh)
        self.logger.info(f"Confidence threshold set to {self.confidence_threshold}")

    def shutdown(self):
        """Perform graceful shutdown (release resources if needed)."""
        # If model has any cleanup API, call it here
        self.logger.info("Shutting down ObjectDetector.")

# ---- Demo usage ----
if __name__ == "__main__":
    # Quick demo: open webcam and run detection. Press 'q' to quit.
    detector = ObjectDetector(config_path="config.yaml", model_path="yolov8n.pt")

    def phone_alert(dobj: DetectedObject):
        print(f"[ALERT] Phone detected: ID={dobj.tracking_id} conf={dobj.confidence:.2f}")

    detector.register_callback(ObjectCategory.PHONE, phone_alert)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera.")
        exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_frame, analysis = detector.process_frame(frame)

            # show frame and simple analysis overlay
            cv2.imshow("NeuroDrive Object Detection", out_frame)
            # You can print or log analysis occasionally
            if detector.frame_count % 30 == 0:
                detector.logger.info(f"Analysis snapshot: objects={analysis['num_current_objects']} avg_proc={analysis['avg_processing_time']:.3f}s")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.shutdown()
