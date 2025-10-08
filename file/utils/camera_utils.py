"""
NeuroDrive Camera Utilities
Handles multiple camera feeds with threading and optimization
"""

import cv2
import threading
import queue
import time
import numpy as np
from typing import Dict, Optional, Tuple, List
import yaml
import logging

class CameraManager:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize camera manager with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.cameras = {}
        self.frame_queues = {}
        self.capture_threads = {}
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_cameras(self) -> bool:
        """Initialize all configured cameras"""
        try:
            camera_config = self.config['cameras']
            
            for cam_name, cam_settings in camera_config.items():
                self.logger.info(f"Initializing camera: {cam_name}")
                
                cap = cv2.VideoCapture(cam_settings['index'])
                if not cap.isOpened():
                    self.logger.warning(f"Failed to open camera {cam_name}")
                    continue
                    
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_settings['resolution'][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_settings['resolution'][1])
                cap.set(cv2.CAP_PROP_FPS, cam_settings['fps'])
                
                # Enable auto-exposure and auto-focus
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
                self.cameras[cam_name] = cap
                self.frame_queues[cam_name] = queue.Queue(maxsize=5)
                
            return len(self.cameras) > 0
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def start_capture(self):
        """Start threaded capture for all cameras"""
        self.running = True
        
        for cam_name, camera in self.cameras.items():
            thread = threading.Thread(
                target=self._capture_loop,
                args=(cam_name, camera),
                daemon=True
            )
            thread.start()
            self.capture_threads[cam_name] = thread
            
        self.logger.info(f"Started capture threads for {len(self.cameras)} cameras")
    
    def _capture_loop(self, cam_name: str, camera: cv2.VideoCapture):
        """Individual camera capture loop"""
        frame_skip = self.config.get('performance', {}).get('frame_skip', 1)
        skip_counter = 0
        
        while self.running:
            ret, frame = camera.read()
            
            if not ret:
                self.logger.warning(f"Failed to read from {cam_name}")
                time.sleep(0.1)
                continue
            
            # Frame skipping for performance
            skip_counter += 1
            if skip_counter % frame_skip != 0:
                continue
                
            # Apply preprocessing
            frame = self._preprocess_frame(frame, cam_name)
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queues[cam_name].put_nowait(frame)
            except queue.Full:
                # Remove oldest frame and add new one
                try:
                    self.frame_queues[cam_name].get_nowait()
                    self.frame_queues[cam_name].put_nowait(frame)
                except queue.Empty:
                    pass
    
    def _preprocess_frame(self, frame: np.ndarray, cam_name: str) -> np.ndarray:
        """Preprocess frame based on camera type"""
        if 'ir' in cam_name.lower():
            # IR camera preprocessing
            frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        elif 'external' in cam_name.lower():
            # External camera preprocessing for weather conditions
            frame = self._enhance_visibility(frame)
            
        # General denoising
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame
    
    def _enhance_visibility(self, frame: np.ndarray) -> np.ndarray:
        """Enhance visibility for weather conditions"""
        # CLAHE for contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def get_frame(self, cam_name: str) -> Optional[np.ndarray]:
        """Get latest frame from specified camera"""
        try:
            return self.frame_queues[cam_name].get_nowait()
        except queue.Empty:
            return None
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Get latest frames from all cameras"""
        frames = {}
        for cam_name in self.cameras.keys():
            frame = self.get_frame(cam_name)
            if frame is not None:
                frames[cam_name] = frame
        return frames
    
    def is_camera_active(self, cam_name: str) -> bool:
        """Check if camera is active and producing frames"""
        return (cam_name in self.cameras and 
                cam_name in self.capture_threads and
                self.capture_threads[cam_name].is_alive())
    
    def get_camera_info(self, cam_name: str) -> Dict:
        """Get camera information"""
        if cam_name not in self.cameras:
            return {}
            
        cap = self.cameras[cam_name]
        return {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
            'active': self.is_camera_active(cam_name)
        }
    
    def set_camera_property(self, cam_name: str, property_id: int, value: float):
        """Set camera property"""
        if cam_name in self.cameras:
            self.cameras[cam_name].set(property_id, value)
    
    def stop_capture(self):
        """Stop all camera capture threads"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.capture_threads.values():
            thread.join(timeout=2.0)
        
        # Release cameras
        for camera in self.cameras.values():
            camera.release()
            
        self.logger.info("All camera captures stopped")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_capture()


class FrameBuffer:
    """Circular buffer for storing recent frames"""
    
    def __init__(self, size: int = 30):
        self.size = size
        self.frames = []
        self.timestamps = []
        self.current_idx = 0
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """Add frame to buffer"""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            if len(self.frames) < self.size:
                self.frames.append(frame.copy())
                self.timestamps.append(timestamp)
            else:
                self.frames[self.current_idx] = frame.copy()
                self.timestamps[self.current_idx] = timestamp
                self.current_idx = (self.current_idx + 1) % self.size
    
    def get_recent_frames(self, count: int = 10) -> List[Tuple[np.ndarray, float]]:
        """Get most recent frames"""
        with self.lock:
            if len(self.frames) <= count:
                return list(zip(self.frames, self.timestamps))
            
            # Get recent frames in chronological order
            if len(self.frames) == self.size:
                start_idx = self.current_idx
                indices = [(start_idx - i - 1) % self.size for i in range(count)]
                indices.reverse()
            else:
                indices = list(range(max(0, len(self.frames) - count), len(self.frames)))
            
            return [(self.frames[i], self.timestamps[i]) for i in indices]
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()
            self.current_idx = 0


def detect_available_cameras(max_index: int = 10) -> List[int]:
    """Detect available camera indices"""
    available = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    
    return available


def calibrate_camera(camera_index: int, checkerboard_size: Tuple[int, int] = (9, 6)) -> Dict:
    """Simple camera calibration"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return {}
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3d points
    imgpoints = []  # 2d points
    
    frames_captured = 0
    required_frames = 20
    
    print(f"Camera calibration started. Show checkerboard pattern to camera {camera_index}")
    
    while frames_captured < required_frames:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
            frames_captured += 1
            
            # Draw corners
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret_corners)
            cv2.putText(frame, f"Captured: {frames_captured}/{required_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frames_captured >= 10:  # Minimum frames for calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        return {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'calibration_error': ret,
            'frames_used': frames_captured
        }
    
    return {}


if __name__ == "__main__":
    # Test camera detection
    available_cameras = detect_available_cameras()
    print(f"Available cameras: {available_cameras}")
    
    # Test camera manager
    if available_cameras:
        logging.basicConfig(level=logging.INFO)
        
        cam_manager = CameraManager()
        if cam_manager.initialize_cameras():
            cam_manager.start_capture()
            
            try:
                for i in range(100):  # Test for 100 frames
                    frames = cam_manager.get_all_frames()
                    print(f"Frame {i}: Got {len(frames)} camera feeds")
                    time.sleep(0.1)
            finally:
                cam_manager.stop_capture()
        else:
            print("Failed to initialize cameras")