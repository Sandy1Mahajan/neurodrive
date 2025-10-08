"""
NeuroDrive Logging and Event Management System
Handles comprehensive logging, event tracking, and data persistence
"""

import logging
import sqlite3
import json
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import csv
import os
import yaml

class EventType(Enum):
    """Event types for the NeuroDrive system"""
    DROWSINESS_DETECTED = "drowsiness_detected"
    DISTRACTION_DETECTED = "distraction_detected"
    UNAUTHORIZED_OBJECT = "unauthorized_object"
    SPEED_VIOLATION = "speed_violation" 
    ALCOHOL_DETECTED = "alcohol_detected"
    GESTURE_DETECTED = "gesture_detected"
    BIO_AUTH_FAILED = "bio_auth_failed"
    SYSTEM_ALERT = "system_alert"
    SYSTEM_ERROR = "system_error"
    TRIP_START = "trip_start"
    TRIP_END = "trip_end"
    CALIBRATION = "calibration"
    MAINTENANCE = "maintenance"

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class NeuroDriveEvent:
    """Data class for NeuroDrive events"""
    timestamp: float
    event_type: EventType
    alert_level: AlertLevel
    driver_id: str
    trip_id: str
    data: Dict[str, Any]
    confidence: float = 0.0
    location: Optional[Dict[str, float]] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['alert_level'] = self.alert_level.value
        result['timestamp_human'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result


class DatabaseManager:
    """Handles database operations for event storage"""
    
    def __init__(self, db_path: str = "neurodrive_data.db"):
        self.db_path = db_path
        self.connection = None
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    alert_level INTEGER NOT NULL,
                    driver_id TEXT NOT NULL,
                    trip_id TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    latitude REAL,
                    longitude REAL,
                    data TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Drivers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drivers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    license_number TEXT,
                    face_encoding TEXT,
                    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Trips table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trips (
                    id TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    start_location TEXT,
                    end_location TEXT,
                    total_distance REAL DEFAULT 0.0,
                    avg_speed REAL DEFAULT 0.0,
                    max_speed REAL DEFAULT 0.0,
                    risk_score REAL DEFAULT 0.0,
                    violations_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (driver_id) REFERENCES drivers (id)
                )
            """)
            
            # Risk scores table (for analytics)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    trip_id TEXT NOT NULL,
                    drowsiness_score REAL DEFAULT 0.0,
                    distraction_score REAL DEFAULT 0.0,
                    speed_score REAL DEFAULT 0.0,
                    overall_score REAL DEFAULT 0.0,
                    FOREIGN KEY (trip_id) REFERENCES trips (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trip_id ON events(trip_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_scores_timestamp ON risk_scores(timestamp)")
            
            conn.commit()
            self.connection = conn
            
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
    
    def insert_event(self, event: NeuroDriveEvent) -> bool:
        """Insert event into database"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO events 
                    (timestamp, event_type, alert_level, driver_id, trip_id, 
                     confidence, latitude, longitude, data, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp,
                    event.event_type.value,
                    event.alert_level.value,
                    event.driver_id,
                    event.trip_id,
                    event.confidence,
                    event.location.get('lat') if event.location else None,
                    event.location.get('lng') if event.location else None,
                    json.dumps(event.data),
                    event.resolved
                ))
                self.connection.commit()
                return True
        except Exception as e:
            logging.error(f"Event insertion error: {e}")
            return False
    
    def insert_risk_score(self, trip_id: str, scores: Dict[str, float]) -> bool:
        """Insert risk score data"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO risk_scores 
                    (timestamp, trip_id, drowsiness_score, distraction_score, 
                     speed_score, overall_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    trip_id,
                    scores.get('drowsiness', 0.0),
                    scores.get('distraction', 0.0),
                    scores.get('speed', 0.0),
                    scores.get('overall', 0.0)
                ))
                self.connection.commit()
                return True
        except Exception as e:
            logging.error(f"Risk score insertion error: {e}")
            return False
    
    def get_events(self, trip_id: str = None, hours: int = 24) -> List[Dict]:
        """Get events from database"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                since_timestamp = time.time() - (hours * 3600)
                
                if trip_id:
                    cursor.execute("""
                        SELECT * FROM events 
                        WHERE trip_id = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    """, (trip_id, since_timestamp))
                else:
                    cursor.execute("""
                        SELECT * FROM events 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    """, (since_timestamp,))
                
                columns = [description[0] for description in cursor.description]
                events = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # Parse JSON data field
                for event in events:
                    if event['data']:
                        event['data'] = json.loads(event['data'])
                
                return events
        except Exception as e:
            logging.error(f"Event retrieval error: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class EventLogger:
    """Main event logging class with threading support"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_manager = DatabaseManager(
            self.config.get('database', {}).get('path', 'neurodrive_data.db')
        )
        
        # Event queue for async processing
        self.event_queue = queue.Queue()
        self.processing = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_events, daemon=True)
        self.process_thread.start()
        
        # Setup file logging
        self._setup_file_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Main log file
        main_handler = logging.FileHandler(f"{log_dir}/neurodrive.log")
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Events log file  
        event_handler = logging.FileHandler(f"{log_dir}/events.log")
        event_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        
        # Setup loggers
        root_logger = logging.getLogger()
        root_logger.addHandler(main_handler)
        
        event_logger = logging.getLogger('events')
        event_logger.addHandler(event_handler)
        
        level = getattr(logging, self.config.get('system', {}).get('log_level', 'INFO'))
        root_logger.setLevel(level)
        event_logger.setLevel(logging.INFO)
    
    def log_event(self, event: NeuroDriveEvent):
        """Log an event (async)"""
        self.event_queue.put(event)
    
    def log_drowsiness(self, driver_id: str, trip_id: str, 
                      severity: AlertLevel, data: Dict[str, Any],
                      confidence: float = 0.0, location: Dict = None):
        """Log drowsiness detection event"""
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.DROWSINESS_DETECTED,
            alert_level=severity,
            driver_id=driver_id,
            trip_id=trip_id,
            data=data,
            confidence=confidence,
            location=location
        )
        self.log_event(event)
    
    def log_distraction(self, driver_id: str, trip_id: str,
                       distraction_type: str, confidence: float,
                       location: Dict = None):
        """Log distraction event"""
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.DISTRACTION_DETECTED,
            alert_level=AlertLevel.MEDIUM,
            driver_id=driver_id,
            trip_id=trip_id,
            data={'distraction_type': distraction_type},
            confidence=confidence,
            location=location
        )
        self.log_event(event)
    
    def log_speed_violation(self, driver_id: str, trip_id: str,
                           current_speed: float, speed_limit: float,
                           location: Dict = None):
        """Log speed violation"""
        severity = AlertLevel.HIGH if current_speed > speed_limit + 20 else AlertLevel.MEDIUM
        
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.SPEED_VIOLATION,
            alert_level=severity,
            driver_id=driver_id,
            trip_id=trip_id,
            data={
                'current_speed': current_speed,
                'speed_limit': speed_limit,
                'violation_amount': current_speed - speed_limit
            },
            location=location
        )
        self.log_event(event)
    
    def log_object_detection(self, driver_id: str, trip_id: str,
                           object_type: str, confidence: float,
                           location: Dict = None):
        """Log unauthorized object detection"""
        severity = AlertLevel.HIGH if object_type in ['phone', 'cigarette'] else AlertLevel.MEDIUM
        
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.UNAUTHORIZED_OBJECT,
            alert_level=severity,
            driver_id=driver_id,
            trip_id=trip_id,
            data={'object_type': object_type},
            confidence=confidence,
            location=location
        )
        self.log_event(event)
    
    def log_gesture(self, driver_id: str, trip_id: str,
                   gesture_type: str, confidence: float,
                   location: Dict = None):
        """Log gesture detection"""
        severity = AlertLevel.CRITICAL if gesture_type == 'sos' else AlertLevel.LOW
        
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.GESTURE_DETECTED,
            alert_level=severity,
            driver_id=driver_id,
            trip_id=trip_id,
            data={'gesture_type': gesture_type},
            confidence=confidence,
            location=location
        )
        self.log_event(event)
    
    def log_system_error(self, error_type: str, error_message: str,
                        driver_id: str = "system", trip_id: str = "system"):
        """Log system errors"""
        event = NeuroDriveEvent(
            timestamp=time.time(),
            event_type=EventType.SYSTEM_ERROR,
            alert_level=AlertLevel.HIGH,
            driver_id=driver_id,
            trip_id=trip_id,
            data={'error_type': error_type, 'message': error_message}
        )
        self.log_event(event)
    
    def _process_events(self):
        """Background thread to process events"""
        event_logger = logging.getLogger('events')
        
        while self.processing:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                # Store in database
                self.db_manager.insert_event(event)
                
                # Log to file
                event_logger.info(json.dumps(event.to_dict()))
                
                # Print critical events to console
                if event.alert_level == AlertLevel.CRITICAL:
                    print(f"CRITICAL ALERT: {event.event_type.value} - {event.data}")
                
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    def get_analytics(self, trip_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get analytics for events"""
        events = self.db_manager.get_events(trip_id, hours)
        
        if not events:
            return {}
        
        # Count events by type
        event_counts = {}
        alert_counts = {level.name: 0 for level in AlertLevel}
        
        for event in events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            alert_level = AlertLevel(event['alert_level']).name
            alert_counts[alert_level] += 1
        
        # Calculate risk metrics
        total_events = len(events)
        critical_events = sum(1 for e in events if e['alert_level'] == AlertLevel.CRITICAL.value)
        high_events = sum(1 for e in events if e['alert_level'] == AlertLevel.HIGH.value)
        
        risk_percentage = ((critical_events * 4 + high_events * 2) / (total_events * 4)) * 100 if total_events > 0 else 0
        
        return {
            'total_events': total_events,
            'event_counts': event_counts,
            'alert_counts': alert_counts,
            'risk_percentage': risk_percentage,
            'critical_events': critical_events,
            'high_priority_events': high_events,
            'time_range_hours': hours
        }
    
    def export_events_csv(self, filename: str, trip_id: str = None, hours: int = 24):
        """Export events to CSV file"""
        events = self.db_manager.get_events(trip_id, hours)
        
        if not events:
            return False
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'timestamp_human', 'event_type', 'alert_level', 
                             'driver_id', 'trip_id', 'confidence', 'latitude', 'longitude', 'data']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for event in events:
                    # Convert timestamp to human readable
                    event['timestamp_human'] = datetime.fromtimestamp(event['timestamp']).isoformat()
                    # Convert data dict to string
                    event['data'] = json.dumps(event.get('data', {}))
                    writer.writerow(event)
            
            return True
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return False
    
    def cleanup_old_events(self, days_to_keep: int = 30):
        """Clean up events older than specified days"""
        try:
            cutoff_timestamp = time.time() - (days_to_keep * 24 * 3600)
            
            with self.db_manager.lock:
                cursor = self.db_manager.connection.cursor()
                cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_timestamp,))
                cursor.execute("DELETE FROM risk_scores WHERE timestamp < ?", (cutoff_timestamp,))
                self.db_manager.connection.commit()
                
                deleted_count = cursor.rowcount
                self.logger.info(f"Cleaned up {deleted_count} old events")
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """Stop event processing"""
        self.processing = False
        self.process_thread.join(timeout=5.0)
        self.db_manager.close()


class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def log_processing_time(self, module_name: str, processing_time: float):
        """Log processing time for a module"""
        with self.lock:
            if module_name not in self.metrics:
                self.metrics[module_name] = {
                    'total_time': 0.0,
                    'call_count': 0,
                    'avg_time': 0.0,
                    'max_time': 0.0,
                    'min_time': float('inf')
                }
            
            metrics = self.metrics[module_name]
            metrics['total_time'] += processing_time
            metrics['call_count'] += 1
            metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
            metrics['max_time'] = max(metrics['max_time'], processing_time)
            metrics['min_time'] = min(metrics['min_time'], processing_time)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'uptime_seconds': uptime,
                'uptime_human': str(timedelta(seconds=int(uptime))),
                'module_metrics': self.metrics.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.start_time = time.time()


# Decorator for timing function execution
def timed_execution(module_name: str, performance_monitor: PerformanceMonitor):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_monitor.log_processing_time(module_name, processing_time)
            
            return result
        return wrapper
    return decorator


# Global instances (initialized in main.py)
event_logger = None
performance_monitor = None

def initialize_logging(config_path: str = "config.yaml"):
    """Initialize global logging instances"""
    global event_logger, performance_monitor
    
    event_logger = EventLogger(config_path)
    performance_monitor = PerformanceMonitor()
    
    return event_logger, performance_monitor


if __name__ == "__main__":
    # Test the logging system
    import uuid
    
    # Initialize
    logger, perf_monitor = initialize_logging()
    
    # Generate test events
    driver_id = "test_driver_001"
    trip_id = str(uuid.uuid4())
    
    # Test various event types
    logger.log_drowsiness(
        driver_id, trip_id, AlertLevel.HIGH,
        {'eye_closure_duration': 3.5, 'yawn_detected': True},
        confidence=0.85
    )
    
    logger.log_distraction(
        driver_id, trip_id, "phone_use", 0.92
    )
    
    logger.log_speed_violation(
        driver_id, trip_id, 85.0, 60.0,
        {'lat': 40.7128, 'lng': -74.0060}
    )
    
    # Wait for processing
    time.sleep(2)
    
    # Get analytics
    analytics = logger.get_analytics(trip_id)
    print("Analytics:", json.dumps(analytics, indent=2))
    
    # Export to CSV
    logger.export_events_csv(f"test_events_{trip_id[:8]}.csv", trip_id)
    
    # Performance report
    perf_report = perf_monitor.get_performance_report()
    print("Performance:", json.dumps(perf_report, indent=2))
    
    # Cleanup
    logger.stop()