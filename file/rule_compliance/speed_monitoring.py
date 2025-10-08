"""
Speed and Rule Monitoring Module for NeuroDrive
Monitors vehicle speed, traffic rules, and driving patterns using GPS and OBD data
"""

import time
import yaml
import logging
import threading
import json
from collections import deque
import numpy as np
import requests

# Simulated OBD and GPS interfaces (replace with actual libraries in production)
# import obd  # python-OBD library
# import gpsd  # GPS daemon interface

class SpeedMonitor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Speed and rule thresholds
        self.speed_limit_buffer = 5  # km/h buffer before violation
        self.harsh_acceleration_threshold = 3.0  # m/s²
        self.harsh_braking_threshold = -3.5  # m/s²
        self.sharp_turn_threshold = 0.5  # g-force
        
        # Vehicle state
        self.current_speed = 0.0  # km/h
        self.current_location = {'lat': 0.0, 'lon': 0.0}
        self.current_heading = 0.0  # degrees
        self.engine_rpm = 0
        self.throttle_position = 0  # percentage
        self.brake_pressure = 0  # percentage
        
        # Speed limit data
        self.current_speed_limit = 50  # Default speed limit
        self.speed_limit_source = 'default'
        
        # Violation tracking
        self.violations = deque(maxlen=1000)
        self.speed_history = deque(maxlen=100)  # Last 100 speed readings
        self.location_history = deque(maxlen=100)
        
        # Risk assessment
        self.driving_risk_score = 0.0
        self.risk_factors = {
            'speeding': 0.0,
            'harsh_acceleration': 0.0,
            'harsh_braking': 0.0,
            'sharp_turns': 0.0,
            'erratic_driving': 0.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Data sources
        self.obd_connection = None
        self.gps_connection = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_connections(self):
        """Initialize OBD and GPS connections"""
        try:
            # Initialize OBD connection
            self.obd_connection = self._initialize_obd()
            
            # Initialize GPS connection
            self.gps_connection = self._initialize_gps()
            
            self.logger.info("Speed monitoring connections initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection initialization error: {e}")
            return False

    def _initialize_obd(self):
        """Initialize OBD-II connection"""
        try:
            # In production, use actual OBD library:
            # import obd
            # connection = obd.OBD()  # auto-connects to available port
            # return connection
            
            # For demo, return simulated connection
            self.logger.info("OBD connection initialized (simulated)")
            return "simulated_obd"
            
        except Exception as e:
            self.logger.error(f"OBD initialization error: {e}")
            return None

    def _initialize_gps(self):
        """Initialize GPS connection"""
        try:
            # In production, use actual GPS library:
            # import gpsd
            # gpsd.connect()
            # return gpsd
            
            # For demo, return simulated connection
            self.logger.info("GPS connection initialized (simulated)")
            return "simulated_gps"
            
        except Exception as e:
            self.logger.error(f"GPS initialization error: {e}")
            return None

    def start_monitoring(self):
        """Start continuous speed and rule monitoring"""
        if not self.initialize_connections():
            self.logger.warning("Starting monitoring without all connections")
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Speed monitoring started")
        return True

    def stop_monitoring(self):
        """Stop speed monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        self.logger.info("Speed monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Read vehicle data
                self._read_vehicle_data()
                
                # Read location data
                self._read_location_data()
                
                # Get speed limit for current location
                self._update_speed_limit()
                
                # Analyze driving patterns
                self._analyze_driving_patterns()
                
                # Check for violations
                self._check_violations()
                
                # Calculate risk score
                self._calculate_driving_risk()
                
                time.sleep(0.1)  # 10Hz monitoring
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)

    def _read_vehicle_data(self):
        """Read vehicle data from OBD-II"""
        try:
            if self.obd_connection == "simulated_obd":
                # Simulate OBD data for demo
                self._simulate_obd_data()
            else:
                # In production, read actual OBD data:
                # speed_cmd = obd.commands.SPEED
                # rpm_cmd = obd.commands.RPM
                # throttle_cmd = obd.commands.THROTTLE_POS
                # 
                # speed_response = self.obd_connection.query(speed_cmd)
                # rpm_response = self.obd_connection.query(rpm_cmd)
                # throttle_response = self.obd_connection.query(throttle_cmd)
                # 
                # if speed_response.value:
                #     self.current_speed = speed_response.value.magnitude
                # if rpm_response.value:
                #     self.engine_rpm = rpm_response.value.magnitude
                # if throttle_response.value:
                #     self.throttle_position = throttle_response.value.magnitude
                pass
            
            # Store speed history
            self.speed_history.append({
                'speed': self.current_speed,
                'timestamp': time.time(),
                'rpm': self.engine_rpm,
                'throttle': self.throttle_position
            })
            
        except Exception as e:
            self.logger.error(f"Vehicle data reading error: {e}")

    def _simulate_obd_data(self):
        """Simulate OBD data for testing"""
        # Simulate realistic driving data
        base_time = time.time()
        
        # Simulate speed variations
        speed_variation = 10 * np.sin(base_time * 0.1) + np.random.normal(0, 2)
        self.current_speed = max(0, 45 + speed_variation)  # Base speed around 45 km/h
        
        # Simulate RPM based on speed
        self.engine_rpm = int(800 + (self.current_speed * 30))
        
        # Simulate throttle position
        self.throttle_position = min(100, max(0, 20 + np.random.normal(0, 10)))
        
        # Occasionally simulate harsh maneuvers
        if np.random.random() < 0.01:  # 1% chance
            # Simulate sudden acceleration or braking
            if np.random.random() < 0.5:
                self.current_speed += 15  # Sudden acceleration
                self.throttle_position = 90
            else:
                self.current_speed = max(0, self.current_speed - 20)  # Hard braking
                self.throttle_position = 5

    def _read_location_data(self):
        """Read GPS location data"""
        try:
            if self.gps_connection == "simulated_gps":
                # Simulate GPS data for demo
                self._simulate_gps_data()
            else:
                # In production, read actual GPS data:
                # packet = gpsd.get_current()
                # self.current_location = {
                #     'lat': packet.lat,
                #     'lon': packet.lon
                # }
                # self.current_heading = packet.track
                pass
            
            # Store location history
            self.location_history.append({
                'lat': self.current_location['lat'],
                'lon': self.current_location['lon'],
                'heading': self.current_heading,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"GPS data reading error: {e}")

    def _simulate_gps_data(self):
        """Simulate GPS data for testing"""
        # Simulate movement along a route
        base_time = time.time()
        
        # Simulate coordinates changing (moving vehicle)
        lat_change = 0.0001 * np.sin(base_time * 0.05)
        lon_change = 0.0001 * np.cos(base_time * 0.05)
        
        # Base coordinates (can be changed for different test locations)
        self.current_location = {
            'lat': 18.5204 + lat_change,  # Pune coordinates
            'lon': 73.8567 + lon_change
        }
        
        # Simulate heading changes
        self.current_heading = (90 + 20 * np.sin(base_time * 0.03)) % 360

    def _update_speed_limit(self):
        """Update speed limit based on current location"""
        try:
            # Method 1: Use mapping APIs (like Google Maps, OpenStreetMap)
            speed_limit = self._get_speed_limit_from_api()
            
            if speed_limit:
                self.current_speed_limit = speed_limit
                self.speed_limit_source = 'api'
            else:
                # Method 2: Use local database of speed limits
                speed_limit = self._get_speed_limit_from_database()
                
                if speed_limit:
                    self.current_speed_limit = speed_limit
                    self.speed_limit_source = 'database'
                else:
                    # Method 3: Use road type heuristics
                    self.current_speed_limit = self._estimate_speed_limit_by_road_type()
                    self.speed_limit_source = 'estimated'
            
        except Exception as e:
            self.logger.error(f"Speed limit update error: {e}")

    def _get_speed_limit_from_api(self):
        """Get speed limit from mapping API"""
        try:
            # Example using a hypothetical speed limit API
            # In production, use services like Google Roads API, HERE API, etc.
            
            # Simulated API response
            if np.random.random() < 0.8:  # 80% success rate
                # Simulate different speed limits based on area type
                if self.current_location['lat'] > 18.52:  # "Highway area"
                    return 80  # km/h
                else:  # "City area"
                    return 50  # km/h
            
            return None  # API failed
            
        except Exception as e:
            self.logger.error(f"Speed limit API error: {e}")
            return None

    def _get_speed_limit_from_database(self):
        """Get speed limit from local database"""
        # In production, this would query a local database of speed limits
        # based on road segments and coordinates
        
        # Simulated database lookup
        road_segments = {
            'highway': 80,
            'arterial': 60,
            'residential': 40,
            'school_zone': 25
        }
        
        # Simple classification based on coordinates (for demo)
        if self.current_speed > 60:  # Assume highway if driving fast
            return road_segments['highway']
        else:
            return road_segments['arterial']

    def _estimate_speed_limit_by_road_type(self):
        """Estimate speed limit based on road type and area"""
        # Fallback method using general rules
        
        # Time-based adjustments
        current_hour = time.localtime().tm_hour
        
        # Base speed limit
        base_limit = 50  # Default city limit
        
        # Adjust for time (school zones, etc.)
        if 7 <= current_hour <= 9 or 15 <= current_hour <= 17:  # School hours
            base_limit = min(base_limit, 40)
        
        return base_limit

    def _analyze_driving_patterns(self):
        """Analyze driving patterns for risk assessment"""
        try:
            if len(self.speed_history) < 5:
                return
            
            recent_speeds = [s['speed'] for s in list(self.speed_history)[-10:]]
            recent_times = [s['timestamp'] for s in list(self.speed_history)[-10:]]
            
            # Calculate acceleration
            if len(recent_speeds) >= 2:
                speed_changes = np.diff(recent_speeds)
                time_changes = np.diff(recent_times)
                
                # Convert km/h to m/s and calculate acceleration
                accelerations = [(ds * 1000/3600) / dt for ds, dt in zip(speed_changes, time_changes) if dt > 0]
                
                if accelerations:
                    current_acceleration = accelerations[-1]
                    
                    # Check for harsh acceleration
                    if current_acceleration > self.harsh_acceleration_threshold:
                        self.risk_factors['harsh_acceleration'] = min(
                            self.risk_factors['harsh_acceleration'] + 0.1, 1.0
                        )
                        self._log_violation('harsh_acceleration', current_acceleration)
                    
                    # Check for harsh braking
                    if current_acceleration < self.harsh_braking_threshold:
                        self.risk_factors['harsh_braking'] = min(
                            self.risk_factors['harsh_braking'] + 0.1, 1.0
                        )
                        self._log_violation('harsh_braking', current_acceleration)
            
            # Analyze speed variability (erratic driving)
            if len(recent_speeds) >= 5:
                speed_std = np.std(recent_speeds)
                speed_mean = np.mean(recent_speeds)
                
                if speed_mean > 0:
                    variability = speed_std / speed_mean
                    if variability > 0.3:  # High variability
                        self.risk_factors['erratic_driving'] = min(
                            self.risk_factors['erratic_driving'] + 0.05, 1.0
                        )
            
            # Analyze turning patterns
            if len(self.location_history) >= 3:
                self._analyze_turning_patterns()
            
            # Decay risk factors over time
            self._decay_risk_factors()
            
        except Exception as e:
            self.logger.error(f"Driving pattern analysis error: {e}")

    def _analyze_turning_patterns(self):
        """Analyze turning patterns for sharp turns"""
        try:
            recent_locations = list(self.location_history)[-5:]
            
            if len(recent_locations) < 3:
                return
            
            headings = [loc['heading'] for loc in recent_locations]
            
            # Calculate heading changes
            heading_changes = []
            for i in range(1, len(headings)):
                change = abs(headings[i] - headings[i-1])
                # Handle wraparound (359° to 1°)
                if change > 180:
                    change = 360 - change
                heading_changes.append(change)
            
            # Check for sharp turns
            if heading_changes:
                max_turn = max(heading_changes)
                if max_turn > 45:  # Sharp turn threshold
                    turn_severity = min(max_turn / 90, 1.0)  # Normalize to 0-1
                    self.risk_factors['sharp_turns'] = min(
                        self.risk_factors['sharp_turns'] + turn_severity * 0.1, 1.0
                    )
                    self._log_violation('sharp_turn', max_turn)
            
        except Exception as e:
            self.logger.error(f"Turning pattern analysis error: {e}")

    def _decay_risk_factors(self):
        """Gradually decay risk factors over time"""
        decay_rate = 0.02  # 2% decay per iteration
        
        for factor in self.risk_factors:
            self.risk_factors[factor] = max(0, self.risk_factors[factor] - decay_rate)

    def _check_violations(self):
        """Check for traffic rule violations"""
        try:
            current_time = time.time()
            
            # Speed limit violation
            if self.current_speed > self.current_speed_limit + self.speed_limit_buffer:
                excess_speed = self.current_speed - self.current_speed_limit
                speed_ratio = excess_speed / self.current_speed_limit
                
                self.risk_factors['speeding'] = min(speed_ratio * 2, 1.0)
                
                self._log_violation('speeding', {
                    'current_speed': self.current_speed,
                    'speed_limit': self.current_speed_limit,
                    'excess': excess_speed
                })
            else:
                # Gradually reduce speeding risk when not speeding
                self.risk_factors['speeding'] = max(0, self.risk_factors['speeding'] - 0.05)
            
        except Exception as e:
            self.logger.error(f"Violation check error: {e}")

    def _log_violation(self, violation_type, data):
        """Log a traffic violation"""
        violation = {
            'type': violation_type,
            'timestamp': time.time(),
            'location': self.current_location.copy(),
            'speed': self.current_speed,
            'speed_limit': self.current_speed_limit,
            'data': data
        }
        
        self.violations.append(violation)
        
        # Log severe violations
        if violation_type in ['harsh_braking', 'harsh_acceleration']:
            self.logger.warning(f"Harsh maneuver detected: {violation_type} = {data:.2f}")
        elif violation_type == 'speeding':
            excess = data['excess'] if isinstance(data, dict) else data
            self.logger.warning(f"Speed violation: {excess:.1f} km/h over limit")

    def _calculate_driving_risk(self):
        """Calculate overall driving risk score"""
        try:
            # Weight different risk factors
            weights = {
                'speeding': 0.3,
                'harsh_acceleration': 0.2,
                'harsh_braking': 0.25,
                'sharp_turns': 0.15,
                'erratic_driving': 0.1
            }
            
            # Calculate weighted risk score
            total_risk = sum(
                self.risk_factors[factor] * weights[factor]
                for factor in weights
            )
            
            self.driving_risk_score = min(total_risk, 1.0)
            
        except Exception as e:
            self.logger.error(f"Risk calculation error: {e}")
            self.driving_risk_score = 0.0

    def get_monitoring_results(self):
        """Get current monitoring results"""
        return {
            'current_speed': self.current_speed,
            'speed_limit': self.current_speed_limit,
            'speed_limit_source': self.speed_limit_source,
            'location': self.current_location.copy(),
            'heading': self.current_heading,
            'risk_score': self.driving_risk_score,
            'risk_factors': self.risk_factors.copy(),
            'recent_violations': list(self.violations)[-10:],  # Last 10 violations
            'is_speeding': self.current_speed > self.current_speed_limit + self.speed_limit_buffer,
            'timestamp': time.time()
        }

    def get_risk_score(self):
        """Get driving-based risk score (0-1)"""
        return self.driving_risk_score

    def get_violation_summary(self, time_window_hours=24):
        """Get violation summary for specified time window"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (time_window_hours * 3600)
            
            # Filter violations by time window
            recent_violations = [
                v for v in self.violations 
                if v['timestamp'] > cutoff_time
            ]
            
            # Count violations by type
            violation_counts = {}
            for violation in recent_violations:
                v_type = violation['type']
                violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
            
            return {
                'total_violations': len(recent_violations),
                'violation_counts': violation_counts,
                'time_window_hours': time_window_hours,
                'violations': recent_violations
            }
            
        except Exception as e:
            self.logger.error(f"Violation summary error: {e}")
            return {'total_violations': 0, 'violation_counts': {}}

    def export_driving_data(self, filename=None):
        """Export driving data to JSON file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"driving_data_{timestamp}.json"
        
        try:
            data = {
                'speed_history': list(self.speed_history),
                'location_history': list(self.location_history),
                'violations': list(self.violations),
                'risk_factors': self.risk_factors,
                'export_timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Driving data exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Data export error: {e}")
            return None

    def reset_monitoring_data(self):
        """Reset all monitoring data"""
        self.speed_history.clear()
        self.location_history.clear()
        self.violations.clear()
        self.risk_factors = {key: 0.0 for key in self.risk_factors}
        self.driving_risk_score = 0.0
        
        self.logger.info("Monitoring data reset")


def test_speed_monitor():
    """Test the speed monitor"""
    monitor = SpeedMonitor()
    
    print("Starting speed monitoring test...")
    
    if not monitor.start_monitoring():
        print("Failed to start monitoring")
        return
    
    try:
        # Monitor for 60 seconds
        for i in range(60):
            time.sleep(1)
            results = monitor.get_monitoring_results()
            
            print(f"Time: {i+1}s | "
                  f"Speed: {results['current_speed']:.1f} km/h | "
                  f"Limit: {results['speed_limit']} km/h | "
                  f"Risk: {results['risk_score']:.3f} | "
                  f"Violations: {len(results['recent_violations'])}")
            
            # Show risk factors if any are significant
            significant_risks = {
                k: v for k, v in results['risk_factors'].items() 
                if v > 0.1
            }
            if significant_risks:
                print(f"  Risk factors: {significant_risks}")
            
            # Show recent violations
            if results['recent_violations']:
                latest_violation = results['recent_violations'][-1]
                print(f"  Latest violation: {latest_violation['type']}")
    
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    
    finally:
        monitor.stop_monitoring()
        
        # Export data
        filename = monitor.export_driving_data()
        if filename:
            print(f"Driving data exported to {filename}")
        
        # Show violation summary
        summary = monitor.get_violation_summary()
        print(f"Violation Summary: {summary['total_violations']} total violations")
        print(f"By type: {summary['violation_counts']}")


if __name__ == "__main__":
    test_speed_monitor()