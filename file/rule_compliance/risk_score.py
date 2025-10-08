"""
Risk Score Calculator for NeuroDrive
Combines all monitoring modules to calculate comprehensive driving risk score
"""

import numpy as np
import time
import yaml
import logging
from collections import deque
import threading

class RiskCalculator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Risk weights from config
        self.risk_weights = self.config['risk_weights']
        self.alert_thresholds = self.config['alerts']
        
        # Risk components
        self.risk_components = {
            'drowsiness': 0.0,
            'distraction': 0.0,
            'phone_use': 0.0,
            'alcohol': 0.0,
            'speed_violation': 0.0,
            'weather_risk': 0.0,
            'audio_distraction': 0.0,
            'behavioral_risk': 0.0
        }
        
        # Historical data for smoothing
        self.risk_history = deque(maxlen=50)  # 50 samples for smoothing
        self.component_history = {
            component: deque(maxlen=30) 
            for component in self.risk_components
        }
        
        # Current risk metrics
        self.current_risk_score = 0.0
        self.current_risk_level = 'normal'
        self.risk_trend = 'stable'  # 'increasing', 'decreasing', 'stable'
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # seconds between similar alerts
        
        # Risk factors analysis
        self.dominant_risk_factors = []
        self.risk_pattern = 'normal'  # 'aggressive', 'distracted', 'drowsy', 'normal'
        
        # Temporal analysis
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        
        # Smoothing parameters
        self.smoothing_factor = 0.3  # For exponential smoothing
        self.trend_window = 10  # Number of samples for trend analysis
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update_risk_components(self, module_results):
        """Update risk components from various monitoring modules"""
        try:
            current_time = time.time()
            
            # Driver monitoring risks
            if 'head_pose' in module_results:
                self.risk_components['distraction'] = module_results['head_pose'].get('distraction_score', 0.0)
            
            if 'face_eye' in module_results:
                self.risk_components['drowsiness'] = module_results['face_eye'].get('drowsiness_score', 0.0)
            
            if 'bio_auth' in module_results:
                # Continuous authentication failure indicates risk
                auth_risk = 1.0 - module_results['bio_auth'].get('confidence', 1.0)
                self.risk_components['behavioral_risk'] = auth_risk
            
            # In-cabin risks
            if 'object_detection' in module_results:
                phone_detected = module_results['object_detection'].get('phone_detected', False)
                phone_confidence = module_results['object_detection'].get('phone_confidence', 0.0)
                self.risk_components['phone_use'] = phone_confidence if phone_detected else 0.0
            
            if 'audio' in module_results:
                audio_risk = module_results['audio'].get('distraction_score', 0.0)
                self.risk_components['audio_distraction'] = audio_risk
            
            # Rule compliance risks
            if 'alcohol' in module_results:
                self.risk_components['alcohol'] = module_results['alcohol'].get('confidence', 0.0)
            
            if 'speed_monitoring' in module_results:
                self.risk_components['speed_violation'] = module_results['speed_monitoring'].get('risk_score', 0.0)
            
            # Environmental risks
            if 'weather' in module_results:
                self.risk_components['weather_risk'] = module_results['weather'].get('risk_score', 0.0)
            
            # Store component history for trend analysis
            for component, value in self.risk_components.items():
                self.component_history[component].append({
                    'value': value,
                    'timestamp': current_time
                })
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"Risk component update error: {e}")

    def calculate_comprehensive_risk(self):
        """Calculate comprehensive risk score using weighted fusion"""
        try:
            # Calculate base weighted risk
            weighted_risk = 0.0
            total_weight = 0.0
            
            # Apply weights to each component
            for component, value in self.risk_components.items():
                if component in self.risk_weights:
                    weight = self.risk_weights[component]
                    weighted_risk += value * weight
                    total_weight += weight
                elif component == 'audio_distraction':
                    # Use phone_use weight for audio
                    weight = self.risk_weights.get('phone_use', 0.2)
                    weighted_risk += value * weight
                    total_weight += weight
                elif component == 'behavioral_risk':
                    # Distribute behavioral risk
                    weight = 0.1
                    weighted_risk += value * weight
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                base_risk = weighted_risk / total_weight
            else:
                base_risk = 0.0
            
            # Apply temporal adjustments
            temporal_adjusted_risk = self._apply_temporal_adjustments(base_risk)
            
            # Apply smoothing
            smoothed_risk = self._apply_smoothing(temporal_adjusted_risk)
            
            # Apply contextual modifiers
            final_risk = self._apply_contextual_modifiers(smoothed_risk)
            
            # Ensure risk is within bounds
            self.current_risk_score = max(0.0, min(1.0, final_risk))
            
            # Update risk history
            self.risk_history.append({
                'risk_score': self.current_risk_score,
                'timestamp': time.time(),
                'components': self.risk_components.copy()
            })
            
            # Analyze risk patterns and trends
            self._analyze_risk_patterns()
            self._calculate_risk_trend()
            
            # Update risk level
            self.current_risk_level = self._get_risk_level(self.current_risk_score)
            
            return self.current_risk_score
            
        except Exception as e:
            self.logger.error(f"Risk calculation error: {e}")
            return 0.0

    def _apply_temporal_adjustments(self, base_risk):
        """Apply temporal adjustments based on time patterns"""
        try:
            current_time = time.time()
            session_duration = current_time - self.session_start_time
            
            # Fatigue factor (increases risk over long sessions)
            if session_duration > 7200:  # 2 hours
                fatigue_factor = min((session_duration - 7200) / 3600, 0.3)  # Max 30% increase
                base_risk += fatigue_factor
            
            # Time of day factor
            hour = time.localtime().tm_hour
            if hour < 6 or hour > 22:  # Night driving
                base_risk *= 1.2  # 20% increase for night driving
            
            # Rapid risk escalation detection
            if len(self.risk_history) >= 5:
                recent_risks = [r['risk_score'] for r in list(self.risk_history)[-5:]]
                risk_acceleration = np.diff(recent_risks)
                
                if np.mean(risk_acceleration) > 0.1:  # Rapidly increasing risk
                    base_risk *= 1.1  # 10% penalty for rapid escalation
            
            return base_risk
            
        except Exception as e:
            self.logger.error(f"Temporal adjustment error: {e}")
            return base_risk

    def _apply_smoothing(self, risk_score):
        """Apply exponential smoothing to reduce noise"""
        try:
            if len(self.risk_history) == 0:
                return risk_score
            
            # Exponential smoothing
            previous_risk = self.risk_history[-1]['risk_score']
            smoothed_risk = (self.smoothing_factor * risk_score + 
                           (1 - self.smoothing_factor) * previous_risk)
            
            return smoothed_risk
            
        except Exception as e:
            self.logger.error(f"Smoothing error: {e}")
            return risk_score

    def _apply_contextual_modifiers(self, risk_score):
        """Apply contextual modifiers based on driving situation"""
        try:
            # Multiple high-risk factors amplify each other
            high_risk_components = [
                component for component, value in self.risk_components.items()
                if value > 0.6
            ]
            
            if len(high_risk_components) >= 2:
                # Multiple high risks - apply amplification
                amplification = 1 + (len(high_risk_components) - 1) * 0.1
                risk_score *= amplification
            
            # Critical combinations
            if (self.risk_components.get('alcohol', 0) > 0.5 and 
                self.risk_components.get('speed_violation', 0) > 0.5):
                risk_score = min(risk_score * 1.5, 1.0)  # Alcohol + speeding is very dangerous
            
            if (self.risk_components.get('drowsiness', 0) > 0.7 and 
                self.risk_components.get('weather_risk', 0) > 0.6):
                risk_score = min(risk_score * 1.3, 1.0)  # Drowsiness + bad weather
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Contextual modifier error: {e}")
            return risk_score

    def _analyze_risk_patterns(self):
        """Analyze risk patterns to identify driving behavior"""
        try:
            # Identify dominant risk factors
            self.dominant_risk_factors = [
                component for component, value in self.risk_components.items()
                if value > 0.4
            ]
            
            # Classify risk pattern
            if self.risk_components.get('drowsiness', 0) > 0.6:
                self.risk_pattern = 'drowsy'
            elif (self.risk_components.get('distraction', 0) > 0.5 or 
                  self.risk_components.get('phone_use', 0) > 0.5):
                self.risk_pattern = 'distracted'
            elif (self.risk_components.get('speed_violation', 0) > 0.6 or
                  any(self.risk_components.get(comp, 0) > 0.7 for comp in ['harsh_acceleration', 'harsh_braking'])):
                self.risk_pattern = 'aggressive'
            elif self.risk_components.get('alcohol', 0) > 0.5:
                self.risk_pattern = 'impaired'
            else:
                self.risk_pattern = 'normal'
            
        except Exception as e:
            self.logger.error(f"Risk pattern analysis error: {e}")

    def _calculate_risk_trend(self):
        """Calculate risk trend over recent history"""
        try:
            if len(self.risk_history) < self.trend_window:
                self.risk_trend = 'stable'
                return
            
            recent_risks = [r['risk_score'] for r in list(self.risk_history)[-self.trend_window:]]
            
            # Calculate trend using linear regression
            x = np.arange(len(recent_risks))
            slope = np.polyfit(x, recent_risks, 1)[0]
            
            if slope > 0.02:
                self.risk_trend = 'increasing'
            elif slope < -0.02:
                self.risk_trend = 'decreasing'
            else:
                self.risk_trend = 'stable'
            
        except Exception as e:
            self.logger.error(f"Risk trend calculation error: {e}")
            self.risk_trend = 'stable'

    def _get_risk_level(self, risk_score):
        """Convert risk score to categorical level"""
        if risk_score >= self.alert_thresholds['critical_risk']:
            return 'critical'
        elif risk_score >= self.alert_thresholds['high_risk']:
            return 'high'
        elif risk_score >= self.alert_thresholds['medium_risk']:
            return 'medium'
        elif risk_score >= self.alert_thresholds['low_risk']:
            return 'low'
        else:
            return 'normal'

    def generate_alerts(self):
        """Generate appropriate alerts based on risk analysis"""
        try:
            current_time = time.time()
            new_alerts = []
            
            # Avoid alert spam
            if current_time - self.last_alert_time < self.alert_cooldown:
                return []
            
            # Critical risk alert
            if self.current_risk_level == 'critical':
                alert = {
                    'type': 'critical_risk',
                    'level': 'critical',
                    'message': 'CRITICAL: Immediate action required - pull over safely',
                    'risk_score': self.current_risk_score,
                    'dominant_factors': self.dominant_risk_factors,
                    'timestamp': current_time
                }
                new_alerts.append(alert)
            
            # High risk alert
            elif self.current_risk_level == 'high':
                alert = {
                    'type': 'high_risk',
                    'level': 'high',
                    'message': f'HIGH RISK: {self.risk_pattern.capitalize()} driving detected',
                    'risk_score': self.current_risk_score,
                    'dominant_factors': self.dominant_risk_factors,
                    'timestamp': current_time
                }
                new_alerts.append(alert)
            
            # Specific pattern alerts
            if self.risk_pattern == 'drowsy' and self.risk_components.get('drowsiness', 0) > 0.7:
                alert = {
                    'type': 'drowsiness_alert',
                    'level': 'high',
                    'message': 'DROWSINESS DETECTED: Take a break immediately',
                    'risk_score': self.risk_components['drowsiness'],
                    'timestamp': current_time
                }
                new_alerts.append(alert)
            
            if self.risk_pattern == 'impaired' and self.risk_components.get('alcohol', 0) > 0.8:
                alert = {
                    'type': 'impairment_alert',
                    'level': 'critical',
                    'message': 'ALCOHOL IMPAIRMENT: Stop driving immediately',
                    'risk_score': self.risk_components['alcohol'],
                    'timestamp': current_time
                }
                new_alerts.append(alert)
            
            # Trend-based alerts
            if self.risk_trend == 'increasing' and self.current_risk_score > 0.6:
                alert = {
                    'type': 'escalating_risk',
                    'level': 'medium',
                    'message': 'Risk levels increasing - adjust driving behavior',
                    'risk_score': self.current_risk_score,
                    'trend': self.risk_trend,
                    'timestamp': current_time
                }
                new_alerts.append(alert)
            
            # Update alert management
            if new_alerts:
                self.active_alerts.extend(new_alerts)
                self.alert_history.extend(new_alerts)
                self.last_alert_time = current_time
            
            # Clean up old active alerts (older than 30 seconds)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if current_time - alert['timestamp'] < 30
            ]
            
            return new_alerts
            
        except Exception as e:
            self.logger.error(f"Alert generation error: {e}")
            return []

    def get_risk_analysis(self):
        """Get comprehensive risk analysis results"""
        return {
            'risk_score': self.current_risk_score,
            'risk_level': self.current_risk_level,
            'risk_trend': self.risk_trend,
            'risk_pattern': self.risk_pattern,
            'components': self.risk_components.copy(),
            'dominant_factors': self.dominant_risk_factors.copy(),
            'active_alerts': self.active_alerts.copy(),
            'session_duration': time.time() - self.session_start_time,
            'timestamp': time.time()
        }

    def get_risk_breakdown(self):
        """Get detailed breakdown of risk components"""
        total_weighted_risk = sum(
            self.risk_components.get(comp, 0) * self.risk_weights.get(comp, 0)
            for comp in self.risk_weights
        )
        
        breakdown = {}
        for component, value in self.risk_components.items():
            weight = self.risk_weights.get(component, 0.1)
            contribution = (value * weight) / max(total_weighted_risk, 0.001)
            
            breakdown[component] = {
                'current_value': value,
                'weight': weight,
                'contribution_percentage': contribution * 100,
                'risk_level': self._get_risk_level(value)
            }
        
        return breakdown

    def get_historical_analysis(self, time_window_minutes=30):
        """Get historical risk analysis for specified time window"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (time_window_minutes * 60)
            
            # Filter historical data
            recent_history = [
                entry for entry in self.risk_history
                if entry['timestamp'] > cutoff_time
            ]
            
            if not recent_history:
                return {
                    'avg_risk': 0.0,
                    'max_risk': 0.0,
                    'risk_episodes': 0,
                    'time_above_threshold': 0.0
                }
            
            # Calculate statistics
            risk_scores = [entry['risk_score'] for entry in recent_history]
            
            avg_risk = np.mean(risk_scores)
            max_risk = np.max(risk_scores)
            
            # Count risk episodes (continuous periods above medium threshold)
            risk_episodes = 0
            in_episode = False
            medium_threshold = self.alert_thresholds['medium_risk']
            
            for score in risk_scores:
                if score > medium_threshold:
                    if not in_episode:
                        risk_episodes += 1
                        in_episode = True
                else:
                    in_episode = False
            
            # Calculate time above threshold
            time_above_threshold = sum(1 for score in risk_scores if score > medium_threshold)
            time_above_threshold_percent = (time_above_threshold / len(risk_scores)) * 100
            
            return {
                'avg_risk': avg_risk,
                'max_risk': max_risk,
                'risk_episodes': risk_episodes,
                'time_above_threshold_percent': time_above_threshold_percent,
                'total_samples': len(recent_history),
                'time_window_minutes': time_window_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Historical analysis error: {e}")
            return {'avg_risk': 0.0, 'max_risk': 0.0, 'risk_episodes': 0}

    def reset_session(self):
        """Reset risk calculator for new driving session"""
        self.risk_components = {component: 0.0 for component in self.risk_components}
        self.risk_history.clear()
        
        for component_hist in self.component_history.values():
            component_hist.clear()
        
        self.active_alerts.clear()
        self.current_risk_score = 0.0
        self.current_risk_level = 'normal'
        self.risk_trend = 'stable'
        self.risk_pattern = 'normal'
        self.session_start_time = time.time()
        
        self.logger.info("Risk calculator session reset")

    def export_risk_data(self, filename=None):
        """Export risk analysis data"""
        if not filename:
            timestamp = int(time.time())
            filename = f"risk_analysis_{timestamp}.json"
        
        try:
            import json
            
            data = {
                'session_info': {
                    'start_time': self.session_start_time,
                    'duration': time.time() - self.session_start_time,
                    'export_time': time.time()
                },
                'current_state': {
                    'risk_score': self.current_risk_score,
                    'risk_level': self.current_risk_level,
                    'risk_pattern': self.risk_pattern,
                    'risk_trend': self.risk_trend,
                    'components': self.risk_components
                },
                'risk_history': list(self.risk_history),
                'alert_history': list(self.alert_history),
                'risk_breakdown': self.get_risk_breakdown(),
                'historical_analysis': self.get_historical_analysis()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Risk data exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Risk data export error: {e}")
            return None

    def get_recommendations(self):
        """Get personalized recommendations based on risk analysis"""
        recommendations = []
        
        try:
            # Based on dominant risk factors
            if 'drowsiness' in self.dominant_risk_factors:
                recommendations.append({
                    'type': 'immediate',
                    'priority': 'high',
                    'message': 'Take a 15-20 minute break to combat drowsiness',
                    'action': 'rest_break'
                })
            
            if 'distraction' in self.dominant_risk_factors:
                recommendations.append({
                    'type': 'behavioral',
                    'priority': 'medium',
                    'message': 'Focus on the road - avoid secondary tasks',
                    'action': 'focus_driving'
                })
            
            if 'speed_violation' in self.dominant_risk_factors:
                recommendations.append({
                    'type': 'behavioral',
                    'priority': 'high',
                    'message': 'Reduce speed and maintain safe following distance',
                    'action': 'reduce_speed'
                })
            
            if 'weather_risk' in self.dominant_risk_factors:
                recommendations.append({
                    'type': 'environmental',
                    'priority': 'medium',
                    'message': 'Adjust driving for weather conditions - reduce speed and increase following distance',
                    'action': 'weather_adjustment'
                })
            
            # Based on risk pattern
            if self.risk_pattern == 'aggressive':
                recommendations.append({
                    'type': 'behavioral',
                    'priority': 'high',
                    'message': 'Adopt defensive driving - smooth acceleration and braking',
                    'action': 'defensive_driving'
                })
            
            # Based on session duration
            session_hours = (time.time() - self.session_start_time) / 3600
            if session_hours > 2:
                recommendations.append({
                    'type': 'immediate',
                    'priority': 'medium',
                    'message': f'Consider taking a break - driving for {session_hours:.1f} hours',
                    'action': 'break_suggestion'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return []


def test_risk_calculator():
    """Test the risk calculator with simulated data"""
    calculator = RiskCalculator()
    
    print("Starting risk calculator test...")
    
    # Simulate module results over time
    for i in range(60):  # 60 iterations
        # Simulate varying risk components
        simulated_results = {
            'head_pose': {
                'distraction_score': max(0, 0.3 + 0.4 * np.sin(i * 0.1) + np.random.normal(0, 0.1))
            },
            'face_eye': {
                'drowsiness_score': max(0, 0.2 + 0.3 * np.sin(i * 0.05) + np.random.normal(0, 0.1))
            },
            'audio': {
                'distraction_score': max(0, 0.1 + 0.2 * np.random.random())
            },
            'weather': {
                'risk_score': 0.3 if i > 30 else 0.1  # Weather gets worse
            },
            'speed_monitoring': {
                'risk_score': max(0, 0.2 + 0.5 * np.sin(i * 0.2) + np.random.normal(0, 0.1))
            },
            'alcohol': {
                'confidence': 0.05 + np.random.normal(0, 0.02)  # Low baseline
            }
        }
        
        # Update risk components
        calculator.update_risk_components(simulated_results)
        
        # Calculate comprehensive risk
        risk_score = calculator.calculate_comprehensive_risk()
        
        # Generate alerts
        alerts = calculator.generate_alerts()
        
        # Get analysis
        analysis = calculator.get_risk_analysis()
        
        print(f"Time: {i+1}s | "
              f"Risk: {risk_score:.3f} ({analysis['risk_level']}) | "
              f"Pattern: {analysis['risk_pattern']} | "
              f"Trend: {analysis['risk_trend']} | "
              f"Alerts: {len(alerts)}")
        
        if analysis['dominant_factors']:
            print(f"  Dominant factors: {analysis['dominant_factors']}")
        
        if alerts:
            for alert in alerts:
                print(f"  ALERT: {alert['message']}")
        
        time.sleep(0.1)  # Simulate real-time processing
    
    # Show final analysis
    print("\n=== SESSION SUMMARY ===")
    historical = calculator.get_historical_analysis()
    print(f"Average Risk: {historical['avg_risk']:.3f}")
    print(f"Maximum Risk: {historical['max_risk']:.3f}")
    print(f"Risk Episodes: {historical['risk_episodes']}")
    print(f"Time Above Threshold: {historical['time_above_threshold_percent']:.1f}%")
    
    # Show recommendations
    recommendations = calculator.get_recommendations()
    if recommendations:
        print("\n=== RECOMMENDATIONS ===")
        for rec in recommendations:
            print(f"[{rec['priority'].upper()}] {rec['message']}")
    
    # Export data
    filename = calculator.export_risk_data()
    if filename:
        print(f"\nRisk data exported to: {filename}")


if __name__ == "__main__":
    test_risk_calculator()