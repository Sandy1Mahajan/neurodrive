"""
Weather Detection Module for NeuroDrive
Detects weather conditions from camera feed and external sensors
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import yaml
import logging
import time
import requests
from collections import deque
import threading

class WeatherDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.env_config = self.config['environment']
        self.weather_threshold = self.env_config['weather_confidence_threshold']
        self.visibility_threshold = self.env_config['visibility_threshold']
        
        # Weather conditions
        self.weather_classes = [
            'clear', 'cloudy', 'fog', 'rain', 'snow', 'storm'
        ]
        
        # Detection results
        self.current_weather = 'clear'
        self.weather_confidence = 1.0
        self.visibility_meters = 1000
        self.weather_risk_score = 0.0
        
        # Frame processing
        self.frame_buffer = deque(maxlen=30)  # 1 second at 30fps
        self.weather_history = deque(maxlen=100)
        
        # Models and processors
        self.cnn_model = None
        self.transform = self._get_transform()
        
        # External API (optional)
        self.api_key = None  # Set your weather API key
        self.last_api_call = 0
        self.api_interval = 300  # 5 minutes
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def initialize_model(self):
        """Initialize weather detection CNN model"""
        try:
            # Create simple weather detection CNN
            self.cnn_model = WeatherCNN(num_classes=len(self.weather_classes))
            
            # In production, load pretrained weights
            # self.cnn_model.load_state_dict(torch.load('weather_model.pth'))
            
            self.cnn_model.eval()
            
            self.logger.info("Weather detection model initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize weather model: {e}")
            return False

    def detect_weather_from_image(self, frame):
        """Detect weather conditions from camera image"""
        try:
            # Add frame to buffer for temporal analysis
            self.frame_buffer.append(frame.copy())
            
            # Method 1: CNN-based weather classification
            cnn_weather = self._classify_weather_cnn(frame)
            
            # Method 2: Classical computer vision features
            cv_weather = self._analyze_weather_features(frame)
            
            # Method 3: Visibility analysis
            visibility = self._estimate_visibility(frame)
            
            # Combine methods
            final_weather = self._combine_weather_predictions(cnn_weather, cv_weather)
            
            # Update state
            self.current_weather = final_weather['class']
            self.weather_confidence = final_weather['confidence']
            self.visibility_meters = visibility
            self.weather_risk_score = self._calculate_weather_risk()
            
            # Store in history
            weather_data = {
                'timestamp': time.time(),
                'weather': self.current_weather,
                'confidence': self.weather_confidence,
                'visibility': self.visibility_meters,
                'risk_score': self.weather_risk_score
            }
            self.weather_history.append(weather_data)
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Weather detection error: {e}")
            return self._get_default_weather()

    def _classify_weather_cnn(self, frame):
        """Classify weather using CNN"""
        try:
            if self.cnn_model is None:
                return {'class': 'clear', 'confidence': 0.5, 'probabilities': {}}
            
            # Preprocess image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.cnn_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Get predictions
            class_probabilities = {
                self.weather_classes[i]: float(probabilities[i])
                for i in range(len(self.weather_classes))
            }
            
            # Get top prediction
            max_idx = torch.argmax(probabilities)
            predicted_class = self.weather_classes[max_idx]
            confidence = float(probabilities[max_idx])
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities
            }
            
        except Exception as e:
            self.logger.error(f"CNN weather classification error: {e}")
            return {'class': 'clear', 'confidence': 0.5, 'probabilities': {}}

    def _analyze_weather_features(self, frame):
        """Analyze weather using classical computer vision"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Feature extraction
            features = {}
            
            # 1. Brightness analysis
            features['brightness'] = np.mean(gray)
            
            # 2. Contrast analysis
            features['contrast'] = np.std(gray)
            
            # 3. Edge density (visibility indicator)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges) / (frame.shape[0] * frame.shape[1])
            
            # 4. Color distribution
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            features['saturation'] = np.mean(hsv[:,:,1])
            features['value'] = np.mean(hsv[:,:,2])
            
            # 5. Texture analysis (LBP-like)
            features['texture'] = self._calculate_texture_metric(gray)
            
            # 6. Motion blur detection (rain/snow)
            features['motion_blur'] = self._detect_motion_blur(gray)
            
            # Rule-based classification
            weather_prediction = self._classify_by_rules(features)
            
            return weather_prediction
            
        except Exception as e:
            self.logger.error(f"Classical weather analysis error: {e}")
            return {'class': 'clear', 'confidence': 0.5}

    def _calculate_texture_metric(self, gray):
        """Calculate texture metric for weather analysis"""
        # Simple texture measure using local binary pattern concept
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        return np.std(texture_response)

    def _detect_motion_blur(self, gray):
        """Detect motion blur (indicator of rain/snow)"""
        # Variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def _classify_by_rules(self, features):
        """Rule-based weather classification"""
        # Initialize scores
        scores = {weather: 0.0 for weather in self.weather_classes}
        
        # Rule 1: Low brightness + low contrast = fog
        if features['brightness'] < 80 and features['contrast'] < 30:
            scores['fog'] += 0.7
        
        # Rule 2: Low edge density = fog/heavy rain
        if features['edge_density'] < 0.1:
            scores['fog'] += 0.5
            scores['rain'] += 0.3
        
        # Rule 3: High motion blur = rain/snow
        if features['motion_blur'] < 100:  # Low sharpness
            scores['rain'] += 0.4
            scores['snow'] += 0.2
        
        # Rule 4: Low saturation = overcast/fog
        if features['saturation'] < 50:
            scores['cloudy'] += 0.4
            scores['fog'] += 0.3
        
        # Rule 5: Very low brightness = storm/heavy rain
        if features['brightness'] < 50:
            scores['storm'] += 0.6
            scores['rain'] += 0.5
        
        # Rule 6: Normal conditions = clear
        if (features['brightness'] > 120 and 
            features['contrast'] > 40 and 
            features['edge_density'] > 0.15):
            scores['clear'] += 0.8
        
        # Get best prediction
        best_class = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_class], 1.0)
        
        # If all scores are low, default to clear
        if confidence < 0.3:
            best_class = 'clear'
            confidence = 0.5
        
        return {
            'class': best_class,
            'confidence': confidence,
            'scores': scores
        }

    def _estimate_visibility(self, frame):
        """Estimate visibility distance in meters"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Contrast-based visibility
            # Higher contrast = better visibility
            contrast = np.std(gray)
            contrast_visibility = min(contrast * 10, 1000)  # Scale to meters
            
            # Method 2: Edge-based visibility
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])
            edge_visibility = min(edge_density * 5000, 1000)
            
            # Method 3: Brightness-based visibility
            brightness = np.mean(gray)
            if brightness < 30:  # Very dark
                brightness_visibility = 50
            elif brightness < 80:  # Dim
                brightness_visibility = 200
            else:  # Normal
                brightness_visibility = 500
            
            # Combine estimates
            estimated_visibility = np.mean([
                contrast_visibility,
                edge_visibility,
                brightness_visibility
            ])
            
            return max(estimated_visibility, 10)  # Minimum 10m
            
        except Exception as e:
            self.logger.error(f"Visibility estimation error: {e}")
            return 500  # Default visibility

    def _combine_weather_predictions(self, cnn_pred, cv_pred):
        """Combine CNN and classical CV predictions"""
        try:
            # Weight the predictions
            cnn_weight = 0.7 if self.cnn_model else 0.0
            cv_weight = 1.0 - cnn_weight
            
            if cnn_weight > 0:
                # Use CNN prediction if available
                if cnn_pred['confidence'] > cv_pred['confidence']:
                    final_class = cnn_pred['class']
                    final_confidence = cnn_pred['confidence'] * cnn_weight + cv_pred['confidence'] * cv_weight
                else:
                    final_class = cv_pred['class']
                    final_confidence = cv_pred['confidence']
            else:
                # Use only CV prediction
                final_class = cv_pred['class']
                final_confidence = cv_pred['confidence']
            
            return {
                'class': final_class,
                'confidence': min(final_confidence, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction combination error: {e}")
            return {'class': 'clear', 'confidence': 0.5}

    def _calculate_weather_risk(self):
        """Calculate weather-based risk score (0-1)"""
        risk_scores = {
            'clear': 0.1,
            'cloudy': 0.3,
            'fog': 0.8,
            'rain': 0.6,
            'snow': 0.7,
            'storm': 0.9
        }
        
        base_risk = risk_scores.get(self.current_weather, 0.5)
        
        # Adjust based on visibility
        if self.visibility_meters < 50:
            visibility_risk = 0.9
        elif self.visibility_meters < 100:
            visibility_risk = 0.7
        elif self.visibility_meters < 200:
            visibility_risk = 0.5
        else:
            visibility_risk = 0.2
        
        # Combine risks
        combined_risk = (base_risk * 0.6) + (visibility_risk * 0.4)
        
        # Adjust by confidence
        final_risk = combined_risk * self.weather_confidence
        
        return min(final_risk, 1.0)

    def get_external_weather(self, lat=None, lon=None):
        """Get weather from external API (optional)"""
        try:
            if not self.api_key or not lat or not lon:
                return None
            
            current_time = time.time()
            if current_time - self.last_api_call < self.api_interval:
                return None  # Don't call API too frequently
            
            # Example using OpenWeatherMap API
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                weather_main = data['weather'][0]['main'].lower()
                description = data['weather'][0]['description']
                visibility = data.get('visibility', 10000) / 1000  # Convert to km
                
                self.last_api_call = current_time
                
                return {
                    'weather': weather_main,
                    'description': description,
                    'visibility_km': visibility,
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"External weather API error: {e}")
            return None

    def get_weather_results(self):
        """Get current weather detection results"""
        return {
            'weather': self.current_weather,
            'confidence': self.weather_confidence,
            'visibility_meters': self.visibility_meters,
            'risk_score': self.weather_risk_score,
            'is_poor_weather': self.weather_risk_score > 0.6,
            'timestamp': time.time()
        }

    def _get_default_weather(self):
        """Get default weather data"""
        return {
            'timestamp': time.time(),
            'weather': 'clear',
            'confidence': 0.5,
            'visibility': 500,
            'risk_score': 0.1
        }

    def draw_weather_info(self, frame):
        """Draw weather information on frame"""
        annotated_frame = frame.copy()
        
        # Draw weather info
        weather_text = f"Weather: {self.current_weather.upper()} ({self.weather_confidence:.2f})"
        cv2.putText(annotated_frame, weather_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        visibility_text = f"Visibility: {self.visibility_meters:.0f}m"
        cv2.putText(annotated_frame, visibility_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        risk_text = f"Weather Risk: {self.weather_risk_score:.2f}"
        risk_color = (0, 0, 255) if self.weather_risk_score > 0.6 else (0, 255, 0)
        cv2.putText(annotated_frame, risk_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        return annotated_frame


class WeatherCNN(nn.Module):
    """Simple CNN for weather classification"""
    
    def __init__(self, num_classes=6):
        super(WeatherCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_weather_detector():
    """Test the weather detector"""
    detector = WeatherDetector()
    
    if not detector.initialize_model():
        print("Failed to initialize weather model")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("Starting weather detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect weather
            weather_data = detector.detect_weather_from_image(frame)
            
            # Draw weather info
            annotated_frame = detector.draw_weather_info(frame)
            
            # Display results
            print(f"Weather: {weather_data['weather']}, "
                  f"Confidence: {weather_data['confidence']:.2f}, "
                  f"Visibility: {weather_data['visibility']:.0f}m, "
                  f"Risk: {weather_data['risk_score']:.2f}")
            
            cv2.imshow('Weather Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'weather_test_{int(time.time())}.jpg', annotated_frame)
                print("Frame saved")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_weather_detector()