"""
Audio Analysis Module for NeuroDrive
Monitors cabin audio for distraction indicators
"""

import numpy as np
import pyaudio
import threading
import time
from collections import deque
import yaml
import librosa
import soundfile as sf
from scipy import signal
import logging

class AudioAnalyzer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.audio_config = self.config['audio']
        self.sample_rate = self.audio_config['sample_rate']
        self.chunk_size = self.audio_config['chunk_size']
        self.noise_threshold = self.audio_config['noise_threshold']
        self.distraction_threshold = self.audio_config['distraction_threshold']
        
        # Audio processing
        self.audio_buffer = deque(maxlen=100)  # Store last 100 chunks
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Analysis results
        self.current_noise_level = 0.0
        self.current_distraction_score = 0.0
        self.detected_events = []
        
        # Feature extraction
        self.feature_window = deque(maxlen=50)  # 50 chunks for analysis
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_audio(self):
        """Initialize PyAudio stream"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.logger.info("Audio stream initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback for real-time processing"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.append(audio_data)
        
        return (None, pyaudio.paContinue)

    def start_monitoring(self):
        """Start audio monitoring in separate thread"""
        if not self.initialize_audio():
            return False
        
        self.is_recording = True
        self.audio_stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Audio monitoring started")
        return True

    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.is_recording = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        self.logger.info("Audio monitoring stopped")

    def _process_audio_loop(self):
        """Main audio processing loop"""
        while self.is_recording:
            try:
                if len(self.audio_buffer) > 0:
                    # Get latest audio chunk
                    audio_chunk = self.audio_buffer[-1]
                    
                    # Process audio chunk
                    features = self._extract_features(audio_chunk)
                    self.feature_window.append(features)
                    
                    # Analyze for distraction indicators
                    self._analyze_distraction_indicators()
                    
                time.sleep(0.01)  # 10ms processing interval
                
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")

    def _extract_features(self, audio_chunk):
        """Extract audio features for analysis"""
        features = {}
        
        try:
            # Basic audio metrics
            features['rms'] = np.sqrt(np.mean(audio_chunk**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_chunk)[0])
            
            # Spectral features
            if len(audio_chunk) >= 512:  # Minimum length for spectral analysis
                # Spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(
                    y=audio_chunk, sr=self.sample_rate)[0]
                features['spectral_centroid'] = np.mean(spectral_centroids)
                
                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio_chunk, sr=self.sample_rate)[0]
                features['spectral_rolloff'] = np.mean(spectral_rolloff)
                
                # MFCC features (first 13 coefficients)
                mfccs = librosa.feature.mfcc(
                    y=audio_chunk, sr=self.sample_rate, n_mfcc=13)
                features['mfcc'] = np.mean(mfccs, axis=1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return {'rms': 0.0, 'zero_crossing_rate': 0.0}

    def _analyze_distraction_indicators(self):
        """Analyze audio for distraction indicators"""
        if len(self.feature_window) < 10:  # Need minimum samples
            return
        
        try:
            # Get recent features
            recent_features = list(self.feature_window)[-10:]
            
            # Calculate noise level
            rms_values = [f.get('rms', 0) for f in recent_features]
            self.current_noise_level = np.mean(rms_values)
            
            # Detect sudden volume changes (arguments, shouting)
            rms_std = np.std(rms_values)
            volume_variability = rms_std / (np.mean(rms_values) + 1e-6)
            
            # Detect high-frequency content (phone ringing, music)
            spectral_centroids = [f.get('spectral_centroid', 0) for f in recent_features]
            high_freq_content = np.mean(spectral_centroids) / 4000.0  # Normalize
            
            # Detect speech patterns vs noise
            zcr_values = [f.get('zero_crossing_rate', 0) for f in recent_features]
            speech_indicator = np.mean(zcr_values)
            
            # Combine indicators for distraction score
            distraction_factors = {
                'high_volume': min(self.current_noise_level * 2, 1.0),
                'volume_variability': min(volume_variability * 3, 1.0),
                'high_frequency': min(high_freq_content, 1.0),
                'speech_activity': min(speech_indicator * 5, 1.0)
            }
            
            # Weighted distraction score
            self.current_distraction_score = (
                distraction_factors['high_volume'] * 0.3 +
                distraction_factors['volume_variability'] * 0.3 +
                distraction_factors['high_frequency'] * 0.2 +
                distraction_factors['speech_activity'] * 0.2
            )
            
            # Detect specific events
            self._detect_audio_events(distraction_factors)
            
        except Exception as e:
            self.logger.error(f"Distraction analysis error: {e}")

    def _detect_audio_events(self, factors):
        """Detect specific audio events"""
        current_time = time.time()
        
        # Clear old events (older than 5 seconds)
        self.detected_events = [
            event for event in self.detected_events 
            if current_time - event['timestamp'] < 5.0
        ]
        
        # Phone ringing detection
        if (factors['high_frequency'] > 0.7 and 
            factors['volume_variability'] < 0.3):
            self._add_event('phone_ringing', 0.8, current_time)
        
        # Argument/shouting detection
        if (factors['high_volume'] > 0.6 and 
            factors['volume_variability'] > 0.5):
            self._add_event('argument', 0.7, current_time)
        
        # Music detection
        if (factors['high_frequency'] > 0.5 and 
            factors['speech_activity'] < 0.3 and
            factors['high_volume'] > 0.4):
            self._add_event('loud_music', 0.6, current_time)

    def _add_event(self, event_type, confidence, timestamp):
        """Add detected event"""
        # Avoid duplicate events within 2 seconds
        recent_events = [
            e for e in self.detected_events 
            if e['type'] == event_type and timestamp - e['timestamp'] < 2.0
        ]
        
        if not recent_events:
            event = {
                'type': event_type,
                'confidence': confidence,
                'timestamp': timestamp,
                'noise_level': self.current_noise_level
            }
            self.detected_events.append(event)
            self.logger.info(f"Audio event detected: {event_type} (confidence: {confidence:.2f})")

    def get_analysis_results(self):
        """Get current audio analysis results"""
        return {
            'noise_level': self.current_noise_level,
            'distraction_score': self.current_distraction_score,
            'is_distracted': self.current_distraction_score > self.distraction_threshold,
            'is_noisy': self.current_noise_level > self.noise_threshold,
            'detected_events': self.detected_events.copy(),
            'timestamp': time.time()
        }

    def get_risk_score(self):
        """Calculate audio-based risk score (0-1)"""
        base_risk = min(self.current_distraction_score, 1.0)
        
        # Add risk from recent events
        current_time = time.time()
        recent_events = [
            e for e in self.detected_events 
            if current_time - e['timestamp'] < 3.0
        ]
        
        event_risk = 0.0
        if recent_events:
            event_risk = max([e['confidence'] for e in recent_events]) * 0.5
        
        total_risk = min(base_risk + event_risk, 1.0)
        return total_risk

# Test function
def test_audio_analyzer():
    """Test the audio analyzer"""
    analyzer = AudioAnalyzer()
    
    print("Starting audio monitoring test...")
    
    if analyzer.start_monitoring():
        try:
            # Monitor for 30 seconds
            for i in range(30):
                time.sleep(1)
                results = analyzer.get_analysis_results()
                risk_score = analyzer.get_risk_score()
                
                print(f"Time: {i+1}s | "
                      f"Noise: {results['noise_level']:.3f} | "
                      f"Distraction: {results['distraction_score']:.3f} | "
                      f"Risk: {risk_score:.3f}")
                
                if results['detected_events']:
                    latest_event = results['detected_events'][-1]
                    print(f"  Event: {latest_event['type']} "
                          f"(confidence: {latest_event['confidence']:.2f})")
        
        except KeyboardInterrupt:
            print("\nStopping audio monitoring...")
        
        finally:
            analyzer.stop_monitoring()
    
    else:
        print("Failed to start audio monitoring")

if __name__ == "__main__":
    test_audio_analyzer()