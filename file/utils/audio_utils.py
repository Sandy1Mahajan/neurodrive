"""
NeuroDrive Audio Processing Utilities
Handles microphone input, noise detection, and audio analysis
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import wave
import librosa
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
import webrtcvad

class AudioProcessor:
    """Real-time audio processing for cabin monitoring"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.audio_config = self.config['audio']
        
        # Audio parameters
        self.sample_rate = self.audio_config['sample_rate']
        self.chunk_size = self.audio_config['chunk_size']
        self.channels = self.audio_config['channels']
        self.device_index = self.audio_config.get('device_index', None)
        
        # Thresholds
        self.noise_threshold = self.audio_config['noise_threshold']
        self.distraction_threshold = self.audio_config['distraction_threshold']
        
        # Audio processing components
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        
        # Data storage
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10 seconds buffer
        
        # Processing thread
        self.process_thread = None
        self.processing = False
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Moderate aggressiveness
        
        # Event callbacks
        self.callbacks = {
            'noise_detected': [],
            'voice_detected': [],
            'distraction_detected': [],
            'silence_detected': []
        }
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_audio(self) -> bool:
        """Initialize audio stream"""
        try:
            # Find available audio devices
            if self.device_index is None:
                self.device_index = self._find_best_device()
            
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.logger.info(f"Audio initialized - Device: {self.device_index}, "
                           f"Rate: {self.sample_rate}Hz, Channels: {self.channels}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio initialization error: {e}")
            return False
    
    def _find_best_device(self) -> int:
        """Find best available input device"""
        device_count = self.pyaudio.get_device_count()
        best_device = 0
        
        for i in range(device_count):
            device_info = self.pyaudio.get_device_info_by_index(i)
            
            if (device_info['maxInputChannels'] > 0 and 
                device_info['defaultSampleRate'] >= self.sample_rate):
                
                self.logger.info(f"Available device {i}: {device_info['name']}")
                if 'USB' in device_info['name'] or 'Microphone' in device_info['name']:
                    best_device = i
        
        return best_device
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if self.recording:
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                pass  # Drop frame if queue is full
        
        return (None, pyaudio.paContinue)
    
    def start_recording(self):
        """Start audio recording and processing"""
        if not self.stream:
            if not self.initialize_audio():
                return False
        
        self.recording = True
        self.processing = True
        
        self.stream.start_stream()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.process_thread.start()
        
        self.logger.info("Audio recording started")
        return True
    
    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        self.processing = False
        
        if self.stream:
            self.stream.stop_stream()
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        self.logger.info("Audio recording stopped")
    
    def _process_audio(self):
        """Process audio data in background thread"""
        while self.processing:
            try:
                # Get audio data
                data = self.audio_queue.get(timeout=1.0)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process audio chunk
                self._analyze_audio_chunk(audio_data)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
    
    def _analyze_audio_chunk(self, audio_data: np.ndarray):
        """Analyze audio chunk for various features"""
        # Normalize audio
        if len(audio_data) == 0:
            return
        
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Calculate RMS (volume level)
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Voice Activity Detection
        audio_bytes = audio_data.tobytes()
        is_speech = False
        
        try:
            # VAD requires specific frame sizes
            frame_size = int(self.sample_rate * 0.01)  # 10ms frames
            if len(audio_data) >= frame_size:
                frame = audio_data[:frame_size].tobytes()
                is_speech = self.vad.is_speech(frame, self.sample_rate)
        except:
            pass
        
        # Frequency analysis
        frequencies, power_spectrum = self._frequency_analysis(audio_float)
        dominant_freq = frequencies[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0
        
        # Detect different audio events
        self._detect_audio_events(rms, is_speech, dominant_freq, audio_float)
    
    def _frequency_analysis(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform frequency analysis of audio"""
        if len(audio_data) < 2:
            return np.array([]), np.array([])
        
        # Apply window function
        windowed = audio_data * np.hanning(len(audio_data))
        
        # FFT
        fft_data = fft(windowed)
        frequencies = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Power spectrum (positive frequencies only)
        power_spectrum = np.abs(fft_data[:len(audio_data)//2])
        frequencies = frequencies[:len(audio_data)//2]
        
        return frequencies, power_spectrum
    
    def _detect_audio_events(self, rms: float, is_speech: bool, 
                           dominant_freq: float, audio_data: np.ndarray):
        """Detect various audio events"""
        current_time = time.time()
        
        # High noise level detection
        if rms > self.noise_threshold:
            self._trigger_callback('noise_detected', {
                'rms': rms,
                'timestamp': current_time,
                'dominant_frequency': dominant_freq
            })
        
        # Voice detection
        if is_speech:
            self._trigger_callback('voice_detected', {
                'rms': rms,
                'timestamp': current_time,
                'dominant_frequency': dominant_freq
            })
        
        # Distraction-level noise
        if rms > self.distraction_threshold:
            # Additional analysis for distraction type
            distraction_type = self._classify_distraction_sound(audio_data, dominant_freq)
            
            self._trigger_callback('distraction_detected', {
                'rms': rms,
                'timestamp': current_time,
                'distraction_type': distraction_type,
                'dominant_frequency': dominant_freq
            })
        
        # Silence detection
        if rms < 0.01 and not is_speech:
            self._trigger_callback('silence_detected', {
                'rms': rms,
                'timestamp': current_time
            })
    
    def _classify_distraction_sound(self, audio_data: np.ndarray, dominant_freq: float) -> str:
        """Classify the type of distracting sound"""
        # Simple classification based on frequency and patterns
        
        if dominant_freq < 100:
            return "engine_noise"
        elif 100 <= dominant_freq < 300:
            return "road_noise"
        elif 300 <= dominant_freq < 1000:
            return "conversation"
        elif 1000 <= dominant_freq < 4000:
            return "music_or_radio"
        elif dominant_freq >= 4000:
            # Check for specific patterns
            if self._detect_phone_ring(audio_data):
                return "phone_ring"
            elif self._detect_horn(audio_data):
                return "horn"
            else:
                return "high_frequency_noise"
        else:
            return "unknown"
    
    def _detect_phone_ring(self, audio_data: np.ndarray) -> bool:
        """Detect phone ring pattern"""
        # Simple pattern detection for repetitive high-frequency sounds
        if len(audio_data) < self.sample_rate:  # Need at least 1 second
            return False
        
        # Check for periodicity in the signal
        correlation = np.correlate(audio_data, audio_data, mode='full')
        correlation = correlation[correlation.size // 2:]
        
        # Look for peaks indicating repetitive pattern
        peaks, _ = signal.find_peaks(correlation, height=0.5 * np.max(correlation))
        
        if len(peaks) > 1:
            # Check if peaks are evenly spaced (ring pattern)
            intervals = np.diff(peaks)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                # Typical phone ring: 2-4 Hz (0.25-0.5 seconds)
                ring_freq = self.sample_rate / avg_interval
                return 2 <= ring_freq <= 4
        
        return False
    
    def _detect_horn(self, audio_data: np.ndarray) -> bool:
        """Detect car horn sound"""
        # Horn detection based on sustained high-amplitude sound
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        # Check for sustained high amplitude
        if rms > 0.3:  # High amplitude threshold
            # Check frequency characteristics (typical horn: 400-800 Hz)
            frequencies, power_spectrum = self._frequency_analysis(audio_data.astype(np.float32))
            if len(power_spectrum) > 0:
                dominant_freq = frequencies[np.argmax(power_spectrum)]
                return 400 <= dominant_freq <= 800
        
        return False
    
    def _trigger_callback(self, event_type: str, data: Dict):
        """Trigger registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event_type}: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove event callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def get_audio_statistics(self) -> Dict:
        """Get audio processing statistics"""
        if not self.audio_buffer:
            return {}
        
        buffer_array = np.array(self.audio_buffer, dtype=np.float32)
        buffer_array = buffer_array / 32768.0  # Normalize
        
        rms = np.sqrt(np.mean(buffer_array ** 2))
        peak = np.max(np.abs(buffer_array))
        
        # Frequency analysis of recent buffer
        frequencies, power_spectrum = self._frequency_analysis(buffer_array[-self.sample_rate:])
        
        dominant_freq = 0
        spectral_centroid = 0
        
        if len(power_spectrum) > 0:
            dominant_freq = frequencies[np.argmax(power_spectrum)]
            # Spectral centroid (brightness measure)
            spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        
        return {
            'rms_level': float(rms),
            'peak_level': float(peak),
            'dominant_frequency': float(dominant_freq),
            'spectral_centroid': float(spectral_centroid),
            'buffer_length_seconds': len(self.audio_buffer) / self.sample_rate,
            'sample_rate': self.sample_rate
        }
    
    def save_audio_snippet(self, filename: str, duration_seconds: int = 5):
        """Save recent audio to file"""
        if not self.audio_buffer:
            return False
        
        try:
            # Get recent audio data
            samples_needed = duration_seconds * self.sample_rate
            recent_data = list(self.audio_buffer)[-samples_needed:]
            
            if len(recent_data) < samples_needed:
                recent_data = list(self.audio_buffer)  # Use all available data
            
            # Convert to numpy array
            audio_array = np.array(recent_data, dtype=np.int16)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            self.logger.info(f"Audio snippet saved: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio save error: {e}")
            return False
    
    def detect_emergency_sounds(self) -> List[Dict]:
        """Detect emergency sounds (sirens, alarms, etc.)"""
        if not self.audio_buffer:
            return []
        
        emergency_sounds = []
        
        # Get recent audio for analysis
        recent_audio = np.array(list(self.audio_buffer)[-self.sample_rate * 3:], dtype=np.float32)
        recent_audio = recent_audio / 32768.0
        
        if len(recent_audio) < self.sample_rate:
            return []
        
        # Analyze in 1-second windows
        window_size = self.sample_rate
        for i in range(0, len(recent_audio) - window_size, window_size):
            window = recent_audio[i:i + window_size]
            
            # Frequency analysis
            frequencies, power_spectrum = self._frequency_analysis(window)
            
            if len(power_spectrum) == 0:
                continue
            
            dominant_freq = frequencies[np.argmax(power_spectrum)]
            rms = np.sqrt(np.mean(window ** 2))
            
            # Siren detection (alternating frequency pattern)
            if self._detect_siren_pattern(window):
                emergency_sounds.append({
                    'type': 'siren',
                    'confidence': 0.8,
                    'timestamp': time.time() - (len(recent_audio) - i) / self.sample_rate,
                    'dominant_frequency': dominant_freq,
                    'rms': rms
                })
            
            # Alarm detection (steady high-frequency)
            elif 2000 <= dominant_freq <= 4000 and rms > 0.2:
                emergency_sounds.append({
                    'type': 'alarm',
                    'confidence': 0.6,
                    'timestamp': time.time() - (len(recent_audio) - i) / self.sample_rate,
                    'dominant_frequency': dominant_freq,
                    'rms': rms
                })
        
        return emergency_sounds
    
    def _detect_siren_pattern(self, audio_data: np.ndarray) -> bool:
        """Detect siren-like frequency modulation"""
        if len(audio_data) < self.sample_rate // 4:  # Need at least 0.25 seconds
            return False
        
        # Calculate short-time Fourier transform
        window_size = len(audio_data) // 10
        hop_length = window_size // 4
        
        frequencies = []
        for i in range(0, len(audio_data) - window_size, hop_length):
            window = audio_data[i:i + window_size]
            freqs, power = self._frequency_analysis(window)
            if len(power) > 0:
                dominant_freq = freqs[np.argmax(power)]
                frequencies.append(dominant_freq)
        
        if len(frequencies) < 5:
            return False
        
        # Check for frequency modulation (up and down pattern)
        freq_array = np.array(frequencies)
        
        # Look for oscillating pattern
        diff = np.diff(freq_array)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        
        # Siren should have multiple direction changes
        return sign_changes >= 2 and np.std(freq_array) > 100
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.stop_recording()
        
        if self.stream:
            self.stream.close()
        
        self.pyaudio.terminate()
        self.logger.info("Audio cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass


def get_audio_devices() -> List[Dict]:
    """Get list of available audio devices"""
    pa = pyaudio.PyAudio()
    devices = []
    
    try:
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'max_input_channels': device_info['maxInputChannels'],
                    'default_sample_rate': device_info['defaultSampleRate']
                })
    finally:
        pa.terminate()
    
    return devices


if __name__ == "__main__":
    # Test audio processing
    logging.basicConfig(level=logging.INFO)
    
    # List available devices
    devices = get_audio_devices()
    print("Available audio devices:")
    for device in devices:
        print(f"  {device['index']}: {device['name']}")
    
    # Test audio processor
    processor = AudioProcessor()
    
    # Add test callbacks
    def on_noise(data):
        print(f"Noise detected: RMS={data['rms']:.3f}, Freq={data['dominant_frequency']:.1f}Hz")
    
    def on_distraction(data):
        print(f"Distraction detected: {data['distraction_type']}, RMS={data['rms']:.3f}")
    
    processor.add_callback('noise_detected', on_noise)
    processor.add_callback('distraction_detected', on_distraction)
    
    if processor.start_recording():
        print("Recording started. Speak or make noise...")
        try:
            for i in range(30):  # Test for 30 seconds
                time.sleep(1)
                stats = processor.get_audio_statistics()
                print(f"Stats: RMS={stats.get('rms_level', 0):.3f}, "
                     f"Peak={stats.get('peak_level', 0):.3f}, "
                     f"Freq={stats.get('dominant_frequency', 0):.1f}Hz")
                
                # Save snippet every 10 seconds
                if i % 10 == 9:
                    processor.save_audio_snippet(f"test_audio_{i//10}.wav")
                
                # Check for emergency sounds every 5 seconds
                if i % 5 == 4:
                    emergency = processor.detect_emergency_sounds()
                    if emergency:
                        print(f"Emergency sounds detected: {emergency}")
        
        except KeyboardInterrupt:
            print("Stopping...")
        
        finally:
            processor.cleanup()
    else:
        print("Failed to start audio recording")