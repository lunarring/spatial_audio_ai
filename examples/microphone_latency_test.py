#!/usr/bin/env python3
"""
Microphone-based End-to-End Latency Measurement

This script measures real-world latency by:
1. Sending distinctive audio pulses through the spatial audio system
2. Recording audio from a microphone (picking up speakers)
3. Cross-correlating to find the actual delay

Requirements:
- Microphone positioned to pick up speaker output
- Quiet environment for accurate detection
"""

import numpy as np
import sounddevice as sd
import time
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
from spatial_audio_ai.config import BLOCKSIZE
from collections import deque
import scipy.signal

from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE


@dataclass
class MicLatencyResult:
    """Result of a microphone-based latency measurement"""
    send_time: float
    detected_time: float
    latency_ms: float
    confidence: float
    pulse_id: int


class MicrophoneLatencyTester:
    """Microphone-based latency measurement system"""
    
    def __init__(self, recording_device: Optional[int] = None):
        self.sample_rate = SAMPLING_RATE
        self.chunk_duration = CHUNKSIZE / SAMPLING_RATE
        
        # Audio parameters for detection
        self.pulse_duration = 0.2  # seconds - longer for better detection
        self.silence_duration = 3.0  # seconds between pulses
        self.detection_threshold = 0.1  # amplitude threshold for detection
        
        # Recording setup
        self.recording_device = recording_device
        self.recording_data = deque(maxlen=int(10 * self.sample_rate))  # 10 seconds buffer
        self.recording_active = False
        self.recording_thread = None
        
        # Results
        self.results: List[MicLatencyResult] = []
        
        # Initialize audio system
        self.spatializer = Spatializer()
        self.sound_streamer = SoundNetworkStreamer()
        
        print("🎤 Microphone Latency Tester Initialized")
        print(f"📻 Sample Rate: {self.sample_rate}Hz")
        print(f"⏱️  Pulse Duration: {self.pulse_duration}s")
        
        # List available input devices
        self.list_input_devices()

    def list_input_devices(self):
        """List available audio input devices"""
        print("\n🎤 Available Input Devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i:2d}: {device['name']} (channels: {device['max_input_channels']})")
        
        if self.recording_device is None:
            try:
                self.recording_device = sd.default.device[0]  # Default input device
                print(f"\n✅ Using default input device: {self.recording_device}")
            except:
                print("\n❌ No default input device found!")
                print("   Please specify --device <id> when running the script")

    def generate_detection_pulse(self, pulse_id: int) -> np.ndarray:
        """Generate a pulse optimized for microphone detection"""
        
        # Create a chirp (frequency sweep) for robust detection
        pulse_samples = int(self.pulse_duration * self.sample_rate)
        t = np.linspace(0, self.pulse_duration, pulse_samples, False)
        
        # Chirp from 500Hz to 2000Hz - good frequency range for detection
        f0, f1 = 500, 2000
        chirp = 0.8 * scipy.signal.chirp(t, f0, self.pulse_duration, f1, method='linear')
        
        # Apply smooth envelope to avoid clicks
        envelope = np.sin(np.pi * t / self.pulse_duration) ** 2
        pulse = envelope * chirp
        
        return pulse

    def audio_recording_callback(self, indata, frames, time_info, status):
        """Callback for recording audio from microphone"""
        if status:
            print(f"🎤 Recording status: {status}")
        
        # Store audio data (mono)
        if indata.shape[1] > 1:
            # Convert to mono if stereo
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata[:, 0]
        
        # Add to circular buffer
        self.recording_data.extend(mono_data)

    def start_recording(self):
        """Start microphone recording"""
        if self.recording_device is None:
            print("❌ No recording device available!")
            return False
        
        try:
            self.recording_active = True
            self.stream = sd.InputStream(
                device=self.recording_device,
                channels=1,
                samplerate=self.sample_rate,
                                            blocksize=BLOCKSIZE,
                callback=self.audio_recording_callback
            )
            self.stream.start()
            print("🎤 Recording started...")
            return True
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop microphone recording"""
        if hasattr(self, 'stream'):
            self.recording_active = False
            self.stream.stop()
            self.stream.close()
            print("🎤 Recording stopped.")

    def detect_pulse_in_recording(self, template_pulse: np.ndarray, 
                                 search_start_time: float, 
                                 search_duration: float = 5.0) -> Tuple[Optional[float], float]:
        """
        Detect when a pulse was recorded using cross-correlation
        
        Returns:
            (detected_time, confidence) or (None, 0.0) if not found
        """
        
        # Convert recording buffer to numpy array
        if len(self.recording_data) == 0:
            return None, 0.0
        
        recording_array = np.array(list(self.recording_data))
        
        # Calculate search window in samples
        search_start_sample = int(search_start_time * self.sample_rate)
        search_samples = int(search_duration * self.sample_rate)
        
        # Extract search window
        if search_start_sample + search_samples > len(recording_array):
            return None, 0.0
        
        search_window = recording_array[search_start_sample:search_start_sample + search_samples]
        
        if len(search_window) < len(template_pulse):
            return None, 0.0
        
        # Cross-correlate with template
        correlation = scipy.signal.correlate(search_window, template_pulse, mode='valid')
        
        # Find peak
        max_idx = np.argmax(np.abs(correlation))
        max_correlation = np.abs(correlation[max_idx])
        
        # Calculate confidence (normalized correlation)
        template_energy = np.sum(template_pulse ** 2)
        window_energy = np.sum(search_window[max_idx:max_idx + len(template_pulse)] ** 2)
        
        if template_energy > 0 and window_energy > 0:
            confidence = max_correlation / np.sqrt(template_energy * window_energy)
        else:
            confidence = 0.0
        
        # Threshold for detection
        if confidence > 0.3:  # Adjustable threshold
            detected_sample = search_start_sample + max_idx
            detected_time = detected_sample / self.sample_rate
            return detected_time, confidence
        
        return None, confidence

    def run_latency_test(self, num_pulses: int = 5):
        """Run complete microphone-based latency test"""
        
        print("\n🎯 MICROPHONE LATENCY TEST")
        print("="*50)
        print("📋 Setup Instructions:")
        print("   1. Position microphone to pick up speaker output")
        print("   2. Ensure reasonably quiet environment")
        print("   3. Adjust speaker volume for clear detection")
        print("   4. Press Enter when ready...")
        
        input()  # Wait for user
        
        # Start recording
        if not self.start_recording():
            return
        
        print(f"\n🔊 Starting test with {num_pulses} pulses...")
        time.sleep(1)  # Let recording stabilize
        
        recording_start_time = time.perf_counter()
        
        try:
            for pulse_num in range(num_pulses):
                print(f"\n🎯 Pulse {pulse_num + 1}/{num_pulses}")
                
                # Generate pulse
                pulse_audio = self.generate_detection_pulse(pulse_num)
                
                # Countdown
                print("⏳ Sending in: 3", end="", flush=True)
                time.sleep(1)
                print(" 2", end="", flush=True)
                time.sleep(1)
                print(" 1", end="", flush=True)
                time.sleep(1)
                print(" 🔊 SENDING NOW!")
                
                # Record exact send time
                send_time = time.perf_counter()
                relative_send_time = send_time - recording_start_time
                
                # Create and stream pulse
                sound_object = SO_Playback(pulse_audio)
                scene = Scene(self.spatializer)
                scene.register(sound_object)
                
                # Stream the pulse
                chunk_count = 0
                for chunk in scene.run():
                    chunk = np.clip(chunk, -1, 1)
                    self.sound_streamer.send(chunk)
                    
                    chunk_count += 1
                    if chunk_count * self.chunk_duration >= self.pulse_duration:
                        break
                    
                    # Precise timing
                    next_time = send_time + chunk_count * self.chunk_duration
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                print("✅ Pulse sent, analyzing recording...")
                
                # Wait a moment for the audio to be recorded
                time.sleep(1.0)
                
                # Detect pulse in recording
                detected_time, confidence = self.detect_pulse_in_recording(
                    pulse_audio, relative_send_time, search_duration=3.0
                )
                
                if detected_time is not None:
                    latency_ms = (detected_time - relative_send_time) * 1000
                    
                    result = MicLatencyResult(
                        send_time=relative_send_time,
                        detected_time=detected_time,
                        latency_ms=latency_ms,
                        confidence=confidence,
                        pulse_id=pulse_num
                    )
                    self.results.append(result)
                    
                    print(f"✅ DETECTED! Latency: {latency_ms:.1f}ms (confidence: {confidence:.2f})")
                else:
                    print(f"❌ Not detected (confidence: {confidence:.2f})")
                    print("   Try adjusting microphone position or speaker volume")
                
                # Wait before next pulse
                if pulse_num < num_pulses - 1:
                    print("⏸️  Waiting for next pulse...")
                    time.sleep(self.silence_duration)
        
        finally:
            self.stop_recording()
            self.sound_streamer.close()

    def print_results(self):
        """Print comprehensive latency test results"""
        print("\n" + "="*60)
        print("📊 MICROPHONE LATENCY TEST RESULTS")
        print("="*60)
        
        if not self.results:
            print("❌ No successful measurements!")
            print("\n💡 Troubleshooting:")
            print("   • Check microphone positioning")
            print("   • Increase speaker volume")
            print("   • Reduce background noise")
            print("   • Try different microphone device")
            return
        
        # Calculate statistics
        latencies = [r.latency_ms for r in self.results]
        confidences = [r.confidence for r in self.results]
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        mean_confidence = np.mean(confidences)
        
        print(f"📈 Successful Measurements: {len(self.results)}")
        print(f"🎯 Mean Latency: {mean_latency:.1f} ± {std_latency:.1f} ms")
        print(f"📏 Range: {min_latency:.1f} - {max_latency:.1f} ms")
        print(f"🔍 Mean Confidence: {mean_confidence:.3f}")
        
        print("\n📋 Individual Results:")
        for i, result in enumerate(self.results):
            print(f"   Pulse {result.pulse_id + 1}: {result.latency_ms:6.1f}ms (conf: {result.confidence:.3f})")
        
        # Compare with theoretical minimum
        theoretical_min = self.chunk_duration * 1000
        print(f"\n🎯 Analysis:")
        print(f"   • Theoretical minimum: {theoretical_min:.1f}ms (1 audio chunk)")
        print(f"   • Measured latency: {mean_latency:.1f}ms")
        print(f"   • Additional latency: {mean_latency - theoretical_min:.1f}ms")
        
        if mean_latency > theoretical_min + 50:  # More than 50ms extra
            print(f"   ⚠️  High additional latency detected!")
            print(f"   💡 Consider investigating buffer settings")
        else:
            print(f"   ✅ Latency within reasonable range")


def main():
    """Main function for microphone latency testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Microphone-based Latency Tester')
    parser.add_argument('--device', '-d', type=int, default=None,
                       help='Audio input device ID (see list on startup)')
    parser.add_argument('--pulses', '-p', type=int, default=5,
                       help='Number of test pulses (default: 5)')
    
    args = parser.parse_args()
    
    tester = MicrophoneLatencyTester(recording_device=args.device)
    
    if tester.recording_device is not None:
        tester.run_latency_test(num_pulses=args.pulses)
        tester.print_results()
    else:
        print("❌ No valid recording device. Use --device <id> to specify one.")


if __name__ == "__main__":
    main() 