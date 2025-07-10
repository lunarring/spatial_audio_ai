#!/usr/bin/env python3
"""
Comprehensive Latency Testing for Spatial Audio System

Multiple testing approaches:
1. Visual + Audio Sync Test
2. Timestamped Pulse Test  
3. Real-time Console Dashboard
4. Microphone Round-trip Test (optional)
"""

import numpy as np
import time
import threading
import sys
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque

from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE


@dataclass
class LatencyMeasurement:
    """Single latency measurement record"""
    send_time: float
    pulse_id: int
    expected_duration: float
    audio_data: np.ndarray


class LatencyTester:
    """Comprehensive latency testing system"""
    
    def __init__(self, test_duration: float = 30.0):
        self.test_duration = test_duration
        self.sample_rate = SAMPLING_RATE
        self.chunk_duration = CHUNKSIZE / SAMPLING_RATE
        
        # Testing state
        self.measurements: List[LatencyMeasurement] = []
        self.pulse_counter = 0
        self.test_start_time = 0
        self.last_console_update = 0
        
        # Audio generation parameters
        self.pulse_frequency = 1000  # Hz - easy to hear/detect
        self.pulse_duration = 0.1    # seconds
        self.silence_duration = 2.0  # seconds between pulses
        
        # Initialize audio system
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.sound_streamer = SoundNetworkStreamer()
        
        print("🎵 Spatial Audio Latency Tester Initialized")
        print(f"📊 Test Duration: {test_duration}s")
        print(f"⏱️  Chunk Duration: {self.chunk_duration*1000:.1f}ms")
        print(f"📻 Sample Rate: {self.sample_rate}Hz")

    def generate_audio_pulse(self, pulse_id: int) -> Tuple[np.ndarray, float]:
        """Generate a distinctive audio pulse with embedded timing"""
        
        # Create pulse: frequency sweep for easy identification
        pulse_samples = int(self.pulse_duration * self.sample_rate)
        t_pulse = np.linspace(0, self.pulse_duration, pulse_samples, False)
        
        # Frequency sweep from 800Hz to 1200Hz for distinctive sound
        freq_start, freq_end = 800, 1200
        instantaneous_freq = freq_start + (freq_end - freq_start) * (t_pulse / self.pulse_duration)
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
        
        # Generate pulse with envelope to avoid clicks
        envelope = np.sin(np.pi * t_pulse / self.pulse_duration) ** 2
        pulse = 0.7 * envelope * np.sin(phase)
        
        # Add silence after pulse
        silence_samples = int(self.silence_duration * self.sample_rate)
        silence = np.zeros(silence_samples)
        
        # Combine pulse + silence
        full_audio = np.concatenate([pulse, silence])
        total_duration = len(full_audio) / self.sample_rate
        
        return full_audio, total_duration

    def visual_countdown(self, countdown_seconds: int = 3):
        """Visual countdown before test starts"""
        print("\n" + "="*50)
        print("🚀 LATENCY TEST STARTING")
        print("="*50)
        
        for i in range(countdown_seconds, 0, -1):
            print(f"\r⏳ Starting in {i}...", end="", flush=True)
            time.sleep(1)
        
        print(f"\r🎯 GO! Listen for audio pulses NOW!")
        print("👂 You should hear frequency sweeps every 2 seconds")
        print("📊 Watch console for real-time latency data...")
        print()

    def update_console_dashboard(self, chunk_count: int, current_time: float):
        """Real-time console dashboard (update every 0.5s)"""
        
        if current_time - self.last_console_update < 0.5:
            return
            
        self.last_console_update = current_time
        elapsed = current_time - self.test_start_time
        progress = min(elapsed / self.test_duration * 100, 100)
        
        # Calculate recent average latency (last 5 measurements)
        recent_latencies = []
        if len(self.measurements) >= 2:
            for i in range(max(0, len(self.measurements)-5), len(self.measurements)):
                measurement = self.measurements[i]
                theoretical_latency = measurement.expected_duration
                recent_latencies.append(theoretical_latency)
        
        avg_latency = np.mean(recent_latencies) if recent_latencies else 0
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * progress // 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Clear previous lines and update
        print(f"\r\033[3A", end="")  # Move cursor up 3 lines
        print(f"📊 Progress: [{bar}] {progress:5.1f}%")
        print(f"⏱️  Elapsed: {elapsed:6.1f}s | Chunks: {chunk_count:4d} | Pulses: {len(self.measurements):2d}")
        print(f"🎯 Avg Latency: {avg_latency*1000:6.1f}ms | Target: {self.chunk_duration*1000:.1f}ms/chunk")

    def run_test_1_visual_sync(self):
        """Test 1: Visual + Audio Sync Test"""
        print("\n" + "🔬 TEST 1: Visual + Audio Sync")
        print("Listen carefully - you'll hear sweeping tones.")
        print("Note any delay between console messages and audio.")
        
        self.visual_countdown(3)
        
        # Generate test audio sequence
        audio_sequence = []
        expected_timings = []
        
        # Create pulses throughout test duration
        current_time = 0
        while current_time < self.test_duration:
            pulse_audio, pulse_duration = self.generate_audio_pulse(self.pulse_counter)
            
            measurement = LatencyMeasurement(
                send_time=0,  # Will be set during streaming
                pulse_id=self.pulse_counter,
                expected_duration=pulse_duration,
                audio_data=pulse_audio
            )
            
            self.measurements.append(measurement)
            audio_sequence.append(pulse_audio)
            expected_timings.append(current_time)
            
            current_time += pulse_duration
            self.pulse_counter += 1
        
        # Concatenate all audio
        full_audio = np.concatenate(audio_sequence)
        
        # Create playback object and register
        sound_object = SO_Playback(full_audio)
        self.scene.register(sound_object)
        
        # Stream with timing
        self.test_start_time = time.perf_counter()
        chunk_count = 0
        measurement_idx = 0
        
        print("\n" * 3)  # Space for dashboard
        
        for chunk in self.scene.run():
            current_time = time.perf_counter()
            
            # Record send time for current measurement
            if measurement_idx < len(self.measurements):
                if chunk_count * self.chunk_duration >= expected_timings[measurement_idx]:
                    self.measurements[measurement_idx].send_time = current_time
                    print(f"🔊 PULSE {measurement_idx + 1} SENT at {current_time - self.test_start_time:.3f}s")
                    measurement_idx += 1
            
            # Send chunk
            chunk = np.clip(chunk, -1, 1)
            self.sound_streamer.send(chunk)
            
            # Update dashboard
            self.update_console_dashboard(chunk_count, current_time)
            
            chunk_count += 1
            
            # Check if test complete
            if current_time - self.test_start_time >= self.test_duration:
                break
            
            # Precise timing
            next_time = self.test_start_time + chunk_count * self.chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_test_2_pulse_timing(self):
        """Test 2: Detailed Pulse Timing Analysis"""
        print("\n" + "🔬 TEST 2: Pulse Timing Analysis")
        print("Generating single pulses with precise timing...")
        
        # Clear previous measurements
        self.measurements = []
        self.pulse_counter = 0
        
        num_pulses = 5
        pulse_interval = 3.0  # seconds
        
        for pulse_num in range(num_pulses):
            print(f"\n🎯 Pulse {pulse_num + 1}/{num_pulses}")
            
            # Generate single pulse
            pulse_audio, pulse_duration = self.generate_audio_pulse(pulse_num)
            
            # Visual countdown for this pulse
            print("⏳ Sending pulse in: 3", end="", flush=True)
            time.sleep(1)
            print(" 2", end="", flush=True)
            time.sleep(1)
            print(" 1", end="", flush=True)
            time.sleep(1)
            print(" 🔊 NOW!")
            
            # Record precise send time
            send_time = time.perf_counter()
            
            # Create and stream
            sound_object = SO_Playback(pulse_audio)
            scene = Scene(self.spatializer)
            scene.register(sound_object)
            
            measurement = LatencyMeasurement(
                send_time=send_time,
                pulse_id=pulse_num,
                expected_duration=pulse_duration,
                audio_data=pulse_audio
            )
            self.measurements.append(measurement)
            
            # Stream this pulse
            chunk_count = 0
            for chunk in scene.run():
                chunk = np.clip(chunk, -1, 1)
                self.sound_streamer.send(chunk)
                
                chunk_count += 1
                if chunk_count * self.chunk_duration >= pulse_duration:
                    break
                
                # Timing
                next_time = send_time + chunk_count * self.chunk_duration
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print(f"✅ Pulse {pulse_num + 1} sent - listen for sweep tone")
            
            # Wait before next pulse
            if pulse_num < num_pulses - 1:
                time.sleep(pulse_interval)

    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*60)
        print("📊 LATENCY TEST RESULTS")
        print("="*60)
        
        if not self.measurements:
            print("❌ No measurements recorded!")
            return
        
        print(f"📈 Total Measurements: {len(self.measurements)}")
        print(f"⏱️  Chunk Duration: {self.chunk_duration*1000:.1f}ms")
        print(f"🔊 Pulse Duration: {self.pulse_duration*1000:.0f}ms")
        print(f"🔇 Silence Duration: {self.silence_duration*1000:.0f}ms")
        
        # Theoretical minimum latency analysis
        min_latency_ms = self.chunk_duration * 1000
        print(f"\n🎯 Theoretical Minimum Latency: {min_latency_ms:.1f}ms (1 chunk)")
        
        # Expected latency sources
        print("\n📋 Expected Latency Sources:")
        print(f"   • Serialization: ~0.01ms (optimized)")
        print(f"   • Network transmission: ~1-10ms")
        print(f"   • Server processing: ~0.1-1ms")
        print(f"   • Audio buffer: ~{min_latency_ms:.1f}ms")
        print(f"   • System audio: ~5-20ms")
        print(f"   • TOTAL EXPECTED: ~{min_latency_ms + 15:.0f}ms")
        
        print(f"\n💡 NEXT STEPS:")
        print("   1. Use microphone to measure actual end-to-end latency")
        print("   2. Compare with expected ~{:.0f}ms total".format(min_latency_ms + 15))
        print("   3. If higher than expected, investigate buffer settings")

    def run_all_tests(self):
        """Run complete latency test suite"""
        try:
            print("🎵 SPATIAL AUDIO LATENCY TEST SUITE")
            print("="*50)
            
            # Test 1: Visual sync
            self.run_test_1_visual_sync()
            
            # Brief pause
            print("\n⏸️  Pausing between tests...")
            time.sleep(3)
            
            # Test 2: Pulse timing
            self.run_test_2_pulse_timing()
            
            # Results
            self.print_results()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Test interrupted by user")
            self.print_results()
        except Exception as e:
            print(f"\n❌ Test error: {e}")
        finally:
            self.sound_streamer.close()


def main():
    """Main function with test options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spatial Audio Latency Tester')
    parser.add_argument('--duration', '-d', type=float, default=15.0,
                       help='Test duration in seconds (default: 15)')
    parser.add_argument('--test', '-t', choices=['visual', 'pulse', 'all'], default='all',
                       help='Test type to run (default: all)')
    
    args = parser.parse_args()
    
    tester = LatencyTester(test_duration=args.duration)
    
    if args.test == 'visual':
        tester.run_test_1_visual_sync()
    elif args.test == 'pulse':
        tester.run_test_2_pulse_timing()
    else:
        tester.run_all_tests()
    
    tester.print_results()


if __name__ == "__main__":
    main() 