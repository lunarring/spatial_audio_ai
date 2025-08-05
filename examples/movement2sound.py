#!/usr/bin/env python3
"""
Real-time Velocity-Triggered Kick Drum Sampler with Gradio Interface

This script provides a real-time velocity-triggered kick drum sampler using the 
SO_SamplePlayback class with a Gradio web interface for parameter control.
When a rigid body's velocity exceeds a threshold, a random kick drum sample is triggered.
"""

import numpy as np
import time
import threading
import os
import glob
import random
from spatial_audio_ai.tools.spatializer import Spatializer, Scene, CHUNKSIZE, SAMPLING_RATE
from spatial_audio_ai.tools.sound_objects import SO_SamplePlayback
from spatial_audio_ai.tools.client import SoundNetworkStreamer
import lunar_tools as lt

# Try to import OptiTrack modules
try:
from optitrack_python.rigid_body import RigidBody
from optitrack_python.motive_receiver import MotiveReceiver
    HAS_OPTITRACK = True
except ImportError:
    HAS_OPTITRACK = False
    print("Warning: optitrack_python not available. Motion tracking will not work.")
    # Create dummy classes for testing
    class RigidBody:
        def __init__(self, *args, **kwargs):
            pass
        def update(self):
            pass
        @property
        def positions(self):
            return MockPositions()
    
    class MotiveReceiver:
        def __init__(self, *args, **kwargs):
            pass
        def get_last(self):
            return None
        def stop(self):
            pass
    
    class MockPositions:
        def get_last(self):
            return None

# Try to import audio libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    try:
        from scipy.io import wavfile
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Warning: gradio not available. Web interface will not work.")


class VelocityTriggerController:
    """Controller class for managing velocity-triggered kick drum samples."""
    
    def __init__(self):
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.5
        
        # Velocity tracking
        self.rigid_bodies = {}
        self.active_rigid_body = 'C'  # Default to rigid body C
        self.velocity_threshold = 3.0  # m/s
        
        # Velocity-to-volume mapping
        self.min_velocity_volume = 0.2  # Minimum volume at threshold velocity
        self.max_velocity_volume = 1.0  # Maximum volume at max velocity
        self.max_velocity_for_volume = 10.0  # Velocity at which max volume is reached
        
        # Position and velocity tracking
        self.current_positions = {}
        self.previous_positions = {}
        self.current_velocities = {}
        self.previous_velocities = {}
        self.last_trigger_time = {}
        self.min_time_between_triggers = 1.0  # Minimum time between triggers
        
        # X-plane crossing detection
        self.x_plane_position = 0.0  # X=0 plane
        self.previous_x_positions = {}  # Track previous X positions for crossing detection
        self.crossing_velocities = {}  # Store velocity when crossing occurs
        
        # Sample management
        self.kick_samples = []
        self.kick_sample_paths = []  # Track file paths for debugging
        self.sample_objects = []
        self.max_concurrent_samples = 8  # Maximum number of samples playing at once
        self.current_sample_index = 0
        
        # Sample selection mode
        self.use_random_samples = False  # Default to fixed sample
        self.fixed_sample_path = "/home/lugo/Downloads/Kicks_StayOnBeat.com_/Kicks_StayOnBeat.com/Kicks_StayOnBeat.com (35).wav"
        self.fixed_sample_index = -1  # Will be set when sample is found
        
        # Audio system
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        self.is_muted = False
        self.sample_amplitude = 0.8
        
        # Load kick drum samples
        self.kick_sample_folder = "/home/lugo/Downloads/Kicks_StayOnBeat.com_/Kicks_StayOnBeat.com/"
        self.load_kick_samples()
        
        # OptiTrack setup
        self.motive = None
        self.setup_optitrack()
        
        # Timing for velocity calculation
        self.last_update_time = time.time()
    
    def load_audio_file(self, filepath):
        """Load audio file using available libraries with proper sample rate handling."""
        # Try soundfile first (best for preserving quality)
        try:
            import soundfile as sf
            # Load original file
            audio_data, original_sr = sf.read(filepath)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to target sample rate using high-quality resampling
            if original_sr != SAMPLING_RATE:
                try:
                    import scipy.signal
                    # Use high-quality resampling to avoid pitch shift
                    audio_data = scipy.signal.resample_poly(
                        audio_data, 
                        up=SAMPLING_RATE, 
                        down=original_sr
                    )
                    print(f"Resampled {filepath} from {original_sr}Hz to {SAMPLING_RATE}Hz")
                except ImportError:
                    # Fallback to simple linear interpolation
                    old_length = len(audio_data)
                    new_length = int(old_length * SAMPLING_RATE / original_sr)
                    indices = np.linspace(0, old_length - 1, new_length)
                    audio_data = np.interp(indices, np.arange(old_length), audio_data)
                    print(f"Basic resampled {filepath} from {original_sr}Hz to {SAMPLING_RATE}Hz")
            
            # Ensure data is float type (required by spatial audio system)
            audio_data = audio_data.astype(float)
            
            # Normalize to prevent clipping
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
            
            return audio_data
            
        except ImportError:
            pass
        except Exception as e:
            print(f"soundfile failed to load {filepath}: {e}")
        
        # Try librosa
        if HAS_LIBROSA:
            try:
                # Load audio file with librosa - preserve original sample rate first
                audio_data, original_sr = librosa.load(filepath, sr=None, mono=True)
                
                # Resample to target if needed
                if original_sr != SAMPLING_RATE:
                    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=SAMPLING_RATE)
                    print(f"Librosa resampled {filepath} from {original_sr}Hz to {SAMPLING_RATE}Hz")
                
                # Ensure data is float type (required by spatial audio system)
                audio_data = audio_data.astype(float)
                
                # Normalize to prevent clipping
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
                
                return audio_data
            except Exception as e:
                print(f"librosa failed to load {filepath}: {e}")
        
        # Try scipy.io.wavfile as fallback
        elif HAS_SCIPY:
            try:
                from scipy.io import wavfile
                # Load audio file with scipy.io.wavfile
                sr, audio_data = wavfile.read(filepath)
                
                # Convert to float and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if necessary (basic resampling)
                if sr != SAMPLING_RATE:
                    old_length = len(audio_data)
                    new_length = int(old_length * SAMPLING_RATE / sr)
                    indices = np.linspace(0, old_length - 1, new_length)
                    audio_data = np.interp(indices, np.arange(old_length), audio_data)
                    print(f"Scipy resampled {filepath} from {sr}Hz to {SAMPLING_RATE}Hz")
                
                # Ensure data is float type (required by spatial audio system)
                audio_data = audio_data.astype(float)
                
                # Normalize to prevent clipping
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
                
                return audio_data
            except Exception as e:
                print(f"scipy.wavfile failed to load {filepath}: {e}")
        
        print(f"No audio loading library available for {filepath}")
        return None
        
    def load_kick_samples(self):
        """Load all kick drum samples from the specified folder."""
        print("Loading kick drum samples...")
        
        if not os.path.exists(self.kick_sample_folder):
            print(f"Warning: Kick sample folder not found: {self.kick_sample_folder}")
            # Create dummy samples for testing
            self.create_dummy_samples()
            return
        
        # Find all WAV files (both .wav and .WAV)
        wav_files = glob.glob(os.path.join(self.kick_sample_folder, "*.wav"))
        wav_files.extend(glob.glob(os.path.join(self.kick_sample_folder, "*.WAV")))
        
        if not wav_files:
            print("No WAV files found in kick sample folder")
            self.create_dummy_samples()
            return
        
        # Load each sample
        loaded_count = 0
        for wav_file in sorted(wav_files):
            try:
                audio_data = self.load_audio_file(wav_file)
                if audio_data is not None:
                    self.kick_samples.append(audio_data)
                    self.kick_sample_paths.append(wav_file)  # Track file paths
                    loaded_count += 1
                
            except Exception as e:
                print(f"Failed to load {wav_file}: {e}")
        
        print(f"Loaded {loaded_count} kick drum samples")
        
        # Find the index of the fixed sample
        self.find_fixed_sample_index()
        
        if loaded_count == 0:
            self.create_dummy_samples()
        
        # Create sample playback objects
        self.create_sample_objects()
    
    def find_fixed_sample_index(self):
        """Find the index of the fixed sample in the loaded samples."""
        for i, path in enumerate(self.kick_sample_paths):
            if path == self.fixed_sample_path:
                self.fixed_sample_index = i
                print(f"Found fixed sample at index {i}: {path}")
                return
        
        # If not found, use the first sample if available
        if len(self.kick_samples) > 0:
            self.fixed_sample_index = 0
            print(f"Fixed sample not found, using first sample: {self.kick_sample_paths[0] if self.kick_sample_paths else 'dummy'}")
        else:
            self.fixed_sample_index = -1
            print("No samples available for fixed sample")
    
    def create_dummy_samples(self):
        """Create dummy kick drum samples for testing when no files are available."""
        print("Creating dummy kick drum samples...")
        
        # Create 10 different synthetic kick drums
        for i in range(10):
            duration = 0.5  # 500ms
            samples = int(duration * SAMPLING_RATE)
            t = np.linspace(0, duration, samples, endpoint=False)
            
            # Create a kick-like sound: low frequency with exponential decay
            base_freq = 50 + i * 10  # Vary base frequency
            click_freq = 1000 + i * 200  # Vary click frequency
            
            # Sub bass component
            bass = np.sin(2 * np.pi * base_freq * t) * np.exp(-t * 8)
            
            # Click component
            click = np.sin(2 * np.pi * click_freq * t) * np.exp(-t * 50)
            
            # Noise component
            noise = np.random.normal(0, 0.1, samples) * np.exp(-t * 20)
            
            # Combine and normalize
            kick = bass + click * 0.3 + noise * 0.2
            kick = kick / np.max(np.abs(kick)) * 0.9
            
            # Ensure float type
            kick = kick.astype(float)
            
            self.kick_samples.append(kick)
            self.kick_sample_paths.append(f"dummy_kick_{i+1}.wav")  # Track dummy paths too
        
        print(f"Created {len(self.kick_samples)} dummy kick samples")
    
    def create_sample_objects(self):
        """Create sample playback objects for concurrent playback."""
        self.sample_objects = []
        
        for i in range(self.max_concurrent_samples):
            # Start with first sample, they'll be updated when triggered
            sample_obj = SO_SamplePlayback(
                sample_data=self.kick_samples[0] if self.kick_samples else np.zeros(1000),
                position=np.zeros(2, dtype=float),
                amplitude=self.sample_amplitude
            )
            self.sample_objects.append(sample_obj)
            self.scene.register(sample_obj)
        
    def setup_optitrack(self):
        """Initialize OptiTrack connection and rigid bodies."""
        if not HAS_OPTITRACK:
            print("OptiTrack not available - creating dummy tracking data")
            self.motive = None
            # Create dummy tracking for testing
            rigid_body_names = ['A', 'B', 'C', 'D']
            for name in rigid_body_names:
                self.rigid_bodies[name] = RigidBody(None, name)
                self.current_positions[name] = np.array([0.0, 0.0, 0.0])
                self.previous_positions[name] = np.array([0.0, 0.0, 0.0])
                self.current_velocities[name] = np.array([0.0, 0.0, 0.0])
                self.last_trigger_time[name] = 0.0
            return
        
        try:
            print("Connecting to OptiTrack...")
            self.motive = MotiveReceiver(server_ip="10.40.49.47")
            
            print("Waiting for data connection...")
            time.sleep(1)
            
            # Test basic connection first
            print("Testing basic connection...")
            for i in range(50):  # Try for 5 seconds
                latest_data = self.motive.get_last()
                if latest_data:
                    print(f"✓ Connection established! Frame ID: {latest_data['frame_id']}")
                    break
                time.sleep(0.1)
            else:
                print("✗ No data received. Check OptiTrack connection.")
                self.motive.stop()
                self.motive = None
                return
            
            # Create rigid bodies A, B, C, D
            rigid_body_names = ['A', 'B', 'C', 'D']
            for name in rigid_body_names:
                self.rigid_bodies[name] = RigidBody(self.motive, name)
                self.current_positions[name] = np.array([0.0, 0.0, 0.0])
                self.previous_positions[name] = np.array([0.0, 0.0, 0.0])
                self.current_velocities[name] = np.array([0.0, 0.0, 0.0])
                self.previous_velocities[name] = np.array([0.0, 0.0, 0.0])
                self.previous_x_positions[name] = 0.0
                self.crossing_velocities[name] = 0.0
                self.last_trigger_time[name] = 0.0
            
            print(f"OptiTrack connection established with rigid bodies: {', '.join(rigid_body_names)}")
            
        except Exception as e:
            print(f"Failed to setup OptiTrack: {e}")
            self.motive = None
            self.rigid_bodies = {}
    
    def update_from_motion(self):
        """Update velocity tracking and trigger samples based on velocity threshold."""
        if not self.rigid_bodies or self.motive is None:
            return
            
        try:
            # Get latest data
            latest_data = self.motive.get_last()
            if not latest_data:
                return
            
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Only track the active rigid body
            if self.active_rigid_body in self.rigid_bodies:
                rigid_body = self.rigid_bodies[self.active_rigid_body]
                
                try:
                    # Update rigid body and get position
                    rigid_body.update()
                    position = rigid_body.positions.get_last()
                    
                    if position is not None and dt > 0:
                        # Store previous position
                        self.previous_positions[self.active_rigid_body] = self.current_positions[self.active_rigid_body].copy()
                        
                        # Update current position
                        self.current_positions[self.active_rigid_body] = np.array([
                            position[0], position[1], position[2]
                        ])
                        
                        # Calculate velocity (3D)
                        if dt > 0.001:  # Avoid division by very small numbers
                            velocity = (
                                self.current_positions[self.active_rigid_body] - 
                                self.previous_positions[self.active_rigid_body]
                            ) / dt
                            
                            self.current_velocities[self.active_rigid_body] = velocity
                            velocity_magnitude = np.linalg.norm(velocity)
                            
                            # Get current and previous X positions
                            current_x = self.current_positions[self.active_rigid_body][0]
                            previous_x = self.previous_x_positions.get(self.active_rigid_body, current_x)
                            
                            # Check for X-plane crossing (negative to positive X)
                            if (previous_x < self.x_plane_position and 
                                current_x >= self.x_plane_position and
                                velocity_magnitude > self.velocity_threshold and
                                current_time - self.last_trigger_time[self.active_rigid_body] > self.min_time_between_triggers):
                                
                                # Store crossing velocity for volume calculation and status display
                                self.crossing_velocities[self.active_rigid_body] = velocity_magnitude
                                
                                # Trigger sample with crossing velocity
                                self.trigger_kick_sample_with_peak_velocity(velocity_magnitude)
                                self.last_trigger_time[self.active_rigid_body] = current_time
                                
                                print(f"X-plane crossing detected! Velocity: {velocity_magnitude:.2f} m/s (X: {previous_x:.3f} -> {current_x:.3f})")
                            
                            # Update previous X position for next iteration
                            self.previous_x_positions[self.active_rigid_body] = current_x
                
                except Exception as e:
                    print(f"Error updating rigid body {self.active_rigid_body}: {e}")
                    
        except Exception as e:
            print(f"Error in motion update: {e}")
        
    def trigger_kick_sample_with_peak_velocity(self, peak_velocity):
        """Trigger a kick drum sample using the detected peak velocity for volume calculation."""
        if not self.kick_samples or self.is_muted:
            return
        
        # Find an available sample object (not currently playing)
        available_obj = None
        for obj in self.sample_objects:
            if not obj.is_playing_sample():
                available_obj = obj
                break
        
        # If all objects are playing, use the next one in rotation
        if available_obj is None:
            available_obj = self.sample_objects[self.current_sample_index]
            self.current_sample_index = (self.current_sample_index + 1) % self.max_concurrent_samples
        
        # Calculate velocity-based amplitude using peak velocity
        velocity_based_amplitude = self.calculate_velocity_amplitude(peak_velocity)
        
        # Select sample based on mode (fixed or random)
        if self.use_random_samples:
            # Random selection (original behavior)
            sample_index = random.randint(0, len(self.kick_samples) - 1)
        else:
            # Fixed sample selection
            sample_index = self.fixed_sample_index if self.fixed_sample_index >= 0 else 0
        
        selected_sample = self.kick_samples[sample_index]
        sample_path = self.kick_sample_paths[sample_index] if sample_index < len(self.kick_sample_paths) else f"sample_{sample_index}"
        
        # Update the sample object with new sample data and velocity-based amplitude
        available_obj.sample_data = selected_sample
        available_obj.set_amplitude(velocity_based_amplitude)
        
        # Set position to current rigid body position (X, Z coordinates)
        if self.active_rigid_body in self.current_positions:
            pos_3d = self.current_positions[self.active_rigid_body]
            pos_2d = np.array([pos_3d[0], pos_3d[2]])  # X, Z coordinates
            available_obj.set_position(pos_2d)
        
        # Trigger the sample
        available_obj.trigger()
        
        print(f"Triggered kick sample: {sample_path} (peak velocity: {peak_velocity:.2f} m/s, volume: {velocity_based_amplitude:.2f})")
    
    def trigger_kick_sample(self):
        """Trigger a random kick drum sample."""
        if not self.kick_samples or self.is_muted:
            return
        
        # Find an available sample object (not currently playing)
        available_obj = None
        for obj in self.sample_objects:
            if not obj.is_playing_sample():
                available_obj = obj
                break
        
        # If all objects are playing, use the next one in rotation
        if available_obj is None:
            available_obj = self.sample_objects[self.current_sample_index]
            self.current_sample_index = (self.current_sample_index + 1) % self.max_concurrent_samples
        
        # Calculate current velocity magnitude
        current_velocity = 0.0
        if self.active_rigid_body in self.current_velocities:
            current_velocity = np.linalg.norm(self.current_velocities[self.active_rigid_body])
        
        # Calculate velocity-based amplitude
        velocity_based_amplitude = self.calculate_velocity_amplitude(current_velocity)
        
        # Select sample based on mode (fixed or random)
        if self.use_random_samples:
            # Random selection (original behavior)
            sample_index = random.randint(0, len(self.kick_samples) - 1)
        else:
            # Fixed sample selection
            sample_index = self.fixed_sample_index if self.fixed_sample_index >= 0 else 0
        
        selected_sample = self.kick_samples[sample_index]
        sample_path = self.kick_sample_paths[sample_index] if sample_index < len(self.kick_sample_paths) else f"sample_{sample_index}"
        
        # Update the sample object with new sample data and velocity-based amplitude
        available_obj.sample_data = selected_sample
        available_obj.set_amplitude(velocity_based_amplitude)
        
        # Set position to current rigid body position (X, Z coordinates)
        if self.active_rigid_body in self.current_positions:
            pos_3d = self.current_positions[self.active_rigid_body]
            pos_2d = np.array([pos_3d[0], pos_3d[2]])  # X, Z coordinates
            available_obj.set_position(pos_2d)
        
        # Trigger the sample
        available_obj.trigger()
        
        print(f"Triggered kick sample: {sample_path} (velocity: {current_velocity:.2f} m/s, volume: {velocity_based_amplitude:.2f})")
    
    def calculate_velocity_amplitude(self, velocity_magnitude):
        """Calculate amplitude based on velocity magnitude."""
        if velocity_magnitude <= self.velocity_threshold:
            return 0.0  # No sound if below threshold
        
        # Map velocity to amplitude range
        # velocity_threshold -> min_velocity_volume
        # max_velocity_for_volume -> max_velocity_volume
        velocity_ratio = (velocity_magnitude - self.velocity_threshold) / (self.max_velocity_for_volume - self.velocity_threshold)
        velocity_ratio = np.clip(velocity_ratio, 0.0, 1.0)  # Clamp to [0, 1]
        
        # Linear interpolation between min and max volume
        amplitude = self.min_velocity_volume + velocity_ratio * (self.max_velocity_volume - self.min_velocity_volume)
        
        # Apply base sample amplitude scaling
        amplitude *= self.sample_amplitude
        
        return amplitude
    
    def manual_trigger_test(self):
        """Manually trigger a sample for testing purposes."""
        # Simulate a peak velocity for testing
        test_peak_velocity = 5.0
        self.trigger_kick_sample_with_peak_velocity(test_peak_velocity)
        return f"Manual trigger activated with peak velocity: {test_peak_velocity:.1f} m/s"
        
    def start_audio(self):
        """Start the audio generation loop."""
        if self.is_running:
            return "Audio already running"
            
        self.is_running = True
        self.audio_thread = threading.Thread(
            target=self._audio_loop, daemon=True
        )
        self.audio_thread.start()
        return "Audio started"
    
    def stop_audio(self):
        """Stop the audio generation loop."""
        if not self.is_running:
            return "Audio not running"
            
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        return "Audio stopped"
    
    def _audio_loop(self):
        """Main audio generation loop running in separate thread."""
        try:
            # Create sound streamer when audio starts
            self.sound_streamer = SoundNetworkStreamer()
            
            # Implement precise real-time timing
            chunk_duration = CHUNKSIZE / SAMPLING_RATE
            start_time = time.perf_counter()
            chunk_counter = 0
            
            for chunk in self.scene.run():
                if not self.is_running:
                    break
                
                # Update velocity tracking and trigger samples
                self.update_from_motion()
                    
                # Clip audio to prevent overflow
                chunk = np.clip(chunk, -1, 1)
                self.sound_streamer.send(chunk)
                chunk_counter += 1
                
                # Schedule next chunk send time (precise timing)
                next_time = start_time + chunk_counter * chunk_duration
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Audio loop error: {e}")
        finally:
            self.is_running = False
            self.sound_streamer = None
    
    def set_active_rigid_body(self, body_name):
        """Set which rigid body to track for velocity triggering."""
        if body_name in ['A', 'B', 'C', 'D']:
            self.active_rigid_body = body_name
            return f"Active Rigid Body: {body_name}"
        return f"Invalid rigid body: {body_name}"
    
    def update_velocity_threshold(self, threshold):
        """Update velocity threshold for triggering."""
        self.velocity_threshold = threshold
        return f"Velocity Threshold: {threshold:.2f} m/s"
    
    def update_min_trigger_interval(self, interval):
        """Update minimum time between triggers."""
        self.min_time_between_triggers = interval
        return f"Min Trigger Interval: {interval:.3f} s"
    
    def update_min_velocity_volume(self, volume):
        """Update minimum volume at threshold velocity."""
        self.min_velocity_volume = volume
        return f"Min Velocity Volume: {volume:.2f}"
    
    def update_max_velocity_volume(self, volume):
        """Update maximum volume at max velocity."""
        self.max_velocity_volume = volume
        return f"Max Velocity Volume: {volume:.2f}"
    
    def update_max_velocity_for_volume(self, velocity):
        """Update velocity at which maximum volume is reached."""
        self.max_velocity_for_volume = velocity
        return f"Max Velocity for Volume: {velocity:.1f} m/s"
    
    def update_x_plane_position(self, x_position):
        """Update X-plane position for crossing detection."""
        self.x_plane_position = x_position
        return f"X-Plane Position: {x_position:.2f}"
    
    def toggle_random_samples(self, use_random):
        """Toggle between random and fixed sample selection."""
        self.use_random_samples = use_random
        if use_random:
            return "🎲 Random Sample Mode: Every trigger uses a different kick"
        else:
            fixed_name = os.path.basename(self.fixed_sample_path) if self.fixed_sample_index >= 0 else "First sample"
            return f"📌 Fixed Sample Mode: Using {fixed_name}"
    
    def update_sample_amplitude(self, amplitude):
        """Update sample playback amplitude."""
        self.sample_amplitude = amplitude
        for obj in self.sample_objects:
            obj.set_amplitude(amplitude)
        return f"Sample Amplitude: {amplitude:.2f}"
    
    def toggle_mute(self, is_muted):
        """Toggle mute on/off."""
        self.is_muted = is_muted
        if is_muted:
            return "🔇 Muted"
        else:
            return "🔊 Unmuted"
    
    def get_status(self):
        """Get current status information."""
        mute_status = "🔇 Muted" if self.is_muted else "🔊 Unmuted"
        tracking_status = ("Connected" if self.rigid_bodies else "Disconnected")
        
        # Current velocity info
        current_velocity = np.array([0.0, 0.0, 0.0])
        velocity_magnitude = 0.0
        
        if self.active_rigid_body in self.current_velocities:
            current_velocity = self.current_velocities[self.active_rigid_body]
            velocity_magnitude = np.linalg.norm(current_velocity)
        
        # Playing samples count
        playing_samples = sum(1 for obj in self.sample_objects if obj.is_playing_sample())
        
        # Position info
        position_info = "No position data"
        if self.active_rigid_body in self.current_positions:
            pos = self.current_positions[self.active_rigid_body]
            position_info = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        
        status = f"""Status: {'Running' if self.is_running else 'Stopped'}
Mute: {mute_status}
OptiTrack: {tracking_status}

Active Rigid Body: {self.active_rigid_body}
Position: {position_info}
Current Velocity: {velocity_magnitude:.3f} m/s (threshold: {self.velocity_threshold:.2f})
Last Crossing Velocity: {self.crossing_velocities.get(self.active_rigid_body, 0.0):.3f} m/s
Velocity Vector: [{current_velocity[0]:.2f}, {current_velocity[1]:.2f}, {current_velocity[2]:.2f}]
X-Position: {self.current_positions.get(self.active_rigid_body, np.array([0,0,0]))[0]:.3f} (plane: {self.x_plane_position:.1f})

Samples:
  Loaded: {len(self.kick_samples)}
  Currently Playing: {playing_samples}/{self.max_concurrent_samples}
  Sample Amplitude: {self.sample_amplitude:.2f}
  Selection Mode: {'🎲 Random' if self.use_random_samples else '📌 Fixed (35)'}
  
X-Plane Crossing System:
  Velocity Threshold: {self.velocity_threshold:.2f} m/s
  Detection Method: X-plane crossing (negative → positive)
  X-Plane Position: {self.x_plane_position:.1f}
  Min Trigger Interval: {self.min_time_between_triggers:.3f} s
  Last Trigger: {time.time() - self.last_trigger_time.get(self.active_rigid_body, 0):.2f}s ago

Velocity-to-Volume Mapping:
  Min Volume (at threshold): {self.min_velocity_volume:.2f}
  Max Volume (at max velocity): {self.max_velocity_volume:.2f}
  Max Velocity for Volume: {self.max_velocity_for_volume:.1f} m/s
  Volume Calculation: Linear mapping from threshold to max velocity"""
        
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    if not HAS_GRADIO:
        print("Cannot create interface: gradio not available")
        return None
    
    controller = VelocityTriggerController()
    
    with gr.Blocks(title="Velocity-Triggered Kick Drum Sampler") as interface:
        gr.Markdown("# Velocity-Triggered Kick Drum Sampler")
        gr.Markdown(
            "Track rigid body velocity and trigger random kick drum samples when velocity exceeds threshold. "
            "**Select which rigid body to track** and **adjust velocity threshold** for triggering. "
            "Samples are randomly selected from the kick drum collection and positioned based on rigid body location."
        )
        
        with gr.Row():
            with gr.Column():
                # Control buttons
                start_btn = gr.Button("Start Audio", variant="primary")
                stop_btn = gr.Button("Stop Audio", variant="secondary")
                
                # Mute control
                mute_checkbox = gr.Checkbox(
                    label="🔇 Mute", value=False
                )
                
                # Rigid body selection
                gr.Markdown("### Rigid Body Selection")
                rigid_body_dropdown = gr.Dropdown(
                    choices=['A', 'B', 'C', 'D'],
                    value='C',
                    label="Active Rigid Body (for velocity tracking)"
                )
                
                # Velocity parameters
                gr.Markdown("### Velocity Trigger Settings")
                velocity_threshold_slider = gr.Slider(
                    minimum=0.1, maximum=10.0, value=3.0, step=0.1,
                    label="Velocity Threshold (m/s)"
                )
                
                x_plane_slider = gr.Slider(
                    minimum=-1.0, maximum=1.0, value=0.0, step=0.01,
                    label="X-Plane Position (crossing trigger)"
                )
                
                min_trigger_interval_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=1.0, step=0.01,
                    label="Minimum Time Between Triggers (s)"
                )
                
                # Velocity-to-volume mapping
                gr.Markdown("### Velocity-to-Volume Mapping")
                gr.Markdown(
                    "**Control how velocity affects volume:** Higher velocity = louder kick drums"
                )
                
                min_velocity_volume_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Minimum Volume (at threshold velocity)"
                )
                
                max_velocity_volume_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                    label="Maximum Volume (at max velocity)"
                )
                
                max_velocity_for_volume_slider = gr.Slider(
                    minimum=2.0, maximum=20.0, value=10.0, step=0.5,
                    label="Max Velocity for Volume (m/s)"
                )
                
                # Audio parameters
                gr.Markdown("### Audio Parameters")
                sample_amplitude_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.8, step=0.05,
                    label="Base Sample Amplitude"
                )
                
                # Sample selection mode
                gr.Markdown("### Sample Selection")
                random_samples_checkbox = gr.Checkbox(
                    value=False,
                    label="Use Random Samples",
                    info="When checked: random kick on each trigger. When unchecked: always use Kicks_StayOnBeat.com (35).wav"
                )
                
                # Manual trigger button for testing
                gr.Markdown("### Manual Testing")
                manual_trigger_btn = gr.Button("Manual Trigger", variant="secondary")
                manual_test_btn = gr.Button("Test Trigger", variant="secondary")
                
            with gr.Column():
                # Status and feedback
                status_output = gr.Textbox(
                    label="Status", 
                    value=controller.get_status(),
                    lines=20
                )
                
                # Parameter feedback
                rigid_body_output = gr.Textbox(
                    label="Active Rigid Body Status", value="Active Rigid Body: C"
                )
                velocity_threshold_output = gr.Textbox(
                    label="Velocity Threshold Status", value="Velocity Threshold: 3.00 m/s"
                )
                x_plane_output = gr.Textbox(
                    label="X-Plane Status", value="X-Plane Position: 0.00"
                )
                min_trigger_interval_output = gr.Textbox(
                    label="Min Trigger Interval Status", value="Min Trigger Interval: 1.000 s"
                )
                
                # Velocity-to-volume status outputs
                min_velocity_volume_output = gr.Textbox(
                    label="Min Velocity Volume Status", value="Min Velocity Volume: 0.20"
                )
                max_velocity_volume_output = gr.Textbox(
                    label="Max Velocity Volume Status", value="Max Velocity Volume: 1.00"
                )
                max_velocity_for_volume_output = gr.Textbox(
                    label="Max Velocity for Volume Status", value="Max Velocity for Volume: 10.0 m/s"
                )
                
                sample_amplitude_output = gr.Textbox(
                    label="Base Sample Amplitude Status", value="Sample Amplitude: 0.80"
                )
                
                # Sample selection status
                random_samples_output = gr.Textbox(
                    label="Sample Selection Status", value="📌 Fixed Sample Mode: Using Kicks_StayOnBeat.com (35).wav"
                )
                
                mute_output = gr.Textbox(
                    label="Mute Status", value="🔊 Unmuted"
                )
        
        # Event handlers
        start_btn.click(
            controller.start_audio,
            outputs=status_output
        )
        
        stop_btn.click(
            controller.stop_audio,
            outputs=status_output
        )
        
        # Parameter update handlers
        rigid_body_dropdown.change(
            controller.set_active_rigid_body,
            inputs=rigid_body_dropdown,
            outputs=rigid_body_output
        )
        
        velocity_threshold_slider.change(
            controller.update_velocity_threshold,
            inputs=velocity_threshold_slider,
            outputs=velocity_threshold_output
        )
        
        x_plane_slider.change(
            controller.update_x_plane_position,
            inputs=x_plane_slider,
            outputs=x_plane_output
        )
        
        min_trigger_interval_slider.change(
            controller.update_min_trigger_interval,
            inputs=min_trigger_interval_slider,
            outputs=min_trigger_interval_output
        )
        
        # Velocity-to-volume mapping event handlers
        min_velocity_volume_slider.change(
            controller.update_min_velocity_volume,
            inputs=min_velocity_volume_slider,
            outputs=min_velocity_volume_output
        )
        
        max_velocity_volume_slider.change(
            controller.update_max_velocity_volume,
            inputs=max_velocity_volume_slider,
            outputs=max_velocity_volume_output
        )
        
        max_velocity_for_volume_slider.change(
            controller.update_max_velocity_for_volume,
            inputs=max_velocity_for_volume_slider,
            outputs=max_velocity_for_volume_output
        )
        
        sample_amplitude_slider.change(
            controller.update_sample_amplitude,
            inputs=sample_amplitude_slider,
            outputs=sample_amplitude_output
        )
        
        # Sample selection mode event handler
        random_samples_checkbox.change(
            controller.toggle_random_samples,
            inputs=random_samples_checkbox,
            outputs=random_samples_output
        )
        
        mute_checkbox.change(
            controller.toggle_mute,
            inputs=mute_checkbox,
            outputs=mute_output
        )
        
        # Manual trigger for testing
        manual_trigger_btn.click(
            controller.trigger_kick_sample,
            outputs=None
        )
        
        manual_test_btn.click(
            controller.manual_trigger_test,
            outputs=None
        )
        
        # Manual status refresh button
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            controller.get_status,
            outputs=status_output
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Velocity-Triggered Kick Drum Sampler")
    print(f"Audio settings: {SAMPLING_RATE} Hz, {CHUNKSIZE} samples per chunk")
    print("Velocity-based triggering:")
    print("  - Track rigid body velocity magnitude")
    print("  - Trigger random kick drum samples when velocity exceeds threshold")
    print("  - Samples positioned based on rigid body X,Z coordinates")
    print("  - Multiple concurrent samples supported")
    
    if HAS_GRADIO:
        print("Open your web browser to control settings and monitor status")
    interface = create_interface()
        if interface:
    interface.launch(
        server_name=lt.get_local_ip(),  # Listen on specific IP address
                server_port=7862,  # Different port from original
        share=False,  # Set to True if you want a public link
        show_api=False
    ) 
    else:
        print("Running without web interface (gradio not available)")
        print("Creating controller for direct use...")
        controller = VelocityTriggerController()
        print("Controller created. You can use it programmatically.")
        print("Example:")
        print("  controller.start_audio()")
        print("  # ... wait for motion ...")
        print("  controller.stop_audio()")