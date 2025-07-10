#!/usr/bin/env python3
"""
Real-time Sine Wave Motion Control with Gradio Interface

This script provides a real-time controllable sine wave generator using the 
SO_PlaybackSine class with a Gradio web interface for parameter control.
The frequency is controlled by the height of OptiTrack rigid body "C".
"""

import numpy as np
import time
import threading
import gradio as gr
from spatial_audio_ai.tools.spatializer import (
    Spatializer, Scene, CHUNKSIZE, SAMPLING_RATE
)
from spatial_audio_ai.tools.sound_objects import SO_PlaybackSine
from spatial_audio_ai.tools.client import SoundNetworkStreamer
import lunar_tools as lt
from optitrack_python.rigid_body import RigidBody
from optitrack_python.motive_receiver import MotiveReceiver


class SineWaveMotionController:
    """Controller class for managing real-time sine wave generation 
    with motion control for rigid body C."""
    
    def __init__(self):
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.3
        
        # Create one sine wave object for rigid body C
        self.sine_object = SO_PlaybackSine()
        self.sine_object.set_frequency(250.0)  # Base frequency for C
        self.sine_object.set_amplitude(0.5)
        self.scene.register(self.sine_object)
        
        self.rigid_body = None
        self.is_active = True
        
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        self.motion_thread = None
        self.motion_update_active = False
        self.is_muted = False
        self.unmuted_amplitude = 0.5
        
        # OptiTrack setup
        self.motive = None
        self.setup_optitrack()
        
        # Frequency mapping parameters (0m = 62.5Hz, 2m = 500Hz)
        self.min_height = 0.0
        self.max_height = 2.0
        self.min_frequency = 62.5
        self.max_frequency = 500.0
        
        # Position and tracking data
        self.current_height = 0.0
        self.current_position = np.array([0.0, 0.0])
        self.position_scale = 1.0
        
        # Verbose mode for debugging
        self.verbose_mode = True
        self.last_verbose_time = 0
        self.verbose_interval = 0.5  # Print every 0.5 seconds
        
    def setup_optitrack(self):
        """Initialize OptiTrack connection and rigid body C."""
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
                    frame_id = latest_data['frame_id']
                    print(f"✓ Connection established! Frame ID: {frame_id}")
                    break
                time.sleep(0.1)
            else:
                print("✗ No data received. Check OptiTrack connection.")
                self.motive.stop()
                self.motive = None
                return
            
            # Create rigid body C
            self.rigid_body = RigidBody(self.motive, "C")
            print("OptiTrack connection established with rigid body C")
            
        except Exception as e:
            print(f"Failed to setup OptiTrack: {e}")
            self.motive = None
            self.rigid_body = None
    
    def print_verbose_status(self, position, frequency):
        """Print verbose debugging information."""
        current_time = time.time()
        if current_time - self.last_verbose_time >= self.verbose_interval:
            print("\n=== VERBOSE DEBUG (Rigid Body C) ===")
            print(f"Raw Position: {position}")
            print(f"X: {position[0]:.3f}, Y(height): {position[1]:.3f}, "
                  f"Z: {position[2]:.3f}")
            print(f"Current Height: {self.current_height:.3f}m")
            print(f"Height Range: {self.min_height}m - {self.max_height}m")
            print(f"Frequency Range: {self.min_frequency}Hz - "
                  f"{self.max_frequency}Hz")
            print(f"Calculated Frequency: {frequency:.2f}Hz")
            scaled_pos = self.current_position * self.position_scale
            print(f"Scaled Position (X,Z): {scaled_pos}")
            print(f"Active: {self.is_active}, Muted: {self.is_muted}")
            current_amp = self.sine_object.current_amplitude
            print(f"Current Amplitude: {current_amp:.3f}")
            print("=====================================\n")
            self.last_verbose_time = current_time
    
    def update_from_motion(self):
        """Update frequency and position based on rigid body C motion."""
        if not self.rigid_body or self.motive is None:
            return
            
        try:
            # Get latest data
            latest_data = self.motive.get_last()
            if not latest_data:
                return
            
            if not self.is_active:
                self.sine_object.set_amplitude(0.0)
                return
            
            try:
                # Update rigid body and get position
                self.rigid_body.update()
                position = self.rigid_body.positions.get_last()
                
                if position is not None:
                    # Extract coordinates: X, Y (height), Z
                    x_pos = position[0]
                    self.current_height = position[1]  # Y coordinate (height)
                    z_pos = position[2]
                    
                    # Update position (X and Z coordinates)
                    self.current_position = np.array([x_pos, z_pos])
                    scaled_position = self.current_position * self.position_scale
                    self.sine_object.set_position(scaled_position)
                    
                    # Update frequency based on height (Y coordinate)
                    height_clamped = np.clip(
                        self.current_height, self.min_height, self.max_height
                    )
                    
                    # Exponential (octave-based) interpolation
                    height_ratio = (
                        (height_clamped - self.min_height) / 
                        (self.max_height - self.min_height)
                    )
                    # Calculate frequency using exponential mapping
                    num_octaves = np.log2(
                        self.max_frequency / self.min_frequency
                    )
                    frequency = self.min_frequency * (
                        2 ** (height_ratio * num_octaves)
                    )
                    
                    # Update sine wave frequency
                    self.sine_object.set_frequency(frequency)
                    
                    # Print verbose debug info
                    if self.verbose_mode:
                        self.print_verbose_status(position, frequency)
                
            except Exception as e:
                print(f"Error updating rigid body C: {e}")
                    
        except Exception as e:
            print(f"Error in motion update: {e}")
    
    def _motion_update_loop(self):
        """Separate high-frequency motion update loop."""
        try:
            while self.motion_update_active:
                self.update_from_motion()
                # Run at ~100Hz for responsive motion tracking
                time.sleep(0.01)
        except Exception as e:
            print(f"Motion update loop error: {e}")
    
    def start_motion_updates(self):
        """Start the high-frequency motion update loop."""
        if self.motion_update_active:
            return
        
        self.motion_update_active = True
        self.motion_thread = threading.Thread(
            target=self._motion_update_loop, daemon=True
        )
        self.motion_thread.start()
        print("High-frequency motion updates started (100Hz)")
    
    def stop_motion_updates(self):
        """Stop the motion update loop."""
        if not self.motion_update_active:
            return
        
        self.motion_update_active = False
        if self.motion_thread:
            self.motion_thread.join(timeout=1.0)
        print("Motion updates stopped")
        
    def start_audio(self):
        """Start the audio generation loop."""
        if self.is_running:
            return "Audio already running"
            
        self.is_running = True
        self.audio_thread = threading.Thread(
            target=self._audio_loop, daemon=True
        )
        self.audio_thread.start()
        return "Audio started with synchronized motion tracking"
    
    def stop_audio(self):
        """Stop the audio generation loop."""
        if not self.is_running:
            return "Audio not running"
            
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        return "Audio and motion tracking stopped"
    
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
                
                # Update motion parameters immediately before each audio chunk
                # This eliminates the thread sync delay and ensures motion changes 
                # are reflected in the very next audio chunk (max 21ms latency)
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
    
    def activate_rigid_body(self, is_active):
        """Activate or deactivate rigid body C."""
        self.is_active = is_active
        if not is_active:
            self.sine_object.set_amplitude(0.0)
        elif not self.is_muted:
            self.sine_object.set_amplitude(self.unmuted_amplitude)
        
        status = "Active" if is_active else "Inactive"
        return f"Rigid Body C: {status}"
    
    def update_amplitude(self, amplitude):
        """Update sine wave amplitude."""
        self.unmuted_amplitude = amplitude
        if not self.is_muted and self.is_active:
            self.sine_object.set_amplitude(amplitude)
        return f"Amplitude: {amplitude:.2f}"
    
    def update_phase(self, phase):
        """Update sine wave phase offset."""
        self.sine_object.set_phase_offset(phase)
        return f"Phase: {phase:.2f} rad"
    
    def update_position_scale(self, scale):
        """Update position scaling factor."""
        self.position_scale = scale
        if self.is_active:
            scaled_position = self.current_position * self.position_scale
            self.sine_object.set_position(scaled_position)
        return f"Position Scale: {scale:.2f}"
    
    def update_min_frequency(self, min_freq):
        """Update minimum frequency."""
        self.min_frequency = min_freq
        return f"Min Frequency: {min_freq:.1f} Hz"
    
    def update_max_frequency(self, max_freq):
        """Update maximum frequency."""
        self.max_frequency = max_freq
        return f"Max Frequency: {max_freq:.1f} Hz"
    
    def update_smoothing(self, smoothing):
        """Update smoothing factor."""
        self.sine_object.set_smoothing_factor(smoothing)
        return f"Smoothing: {smoothing:.2f}"
    
    def toggle_mute(self, is_muted):
        """Toggle mute on/off."""
        self.is_muted = is_muted
        if is_muted:
            self.sine_object.set_amplitude(0.0)
            return "🔇 Muted"
        else:
            if self.is_active:
                self.sine_object.set_amplitude(self.unmuted_amplitude)
            return "🔊 Unmuted"
    
    def toggle_verbose(self, verbose):
        """Toggle verbose debug output."""
        self.verbose_mode = verbose
        status = "ON" if verbose else "OFF"
        return f"Verbose Debug: {status}"
    
    def get_status(self):
        """Get current status information."""
        mute_status = "🔇 Muted" if self.is_muted else "🔊 Unmuted"
        tracking_status = "Connected" if self.rigid_body else "Disconnected"
        
        freq_range = f"{self.min_frequency:.0f} Hz - {self.max_frequency:.0f} Hz"
        
        current_freq = self.sine_object.current_frequency
        current_amp = self.sine_object.current_amplitude
        smoothing = self.sine_object.smoothing_factor
        
        status = f"""Status: {'Running' if self.is_running else 'Stopped'}
Mute: {mute_status}
OptiTrack: {tracking_status}
Verbose Debug: {'ON' if self.verbose_mode else 'OFF'}

Rigid Body C: {'Active' if self.is_active else 'Inactive'}
Position: ({self.current_position[0]:.2f}, {self.current_height:.2f}, {self.current_position[1]:.2f})
Position Scale Factor: {self.position_scale:.2f}
Frequency Range: {freq_range}

Current Settings:
  Frequency: {current_freq:.1f} Hz
  Amplitude: {current_amp:.2f} (stored: {self.unmuted_amplitude:.2f})
  Smoothing: {smoothing:.2f}"""
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    controller = SineWaveMotionController()
    
    with gr.Blocks(title="Real-time Sine Wave Motion Control") as interface:
        gr.Markdown("# Real-time Sine Wave Motion Control (Rigid Body C)")
        gr.Markdown(
            "Control a real-time generated sine wave with spatial positioning. "
            "**Frequency is controlled by OptiTrack rigid body C height** "
            "(adjustable frequency range with exponential mapping). "
            "**Position is controlled by X and Z coordinates**."
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
                
                # Debug control
                verbose_checkbox = gr.Checkbox(
                    label="🔍 Verbose Debug Output", value=True
                )
                
                # Rigid body activation
                gr.Markdown("### Rigid Body Control")
                rb_c_checkbox = gr.Checkbox(label="Rigid Body C", value=True)
                
                # Parameter controls
                gr.Markdown("### Audio Parameters")
                gr.Markdown(
                    "**Frequency:** Motion Controlled "
                    "(Rigid Body Height - Exponential Mapping)"
                )
                
                # Frequency range controls
                min_freq_slider = gr.Slider(
                    minimum=20.0, maximum=1000.0, value=62.5, step=0.5,
                    label="Minimum Frequency (0m height)"
                )
                
                max_freq_slider = gr.Slider(
                    minimum=100.0, maximum=20000.0, value=500.0, step=10.0,
                    label="Maximum Frequency (2m height)"
                )
                
                amp_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                    label="Amplitude"
                )
                
                phase_slider = gr.Slider(
                    minimum=0.0, maximum=2*np.pi, value=0.0, step=0.1,
                    label="Phase Offset (rad)"
                )
                
                gr.Markdown("### Spatial Position")
                gr.Markdown(
                    "**Position:** Motion Controlled (Rigid Body X,Z coordinates)"
                )
                
                scale_slider = gr.Slider(
                    minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                    label="Position Scale Factor"
                )
                
                gr.Markdown("### Smoothing Control")
                
                smoothing_slider = gr.Slider(
                    minimum=0.0, maximum=0.99, value=0.95, step=0.01,
                    label="Smoothing Factor (higher = smoother)"
                )
                
            with gr.Column():
                # Status and feedback
                status_output = gr.Textbox(
                    label="Status", 
                    value=controller.get_status(),
                    lines=15
                )
                
                # Parameter feedback
                min_freq_output = gr.Textbox(
                    label="Min Frequency Status", 
                    value="Min Frequency: 62.5 Hz"
                )
                max_freq_output = gr.Textbox(
                    label="Max Frequency Status", 
                    value="Max Frequency: 500.0 Hz"
                )
                amp_output = gr.Textbox(
                    label="Amplitude Status", value="Amplitude: 0.50"
                )
                phase_output = gr.Textbox(
                    label="Phase Status", value="Phase: 0.00 rad"
                )
                scale_output = gr.Textbox(
                    label="Position Scale Status", value="Position Scale: 1.00"
                )
                smoothing_output = gr.Textbox(
                    label="Smoothing Status", value="Smoothing: 0.95"
                )
                mute_output = gr.Textbox(
                    label="Mute Status", value="🔊 Unmuted"
                )
                verbose_output = gr.Textbox(
                    label="Verbose Status", value="Verbose Debug: ON"
                )
                
                # Rigid body status output
                rb_c_output = gr.Textbox(
                    label="Rigid Body C Status", value="Rigid Body C: Active"
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
        min_freq_slider.change(
            controller.update_min_frequency,
            inputs=min_freq_slider,
            outputs=min_freq_output
        )
        
        max_freq_slider.change(
            controller.update_max_frequency,
            inputs=max_freq_slider,
            outputs=max_freq_output
        )
        
        amp_slider.change(
            controller.update_amplitude,
            inputs=amp_slider,
            outputs=amp_output
        )
        
        phase_slider.change(
            controller.update_phase,
            inputs=phase_slider,
            outputs=phase_output
        )
        
        scale_slider.change(
            controller.update_position_scale,
            inputs=scale_slider,
            outputs=scale_output
        )
        
        smoothing_slider.change(
            controller.update_smoothing,
            inputs=smoothing_slider,
            outputs=smoothing_output
        )
        
        mute_checkbox.change(
            controller.toggle_mute,
            inputs=mute_checkbox,
            outputs=mute_output
        )
        
        verbose_checkbox.change(
            controller.toggle_verbose,
            inputs=verbose_checkbox,
            outputs=verbose_output
        )
        
        # Rigid body activation handler
        rb_c_checkbox.change(
            controller.activate_rigid_body,
            inputs=rb_c_checkbox,
            outputs=rb_c_output
        )
        
        # Manual status refresh button
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            controller.get_status,
            outputs=status_output
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Real-time Sine Wave Motion Control Interface")
    print(f"Audio settings: {SAMPLING_RATE} Hz, {CHUNKSIZE} samples per chunk")
    print("Motion control by OptiTrack Rigid Body C:")
    print("  Y-axis (height): 0m = 62.5 Hz, 2m = 500 Hz (exponential)")
    print("  X-axis and Z-axis: Control spatial position")
    print("  Base frequency: C=250Hz")
    print("Open your web browser to control parameters")
    print("Verbose debug output will print position/frequency info to console")
    
    interface = create_interface()
    interface.launch(
        server_name=lt.get_local_ip(),
        server_port=7860,
        share=False,
        show_api=False
    ) 