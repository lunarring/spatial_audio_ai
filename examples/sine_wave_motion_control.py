#!/usr/bin/env python3
"""
Real-time Sine Wave Motion Control with Gradio Interface

This script provides a real-time controllable sine wave generator using the 
SO_PlaybackSine class with a Gradio web interface for parameter control.
The frequency is controlled by the height of OptiTrack rigid body "A".
"""

import numpy as np
import time
import threading
import gradio as gr
from spatial_audio_ai.tools.spatializer import (
    SO_PlaybackSine, 
    Spatializer, 
    Scene,
    CHUNKSIZE,
    SAMPLING_RATE
)
from spatial_audio_ai.tools.client import SoundNetworkStreamer
import lunar_tools as lt
from optitrack_python.rigid_body import RigidBody
from optitrack_python.motive_receiver import MotiveReceiver


class SineWaveMotionController:
    """Controller class for managing real-time sine wave generation 
    with motion control."""
    
    def __init__(self):
        self.sine_object = SO_PlaybackSine()
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.3
        self.scene.register(self.sine_object)
        
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        self.is_muted = False
        self.unmuted_amplitude = 0.5  # Store original amplitude when muted
        
        # OptiTrack setup
        self.motive = None
        self.rigid_body = None
        self.setup_optitrack()
        
        # Frequency mapping parameters (0m = 50Hz, 2m = 15KHz)
        self.min_height = 0.0
        self.max_height = 2.0
        self.min_frequency = 50.0
        self.max_frequency = 15000.0
        self.current_height = 0.0
        
    def setup_optitrack(self):
        """Initialize OptiTrack connection and rigid body."""
        try:
            # Use the exact same setup as the working motive_receiver.py example
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
                self.rigid_body = None
                return
            
            # Create a single rigid body "B" for demonstration
            self.rigid_body = RigidBody(self.motive, "B")
            print("OptiTrack connection established with rigid body 'B'")
            
        except Exception as e:
            print(f"Failed to setup OptiTrack: {e}")
            self.motive = None
            self.rigid_body = None
    
    def update_frequency_from_motion(self):
        """Update frequency based on rigid body height."""
        if self.rigid_body is None or self.motive is None:
            return 440.0  # Default frequency if no tracking
            
        try:
            # Get latest data like the motive_receiver example
            latest_data = self.motive.get_last()
            if not latest_data:
                return self.sine_object.target_frequency
            
            # Update rigid body and get position
            self.rigid_body.update()
            position = self.rigid_body.positions.get_last()
            
            if position is not None:
                # Use Y coordinate (height) - index 1
                self.current_height = position[1]
                
                # Clamp height to valid range
                height_clamped = np.clip(
                    self.current_height, self.min_height, self.max_height
                )
                
                # Linear interpolation
                height_ratio = (
                    (height_clamped - self.min_height) / 
                    (self.max_height - self.min_height)
                )
                frequency = (
                    self.min_frequency + 
                    height_ratio * (self.max_frequency - self.min_frequency)
                )
                
                # Update sine wave frequency
                self.sine_object.set_frequency(frequency)
                return frequency
        except Exception as e:
            print(f"Error updating frequency from motion: {e}")
            
        return self.sine_object.target_frequency
        
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
                
                # Update frequency from motion data
                self.update_frequency_from_motion()
                    
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
    
    def update_amplitude(self, amplitude):
        """Update sine wave amplitude."""
        self.unmuted_amplitude = amplitude  # Always store the unmuted value
        if not self.is_muted:
            self.sine_object.set_amplitude(amplitude)
        return f"Amplitude: {amplitude:.2f}"
    
    def update_phase(self, phase):
        """Update sine wave phase offset."""
        self.sine_object.set_phase_offset(phase)
        return f"Phase: {phase:.2f} rad"
    
    def update_position_x(self, x):
        """Update X position."""
        current_pos = self.sine_object.get_position()
        new_pos = np.array([x, current_pos[1]], dtype=float)
        self.sine_object.set_position(new_pos)
        return f"X Position: {x:.1f}"
    
    def update_position_y(self, y):
        """Update Y position."""
        current_pos = self.sine_object.get_position()
        new_pos = np.array([current_pos[0], y], dtype=float)
        self.sine_object.set_position(new_pos)
        return f"Y Position: {y:.1f}"
    
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
            self.sine_object.set_amplitude(self.unmuted_amplitude)
            return "🔊 Unmuted"
    
    def get_status(self):
        """Get current status information."""
        pos = self.sine_object.get_position()
        target_pos = self.sine_object.target_position
        
        mute_status = "🔇 Muted" if self.is_muted else "🔊 Unmuted"
        tracking_status = ("Connected" if self.rigid_body is not None
                           else "Disconnected")
        
        freq_range = f"{self.min_frequency:.0f} Hz - {self.max_frequency:.0f} Hz"
        
        status = f"""Status: {'Running' if self.is_running else 'Stopped'}
Mute: {mute_status}
OptiTrack: {tracking_status}

Motion Control:
  Current Height: {self.current_height:.3f} m
  Frequency Range: {freq_range}

Target Values:
  Frequency: {self.sine_object.target_frequency:.1f} Hz (motion controlled)
  Amplitude: {self.unmuted_amplitude:.2f} (stored)
  Phase: {self.sine_object.target_phase_offset:.2f} rad
  Position: ({target_pos[0]:.1f}, {target_pos[1]:.1f})

Current Values (Smoothed):
  Frequency: {self.sine_object.current_frequency:.1f} Hz
  Amplitude: {self.sine_object.current_amplitude:.2f}
  Phase: {self.sine_object.current_phase_offset:.2f} rad
  Position: ({pos[0]:.1f}, {pos[1]:.1f})

Smoothing: {self.sine_object.smoothing_factor:.2f}"""
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    controller = SineWaveMotionController()
    
    with gr.Blocks(title="Real-time Sine Wave Motion Control") as interface:
        gr.Markdown("# Real-time Sine Wave Motion Control")
        gr.Markdown(
            "Control a real-time generated sine wave with spatial positioning. "
            "**Frequency is controlled by OptiTrack rigid body 'B' height** "
            "(0m = 50Hz, 2m = 15KHz)"
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
                
                # Parameter controls (no frequency slider - motion controlled)
                gr.Markdown("### Audio Parameters")
                gr.Markdown(
                    "**Frequency:** Motion Controlled (Rigid Body 'B' Height)"
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
                
                x_slider = gr.Slider(
                    minimum=-10, maximum=10, value=0, step=0.1,
                    label="X Position"
                )
                
                y_slider = gr.Slider(
                    minimum=-10, maximum=10, value=0, step=0.1,
                    label="Y Position"
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
                    lines=12
                )
                
                # Parameter feedback (no frequency output - motion controlled)
                amp_output = gr.Textbox(
                    label="Amplitude Status", value="Amplitude: 0.50"
                )
                phase_output = gr.Textbox(
                    label="Phase Status", value="Phase: 0.00 rad"
                )
                x_output = gr.Textbox(
                    label="X Position Status", value="X Position: 0.0"
                )
                y_output = gr.Textbox(
                    label="Y Position Status", value="Y Position: 0.0"
                )
                smoothing_output = gr.Textbox(
                    label="Smoothing Status", value="Smoothing: 0.95"
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
        
        # Parameter update handlers (no frequency handler - motion controlled)
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
        
        x_slider.change(
            controller.update_position_x,
            inputs=x_slider,
            outputs=x_output
        )
        
        y_slider.change(
            controller.update_position_y,
            inputs=y_slider,
            outputs=y_output
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
    print("Frequency controlled by OptiTrack Rigid Body 'B' height:")
    print("  0m height = 50 Hz")
    print("  2m height = 15,000 Hz")
    print("Open your web browser to control the sine wave parameters")
    
    interface = create_interface()
    interface.launch(
        server_name=lt.get_local_ip(),  # Listen on specific IP address
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_api=False
    ) 