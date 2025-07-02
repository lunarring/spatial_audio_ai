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
    with motion control for multiple rigid bodies."""
    
    def __init__(self):
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.3
        
        # Create four sine wave objects for rigid bodies A, B, C, D
        self.sine_objects = {}
        self.rigid_bodies = {}
        self.active_objects = {}  # Track which objects are active
        
        rigid_body_names = ['A', 'B', 'C', 'D']
        base_frequencies = [62.5, 125.0, 250.0, 500.0]  # Different base frequencies for each
        
        for i, name in enumerate(rigid_body_names):
            sine_obj = SO_PlaybackSine()
            sine_obj.set_frequency(base_frequencies[i])
            sine_obj.set_amplitude(0.3)  # Lower amplitude since we have 4 objects
            self.sine_objects[name] = sine_obj
            self.scene.register(sine_obj)
            self.active_objects[name] = True  # Start with all active
        
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        self.is_muted = False
        self.unmuted_amplitude = 0.3  # Lower for 4 objects
        
        # OptiTrack setup
        self.motive = None
        self.setup_optitrack()
        
        # Frequency mapping parameters (0m = 62.5Hz, 2m = 500Hz) - 3 octaves
        self.min_height = 0.0
        self.max_height = 2.0
        self.min_frequency = 62.5  # Base frequency
        self.max_frequency = 500.0  # 3 octaves higher (62.5 * 2^3)
        
        # Position and tracking data for each rigid body
        self.current_heights = {name: 0.0 for name in rigid_body_names}
        self.current_positions = {name: np.array([0.0, 0.0]) for name in rigid_body_names}
        self.position_scale = 1.0  # Scaling factor for position
        
    def setup_optitrack(self):
        """Initialize OptiTrack connection and all rigid bodies."""
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
                return
            
            # Create rigid bodies A, B, C, D
            rigid_body_names = ['A', 'B', 'C', 'D']
            for name in rigid_body_names:
                self.rigid_bodies[name] = RigidBody(self.motive, name)
            
            print(f"OptiTrack connection established with rigid bodies: {', '.join(rigid_body_names)}")
            
        except Exception as e:
            print(f"Failed to setup OptiTrack: {e}")
            self.motive = None
            self.rigid_bodies = {}
    
    def update_from_motion(self):
        """Update frequency and position based on rigid body motion for all active objects."""
        if not self.rigid_bodies or self.motive is None:
            return
            
        try:
            # Get latest data like the motive_receiver example
            latest_data = self.motive.get_last()
            if not latest_data:
                return
            
            # Update all rigid bodies and their corresponding sine objects
            for name, rigid_body in self.rigid_bodies.items():
                if not self.active_objects[name]:
                    # Set amplitude to 0 for inactive objects
                    self.sine_objects[name].set_amplitude(0.0)
                    continue
                
                try:
                    # Update rigid body and get position
                    rigid_body.update()
                    position = rigid_body.positions.get_last()
                    
                    if position is not None:
                        # Extract coordinates: X, Y (height), Z
                        x_pos = position[0]  # X coordinate
                        self.current_heights[name] = position[1]  # Y coordinate (height)
                        z_pos = position[2]  # Z coordinate
                        
                        # Update position (X and Z coordinates map to sound object X and Y)
                        # Apply scaling factor to the position
                        self.current_positions[name] = np.array([x_pos, z_pos])
                        scaled_position = self.current_positions[name] * self.position_scale
                        self.sine_objects[name].set_position(scaled_position)
                        
                        # Update frequency based on height (Y coordinate)
                        # Clamp height to valid range
                        height_clamped = np.clip(
                            self.current_heights[name], self.min_height, self.max_height
                        )
                        
                        # Exponential (octave-based) interpolation
                        height_ratio = (
                            (height_clamped - self.min_height) / 
                            (self.max_height - self.min_height)
                        )
                        # Calculate frequency using exponential mapping (octaves)
                        # frequency = min_freq * 2^(height_ratio * num_octaves)
                        num_octaves = np.log2(self.max_frequency / self.min_frequency)
                        frequency = self.min_frequency * (2 ** (height_ratio * num_octaves))
                        
                        # Update sine wave frequency
                        self.sine_objects[name].set_frequency(frequency)
                
                except Exception as e:
                    print(f"Error updating rigid body {name}: {e}")
                    
        except Exception as e:
            print(f"Error in motion update: {e}")
        
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
                
                # Update frequency and position from motion data
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
    
    def activate_rigid_body(self, name, is_active):
        """Activate or deactivate a rigid body."""
        self.active_objects[name] = is_active
        if not is_active:
            # Mute the sine object when deactivated
            self.sine_objects[name].set_amplitude(0.0)
        elif not self.is_muted:
            # Restore amplitude when activated (if not globally muted)
            self.sine_objects[name].set_amplitude(self.unmuted_amplitude)
        
        status = "Active" if is_active else "Inactive"
        return f"Rigid Body {name}: {status}"
    
    def update_amplitude(self, amplitude):
        """Update sine wave amplitude for all objects."""
        self.unmuted_amplitude = amplitude  # Always store the unmuted value
        if not self.is_muted:
            # Apply to all active objects
            for name, is_active in self.active_objects.items():
                if is_active:
                    self.sine_objects[name].set_amplitude(amplitude)
        return f"Amplitude: {amplitude:.2f}"
    
    def update_phase(self, phase):
        """Update sine wave phase offset for all objects."""
        for sine_obj in self.sine_objects.values():
            sine_obj.set_phase_offset(phase)
        return f"Phase: {phase:.2f} rad"
    
    def update_position_scale(self, scale):
        """Update position scaling factor."""
        self.position_scale = scale
        # Re-apply current positions with new scaling for all active objects
        for name, is_active in self.active_objects.items():
            if is_active and name in self.current_positions:
                scaled_position = self.current_positions[name] * self.position_scale
                self.sine_objects[name].set_position(scaled_position)
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
        """Update smoothing factor for all objects."""
        for sine_obj in self.sine_objects.values():
            sine_obj.set_smoothing_factor(smoothing)
        return f"Smoothing: {smoothing:.2f}"
    
    def toggle_mute(self, is_muted):
        """Toggle mute on/off for all objects."""
        self.is_muted = is_muted
        if is_muted:
            # Mute all objects
            for sine_obj in self.sine_objects.values():
                sine_obj.set_amplitude(0.0)
            return "🔇 Muted"
        else:
            # Unmute only active objects
            for name, is_active in self.active_objects.items():
                if is_active:
                    self.sine_objects[name].set_amplitude(self.unmuted_amplitude)
            return "🔊 Unmuted"
    
    def get_status(self):
        """Get current status information."""
        mute_status = "🔇 Muted" if self.is_muted else "🔊 Unmuted"
        tracking_status = ("Connected" if self.rigid_bodies else "Disconnected")
        
        freq_range = f"{self.min_frequency:.0f} Hz - {self.max_frequency:.0f} Hz"
        
        # Build active objects summary
        active_list = [name for name, is_active in self.active_objects.items() if is_active]
        active_summary = ", ".join(active_list) if active_list else "None"
        
        # Build detailed info for each rigid body
        body_details = []
        for name in ['A', 'B', 'C', 'D']:
            is_active = self.active_objects[name]
            if is_active and name in self.sine_objects:
                pos = self.sine_objects[name].get_position()
                current_freq = self.sine_objects[name].current_frequency
                current_amp = self.sine_objects[name].current_amplitude
                rb_pos = self.current_positions[name]
                height = self.current_heights[name]
                
                body_details.append(
                    f"  {name}: {'Active' if is_active else 'Inactive'} | "
                    f"Pos: ({rb_pos[0]:.2f}, {height:.2f}, {rb_pos[1]:.2f}) | "
                    f"Freq: {current_freq:.1f} Hz | Amp: {current_amp:.2f}"
                )
            else:
                body_details.append(f"  {name}: Inactive")
        
        # Get smoothing from first object (they should all be the same)
        smoothing = list(self.sine_objects.values())[0].smoothing_factor if self.sine_objects else 0.0
        
        status = f"""Status: {'Running' if self.is_running else 'Stopped'}
Mute: {mute_status}
OptiTrack: {tracking_status}

Active Rigid Bodies: {active_summary}
Position Scale Factor: {self.position_scale:.2f}
Frequency Range: {freq_range}

Global Settings:
  Amplitude: {self.unmuted_amplitude:.2f} (stored)
  Smoothing: {smoothing:.2f}

Rigid Body Details:
{chr(10).join(body_details)}"""
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    controller = SineWaveMotionController()
    
    with gr.Blocks(title="Real-time Sine Wave Motion Control") as interface:
        gr.Markdown("# Real-time Multi-Body Sine Wave Motion Control")
        gr.Markdown(
            "Control up to 4 real-time generated sine waves with spatial positioning. "
            "**Frequency is controlled by OptiTrack rigid body height** "
            "(adjustable frequency range with exponential mapping). "
            "**Position is controlled by X and Z coordinates** of each rigid body."
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
                
                # Rigid body activation
                gr.Markdown("### Rigid Body Activation")
                rb_a_checkbox = gr.Checkbox(label="Rigid Body A", value=True)
                rb_b_checkbox = gr.Checkbox(label="Rigid Body B", value=True)
                rb_c_checkbox = gr.Checkbox(label="Rigid Body C", value=True)
                rb_d_checkbox = gr.Checkbox(label="Rigid Body D", value=True)
                
                # Parameter controls (no frequency slider - motion controlled)
                gr.Markdown("### Audio Parameters")
                gr.Markdown(
                    "**Frequency:** Motion Controlled (Rigid Body Height - Exponential Mapping)"
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
                    lines=12
                )
                
                # Parameter feedback
                min_freq_output = gr.Textbox(
                    label="Min Frequency Status", value="Min Frequency: 62.5 Hz"
                )
                max_freq_output = gr.Textbox(
                    label="Max Frequency Status", value="Max Frequency: 500.0 Hz"
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
                
                # Rigid body status outputs
                rb_a_output = gr.Textbox(
                    label="Rigid Body A Status", value="Rigid Body A: Active"
                )
                rb_b_output = gr.Textbox(
                    label="Rigid Body B Status", value="Rigid Body B: Active"
                )
                rb_c_output = gr.Textbox(
                    label="Rigid Body C Status", value="Rigid Body C: Active"
                )
                rb_d_output = gr.Textbox(
                    label="Rigid Body D Status", value="Rigid Body D: Active"
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
        
        # Rigid body activation handlers
        rb_a_checkbox.change(
            lambda x: controller.activate_rigid_body('A', x),
            inputs=rb_a_checkbox,
            outputs=rb_a_output
        )
        
        rb_b_checkbox.change(
            lambda x: controller.activate_rigid_body('B', x),
            inputs=rb_b_checkbox,
            outputs=rb_b_output
        )
        
        rb_c_checkbox.change(
            lambda x: controller.activate_rigid_body('C', x),
            inputs=rb_c_checkbox,
            outputs=rb_c_output
        )
        
        rb_d_checkbox.change(
            lambda x: controller.activate_rigid_body('D', x),
            inputs=rb_d_checkbox,
            outputs=rb_d_output
        )
        
        # Manual status refresh button
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            controller.get_status,
            outputs=status_output
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Real-time Multi-Body Sine Wave Motion Control Interface")
    print(f"Audio settings: {SAMPLING_RATE} Hz, {CHUNKSIZE} samples per chunk")
    print("Motion control by OptiTrack Rigid Bodies A, B, C, D:")
    print("  Y-axis (height): 0m = 62.5 Hz, 2m = 500 Hz (3 octaves, exponential)")
    print("  X-axis and Z-axis: Control spatial position of each sound object")
    print("  Base frequencies: A=62.5Hz, B=125Hz, C=250Hz, D=500Hz")
    print("Open your web browser to control activation and parameters")
    
    interface = create_interface()
    interface.launch(
        server_name=lt.get_local_ip(),  # Listen on specific IP address
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_api=False
    ) 