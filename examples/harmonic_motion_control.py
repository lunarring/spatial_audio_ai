#!/usr/bin/env python3
"""
Real-time Multi-Harmonic Sine Wave Motion Control with Gradio Interface

This script provides a real-time controllable multi-harmonic sine wave generator using the 
SO_PlaybackMultiHarmonic class with a Gradio web interface for parameter control.
The frequency is controlled by the height of OptiTrack rigid bodies.
The harmonics are controlled by the orientation (quaternion) of the rigid bodies.
"""

import numpy as np
import time
import threading
import gradio as gr
from spatial_audio_ai.tools.spatializer import (
    SO_PlaybackMultiHarmonic, 
    Spatializer, 
    Scene,
    CHUNKSIZE,
    SAMPLING_RATE
)
from spatial_audio_ai.tools.client import SoundNetworkStreamer
import lunar_tools as lt
from optitrack_python.rigid_body import RigidBody
from optitrack_python.motive_receiver import MotiveReceiver


class HarmonicMotionController:
    """Controller class for managing real-time multi-harmonic sine wave generation 
    with motion and orientation control for multiple rigid bodies."""
    
    def __init__(self):
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.2  # Lower volume since harmonics can be loud
        
        # Create four multi-harmonic sine wave objects for rigid bodies A, B, C, D
        self.harmonic_objects = {}
        self.rigid_bodies = {}
        self.active_objects = {}  # Track which objects are active
        
        rigid_body_names = ['A', 'B', 'C', 'D']
        base_frequencies = [62.5, 125.0, 250.0, 500.0]  # Different base frequencies for each
        
        for i, name in enumerate(rigid_body_names):
            harmonic_obj = SO_PlaybackMultiHarmonic(
                fundamental_frequency=base_frequencies[i],
                amplitude=0.2,  # Lower amplitude since we have harmonics and 4 objects
                num_harmonics=6,  # 6 harmonics for rich sound
                harmonic_decay=0.6  # Decay factor for harmonics
            )
            self.harmonic_objects[name] = harmonic_obj
            self.scene.register(harmonic_obj)
            self.active_objects[name] = True  # Start with all active
        
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        self.is_muted = False
        self.unmuted_amplitude = 0.2  # Lower for harmonics
        
        # OptiTrack setup
        self.motive = None
        self.setup_optitrack()
        
        # Frequency mapping parameters (0m = 62.5Hz, 2m = 500Hz) - 3 octaves
        self.min_height = 0.0
        self.max_height = 2.0
        self.min_frequency = 62.5  # Base frequency
        self.max_frequency = 500.0  # 3 octaves higher (62.5 * 2^3)
        
        # Position, orientation and tracking data for each rigid body
        self.current_heights = {name: 0.0 for name in rigid_body_names}
        self.current_positions = {name: np.array([0.0, 0.0]) for name in rigid_body_names}
        self.current_orientations = {name: np.array([0.0, 0.0, 0.0, 1.0]) for name in rigid_body_names}
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
        """Update frequency, position, and harmonics based on rigid body motion and orientation for all active objects."""
        if not self.rigid_bodies or self.motive is None:
            return
            
        try:
            # Get latest data like the motive_receiver example
            latest_data = self.motive.get_last()
            if not latest_data:
                return
            
            # Update all rigid bodies and their corresponding harmonic objects
            for name, rigid_body in self.rigid_bodies.items():
                if not self.active_objects[name]:
                    # Set amplitude to 0 for inactive objects
                    self.harmonic_objects[name].set_amplitude(0.0)
                    continue
                
                try:
                    # Update rigid body and get position and orientation
                    rigid_body.update()
                    position = rigid_body.positions.get_last()
                    orientation = rigid_body.orientations.get_last()
                    
                    if position is not None:
                        # Extract coordinates: X, Y (height), Z
                        x_pos = position[0]  # X coordinate
                        self.current_heights[name] = position[1]  # Y coordinate (height)
                        z_pos = position[2]  # Z coordinate
                        
                        # Update position (X and Z coordinates map to sound object X and Y)
                        # Apply scaling factor to the position
                        self.current_positions[name] = np.array([x_pos, z_pos])
                        scaled_position = self.current_positions[name] * self.position_scale
                        self.harmonic_objects[name].set_position(scaled_position)
                        
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
                        self.harmonic_objects[name].set_frequency(frequency)
                    
                    if orientation is not None:
                        # Update orientation and thus harmonics
                        self.current_orientations[name] = orientation
                        self.harmonic_objects[name].set_orientation(orientation)
                
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
                
                # Update frequency, position, and harmonics from motion data
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
            # Mute the harmonic object when deactivated
            self.harmonic_objects[name].set_amplitude(0.0)
        elif not self.is_muted:
            # Restore amplitude when activated (if not globally muted)
            self.harmonic_objects[name].set_amplitude(self.unmuted_amplitude)
        
        status = "Active" if is_active else "Inactive"
        return f"Rigid Body {name}: {status}"
    
    def update_amplitude(self, amplitude):
        """Update amplitude for all objects."""
        self.unmuted_amplitude = amplitude  # Always store the unmuted value
        if not self.is_muted:
            # Apply to all active objects
            for name, is_active in self.active_objects.items():
                if is_active:
                    self.harmonic_objects[name].set_amplitude(amplitude)
        return f"Amplitude: {amplitude:.2f}"
    
    def update_phase(self, phase):
        """Update phase offset for all objects."""
        for harmonic_obj in self.harmonic_objects.values():
            harmonic_obj.set_phase_offset(phase)
        return f"Phase: {phase:.2f} rad"
    
    def update_position_scale(self, scale):
        """Update position scaling factor."""
        self.position_scale = scale
        # Re-apply current positions with new scaling for all active objects
        for name, is_active in self.active_objects.items():
            if is_active and name in self.current_positions:
                scaled_position = self.current_positions[name] * self.position_scale
                self.harmonic_objects[name].set_position(scaled_position)
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
        for harmonic_obj in self.harmonic_objects.values():
            harmonic_obj.set_smoothing_factor(smoothing)
        return f"Smoothing: {smoothing:.2f}"
    
    def update_harmonic_decay(self, decay):
        """Update harmonic decay factor for all objects."""
        for harmonic_obj in self.harmonic_objects.values():
            harmonic_obj.set_harmonic_decay(decay)
        return f"Harmonic Decay: {decay:.2f}"
    
    def update_orientation_mapping(self, mode):
        """Update orientation mapping mode for all objects."""
        for harmonic_obj in self.harmonic_objects.values():
            harmonic_obj.set_orientation_mapping_mode(mode)
        return f"Orientation Mapping: {mode}"
    
    def update_frequency_filter(self, enabled, filter_type, cutoff_low, cutoff_high, rolloff):
        """Update frequency filtering for all objects."""
        for harmonic_obj in self.harmonic_objects.values():
            harmonic_obj.set_frequency_filter(
                enabled=enabled,
                filter_type=filter_type,
                cutoff_low=cutoff_low,
                cutoff_high=cutoff_high,
                rolloff=rolloff
            )
        status = "Enabled" if enabled else "Disabled"
        return f"Filter: {status} ({filter_type}, {cutoff_low}-{cutoff_high}Hz, {rolloff}dB/oct)"
    
    def set_custom_harmonic_mask(self, mask_values):
        """Set custom harmonic mask for all objects."""
        # Parse mask values (comma-separated string to array)
        try:
            mask = np.array([float(x.strip()) for x in mask_values.split(',')])
            for harmonic_obj in self.harmonic_objects.values():
                if len(mask) == harmonic_obj.num_harmonics:
                    harmonic_obj.set_custom_harmonic_mask(mask)
            return f"Custom Mask: [{', '.join([f'{x:.2f}' for x in mask])}]"
        except:
            return "Error: Invalid mask format (use comma-separated numbers)"
    
    def toggle_mute(self, is_muted):
        """Toggle mute on/off for all objects."""
        self.is_muted = is_muted
        if is_muted:
            # Mute all objects
            for harmonic_obj in self.harmonic_objects.values():
                harmonic_obj.set_amplitude(0.0)
            return "🔇 Muted"
        else:
            # Unmute only active objects
            for name, is_active in self.active_objects.items():
                if is_active:
                    self.harmonic_objects[name].set_amplitude(self.unmuted_amplitude)
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
            if is_active and name in self.harmonic_objects:
                pos = self.harmonic_objects[name].get_position()
                current_freq = self.harmonic_objects[name].current_frequency
                current_amp = self.harmonic_objects[name].current_amplitude
                rb_pos = self.current_positions[name]
                height = self.current_heights[name]
                orientation = self.current_orientations[name]
                harmonic_amps = self.harmonic_objects[name].get_harmonic_amplitudes()
                
                # Format harmonic amplitudes for display
                harmonic_str = ", ".join([f"{amp:.2f}" for amp in harmonic_amps[:4]])  # Show first 4
                
                body_details.append(
                    f"  {name}: {'Active' if is_active else 'Inactive'} | "
                    f"Pos: ({rb_pos[0]:.2f}, {height:.2f}, {rb_pos[1]:.2f}) | "
                    f"Freq: {current_freq:.1f} Hz | Amp: {current_amp:.2f} | "
                    f"Orient: ({orientation[0]:.2f}, {orientation[1]:.2f}, {orientation[2]:.2f}, {orientation[3]:.2f}) | "
                    f"Harmonics: [{harmonic_str}...]"
                )
            else:
                body_details.append(f"  {name}: Inactive")
        
        # Get harmonic settings from first object (they should all be the same)
        if self.harmonic_objects:
            first_obj = list(self.harmonic_objects.values())[0]
            smoothing = first_obj.smoothing_factor
            harmonic_decay = first_obj.harmonic_decay
            num_harmonics = first_obj.num_harmonics
            orientation_mode = first_obj.orientation_mapping_mode
            filter_info = first_obj.get_frequency_filter_info()
        else:
            smoothing = harmonic_decay = num_harmonics = 0
            orientation_mode = "None"
            filter_info = {"enabled": False, "type": "none"}
        
        status = f"""Status: {'Running' if self.is_running else 'Stopped'}
Mute: {mute_status}
OptiTrack: {tracking_status}

Active Rigid Bodies: {active_summary}
Position Scale Factor: {self.position_scale:.2f}
Frequency Range: {freq_range}

Global Settings:
  Amplitude: {self.unmuted_amplitude:.2f} (stored)
  Smoothing: {smoothing:.2f}
  Harmonics: {num_harmonics}
  Harmonic Decay: {harmonic_decay:.2f}
  Orientation Mapping: {orientation_mode}
  
Frequency Filter:
  Enabled: {filter_info['enabled']}
  Type: {filter_info['type']}
  Cutoffs: {filter_info['cutoff_low']:.0f} - {filter_info['cutoff_high']:.0f} Hz
  Rolloff: {filter_info['rolloff']} dB/octave

Rigid Body Details:
{chr(10).join(body_details)}"""
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    controller = HarmonicMotionController()
    
    with gr.Blocks(title="Real-time Multi-Harmonic Motion Control") as interface:
        gr.Markdown("# Real-time Multi-Harmonic Sine Wave Motion Control")
        gr.Markdown(
            "Control up to 4 real-time generated multi-harmonic sine waves with spatial positioning. "
            "**Frequency is controlled by OptiTrack rigid body height** "
            "(adjustable frequency range with exponential mapping). "
            "**Position is controlled by X and Z coordinates**. "
            "**Harmonics are controlled by orientation (quaternion)** of each rigid body."
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
                    minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                    label="Amplitude"
                )
                
                phase_slider = gr.Slider(
                    minimum=0.0, maximum=2*np.pi, value=0.0, step=0.1,
                    label="Phase Offset (rad)"
                )
                
                gr.Markdown("### Harmonic Controls")
                gr.Markdown(
                    "**Harmonics:** Orientation Controlled (Rigid Body Quaternion)"
                )
                
                harmonic_decay_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                    label="Harmonic Decay Factor"
                )
                
                orientation_mapping_dropdown = gr.Dropdown(
                    choices=[
                        "quaternion_simple", "quaternion_complex", 
                        "basis_fourier", "basis_chebyshev", "basis_legendre", 
                        "basis_wavelets", "basis_radial"
                    ],
                    value="basis_fourier",
                    label="Orientation Mapping Mode"
                )
                
                gr.Markdown("### Frequency Filtering")
                gr.Markdown(
                    "**Filter harmonics** to remove unwanted frequencies (e.g., remove higher harmonics)"
                )
                
                filter_enabled_checkbox = gr.Checkbox(
                    label="Enable Frequency Filter", value=True
                )
                
                filter_type_dropdown = gr.Dropdown(
                    choices=["lowpass", "highpass", "bandpass", "notch", "custom"],
                    value="lowpass",
                    label="Filter Type"
                )
                
                filter_cutoff_low_slider = gr.Slider(
                    minimum=50.0, maximum=10000.0, value=1000.0, step=50.0,
                    label="Low Cutoff Frequency (Hz)"
                )
                
                filter_cutoff_high_slider = gr.Slider(
                    minimum=100.0, maximum=20000.0, value=4000.0, step=100.0,
                    label="High Cutoff Frequency (Hz)"
                )
                
                filter_rolloff_dropdown = gr.Dropdown(
                    choices=[6, 12, 18, 24],
                    value=12,
                    label="Filter Rolloff (dB/octave)"
                )
                
                custom_mask_textbox = gr.Textbox(
                    label="Custom Harmonic Mask (comma-separated, 0.0=mute, 1.0=full)",
                    value="1.0, 1.0, 0.5, 0.3, 0.1, 0.05",
                    placeholder="1.0, 0.8, 0.6, 0.4, 0.2, 0.1"
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
                    label="Min Frequency Status", value="Min Frequency: 62.5 Hz"
                )
                max_freq_output = gr.Textbox(
                    label="Max Frequency Status", value="Max Frequency: 500.0 Hz"
                )
                amp_output = gr.Textbox(
                    label="Amplitude Status", value="Amplitude: 0.20"
                )
                phase_output = gr.Textbox(
                    label="Phase Status", value="Phase: 0.00 rad"
                )
                harmonic_decay_output = gr.Textbox(
                    label="Harmonic Decay Status", value="Harmonic Decay: 0.60"
                )
                orientation_mapping_output = gr.Textbox(
                    label="Orientation Mapping Status", value="Orientation Mapping: basis_fourier"
                )
                filter_output = gr.Textbox(
                    label="Filter Status", value="Filter: Enabled (lowpass, 1000-4000Hz, 12dB/oct)"
                )
                custom_mask_output = gr.Textbox(
                    label="Custom Mask Status", value="Custom Mask: [1.00, 1.00, 0.50, 0.30, 0.10, 0.05]"
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
        
        harmonic_decay_slider.change(
            controller.update_harmonic_decay,
            inputs=harmonic_decay_slider,
            outputs=harmonic_decay_output
        )
        
        orientation_mapping_dropdown.change(
            controller.update_orientation_mapping,
            inputs=orientation_mapping_dropdown,
            outputs=orientation_mapping_output
        )
        
        # Frequency filter event handlers
        def update_filter_wrapper(*args):
            return controller.update_frequency_filter(*args)
        
        filter_enabled_checkbox.change(
            update_filter_wrapper,
            inputs=[filter_enabled_checkbox, filter_type_dropdown, 
                   filter_cutoff_low_slider, filter_cutoff_high_slider, 
                   filter_rolloff_dropdown],
            outputs=filter_output
        )
        
        filter_type_dropdown.change(
            update_filter_wrapper,
            inputs=[filter_enabled_checkbox, filter_type_dropdown, 
                   filter_cutoff_low_slider, filter_cutoff_high_slider, 
                   filter_rolloff_dropdown],
            outputs=filter_output
        )
        
        filter_cutoff_low_slider.change(
            update_filter_wrapper,
            inputs=[filter_enabled_checkbox, filter_type_dropdown, 
                   filter_cutoff_low_slider, filter_cutoff_high_slider, 
                   filter_rolloff_dropdown],
            outputs=filter_output
        )
        
        filter_cutoff_high_slider.change(
            update_filter_wrapper,
            inputs=[filter_enabled_checkbox, filter_type_dropdown, 
                   filter_cutoff_low_slider, filter_cutoff_high_slider, 
                   filter_rolloff_dropdown],
            outputs=filter_output
        )
        
        filter_rolloff_dropdown.change(
            update_filter_wrapper,
            inputs=[filter_enabled_checkbox, filter_type_dropdown, 
                   filter_cutoff_low_slider, filter_cutoff_high_slider, 
                   filter_rolloff_dropdown],
            outputs=filter_output
        )
        
        custom_mask_textbox.change(
            controller.set_custom_harmonic_mask,
            inputs=custom_mask_textbox,
            outputs=custom_mask_output
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
    print("Starting Real-time Multi-Harmonic Motion Control Interface")
    print(f"Audio settings: {SAMPLING_RATE} Hz, {CHUNKSIZE} samples per chunk")
    print("Motion control by OptiTrack Rigid Bodies A, B, C, D:")
    print("  Y-axis (height): 0m = 62.5 Hz, 2m = 500 Hz (3 octaves, exponential)")
    print("  X-axis and Z-axis: Control spatial position of each sound object")
    print("  Orientation (quaternion): Controls harmonic amplitudes")
    print("  Base frequencies: A=62.5Hz, B=125Hz, C=250Hz, D=500Hz")
    print("  6 harmonics per object with orientation-controlled amplitudes")
    print("Open your web browser to control activation and parameters")
    
    interface = create_interface()
    interface.launch(
        server_name=lt.get_local_ip(),  # Listen on specific IP address
        server_port=7861,  # Different port from original
        share=False,  # Set to True if you want a public link
        show_api=False
    ) 