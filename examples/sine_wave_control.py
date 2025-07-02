#!/usr/bin/env python3
"""
Real-time Sine Wave Control with Gradio Interface

This script provides a real-time controllable sine wave generator using the 
SO_PlaybackSine class with a Gradio web interface for parameter control.
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


class SineWaveController:
    """Controller class for managing real-time sine wave generation."""
    
    def __init__(self):
        self.sine_object = SO_PlaybackSine()
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = 0.3
        self.scene.register(self.sine_object)
        
        self.sound_streamer = None
        self.is_running = False
        self.audio_thread = None
        
    def start_audio(self):
        """Start the audio generation loop."""
        if self.is_running:
            return "Audio already running"
            
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
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
            
            for chunk in self.scene.run():
                if not self.is_running:
                    break
                    
                # Clip audio to prevent overflow
                chunk = np.clip(chunk, -1, 1)
                self.sound_streamer.send(chunk)
                
                # Sleep to maintain proper timing
                time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
                
        except Exception as e:
            print(f"Audio loop error: {e}")
        finally:
            self.is_running = False
            self.sound_streamer = None
    
    def update_frequency(self, frequency):
        """Update sine wave frequency."""
        self.sine_object.set_frequency(frequency)
        return f"Frequency: {frequency:.1f} Hz"
    
    def update_amplitude(self, amplitude):
        """Update sine wave amplitude."""
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
    
    def get_status(self):
        """Get current status information."""
        pos = self.sine_object.get_position()
        status = f"""
        Status: {'Running' if self.is_running else 'Stopped'}
        Frequency: {self.sine_object.frequency:.1f} Hz
        Amplitude: {self.sine_object.amplitude:.2f}
        Phase: {self.sine_object.phase_offset:.2f} rad
        Position: ({pos[0]:.1f}, {pos[1]:.1f})
        """
        return status


def create_interface():
    """Create and configure the Gradio interface."""
    
    controller = SineWaveController()
    
    with gr.Blocks(title="Real-time Sine Wave Control") as interface:
        gr.Markdown("# Real-time Sine Wave Spatial Audio Control")
        gr.Markdown(
            "Control a real-time generated sine wave with spatial positioning"
        )
        
        with gr.Row():
            with gr.Column():
                # Control buttons
                start_btn = gr.Button("Start Audio", variant="primary")
                stop_btn = gr.Button("Stop Audio", variant="secondary")
                
                # Parameter controls
                gr.Markdown("### Audio Parameters")
                
                freq_slider = gr.Slider(
                    minimum=50, maximum=2000, value=440, step=1,
                    label="Frequency (Hz)"
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
                
            with gr.Column():
                # Status and feedback
                status_output = gr.Textbox(
                    label="Status", 
                    value=controller.get_status(),
                    lines=8
                )
                
                # Parameter feedback
                freq_output = gr.Textbox(
                    label="Frequency Status", value="Frequency: 440.0 Hz"
                )
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
        freq_slider.change(
            controller.update_frequency,
            inputs=freq_slider,
            outputs=freq_output
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
        
        # Manual status refresh button
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            controller.get_status,
            outputs=status_output
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Real-time Sine Wave Control Interface")
    print(f"Audio settings: {SAMPLING_RATE} Hz, {CHUNKSIZE} samples per chunk")
    print("Open your web browser to control the sine wave parameters")
    
    interface = create_interface()
    interface.launch(
        server_name="10.40.49.109",  # Listen on specific IP address
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_api=False
    ) 