import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
import soundfile as sf
import time
from spatial_audio_ai.tools.numpysocket import NumpySocket
import os
import random
from collections import deque
import threading
import logging
import gradio as gr

BLOCKSIZE = 1024
CHUNKSIZE = BLOCKSIZE * 4
SAMPLING_RATE = 44100

sd.default.blocksize = BLOCKSIZE

# Dummy simulated socket class for simulation mode
class SimulatedSocket:
    def __init__(self):
        self.connected = False
        self.last_data = None

    def connect(self, addr):
        self.connected = True
        self.addr = addr
        print(f"Simulated socket connected to {addr}")

    def sendall(self, data):
        if not self.connected:
            raise Exception("Simulated socket not connected")
        self.last_data = data
        print("Simulated socket sent data")

    def recv(self):
        if not self.connected:
            raise Exception("Simulated socket not connected")
        print("Simulated socket returning echoed data")
        return self.last_data

    def close(self):
        self.connected = False
        print("Simulated socket closed.")


class SoundNetworkStreamer:
    def __init__(self, host: str = "10.40.49.47", port: int = 9999, simulate: bool = False):
        self.host = host
        self.port = port
        self.simulate = simulate
        if self.simulate:
            self.socket = SimulatedSocket()
        else:
            self.socket = NumpySocket()
        self.socket_connected = False
        self.lock = threading.Lock()  # To ensure thread safety if needed
        self.__enter__()

    def __enter__(self):
        self.socket.connect((self.host, self.port))
        self.socket_connected = True
        if self.simulate:
            print(f"[Simulation] Connected to simulated server at {self.host}:{self.port}")
        else:
            print(f"Connected to server at {self.host}:{self.port}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.socket_connected:
            self.socket.close()
            self.socket_connected = False
            print("Connection closed.")

    def send(self, data: np.ndarray):
        if data.ndim == 2:
            if data.shape[1] % BLOCKSIZE == 0:
                try:
                    self.socket.sendall(data)
                    # print(f"Sent data with shape {data.shape} to the server.")
                except Exception as e:
                    print(f"Failed to send data: {e}")
            else:
                print(f"Data shape[1] must be divisible by blocksize. Current shape[1]: {data.shape[1]}, blocksize: {BLOCKSIZE}")
        else:
            print("Data must be a 2-dimensional numpy array")

    def receive(self) -> np.ndarray:
        try:
            response = self.socket.recv()
            if response is not None:
                print(f"Received data from server: {response}")
            else:
                print("No data received. The connection might be closed.")
            return response
        except Exception as e:
            print(f"Failed to receive data: {e}")
            return None

    def send_and_receive(self, data: np.ndarray) -> np.ndarray:
        """
        Sends data to the server and waits to receive a response.
        """
        with self.lock:
            self.send(data)
            print("waiting to receive...")
            return self.receive()


class BlackHoleStereoRelayer:
    """
    A class to record system audio from BlackHole and relay it using SoundNetworkStreamer with advanced channel mapping.
    """

    def __init__(self,
                 sample_rate=44100,
                 channels=2,
                 chunk_size=1024 * 10,
                 device_name="BlackHole 64ch",
                 max_queue_size=1000,
                 stream_volume=0.7,
                 mapping_scheme='alternating'):
        """
        Initializes the BlackHoleStereoRelayer.

        Parameters:
        - sample_rate (int): Sampling rate in Hz.
        - channels (int): Number of audio channels (stereo).
        - chunk_size (int): Number of samples per audio chunk.
        - device_name (str): Name of the BlackHole device.
        - max_queue_size (int): Maximum number of chunks to store in the deque.
        - stream_volume (float): Volume scaling factor for the audio stream.
        - mapping_scheme (str): Playback mapping scheme ('stereo' or 'alternating').
        """
        # Configure logging
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_name = device_name
        self.max_queue_size = max_queue_size
        self.stream_volume = stream_volume
        self.mapping_scheme = mapping_scheme.lower()
        
        # Individual channel volumes (13 channels)
        self.channel_volumes = [1.0] * 13
        
        # Solo states for each channel (13 channels)
        self.channel_solo = [False] * 13

        # Validate mapping scheme
        if self.mapping_scheme not in ['stereo', 'alternating', 'mono']:
            raise ValueError("Invalid mapping_scheme. Choose 'stereo', 'alternating', or 'mono'.")

        # Initialize the deque to store audio chunks
        self.audio_deque = deque(maxlen=self.max_queue_size)

        # Initialize the SoundNetworkStreamer (using real connection)
        self.sound_streamer = SoundNetworkStreamer()

        # Thread control
        self._recording_thread = None
        self._stop_event = threading.Event()

        # Get device index dynamically
        self.device_index = self.get_device_index_by_name(self.device_name)

    def get_device_index_by_name(self, device_name):
        """
        Retrieves the device index by its name.

        Parameters:
        - device_name (str): The name of the audio device.

        Returns:
        - int: The index of the device.

        Raises:
        - ValueError: If the device is not found.
        """
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device_name.lower() in device['name'].lower():
                logging.info(f"Selected device '{device['name']}' with index {idx}.")
                return idx
        raise ValueError(f"Device '{device_name}' not found.")

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function called by sounddevice for each audio block.

        Parameters:
        - indata (numpy.ndarray): Incoming audio data.
        - frames (int): Number of frames.
        - time_info (dict): Dictionary containing timing information.
        - status (sounddevice.CallbackFlags): Callback status.
        """
        if status:
            logging.warning(f"Status: {status}")
        # Append a copy of the audio chunk to the deque
        latest_chunk = self._map_channels(indata.copy())
        self.audio_deque.append(latest_chunk)

    def _record_audio(self):
        """
        Internal method to start the audio recording stream.
        Runs in a separate thread.
        """
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                device=self.device_index,
                                channels=self.channels,
                                blocksize=self.chunk_size,
                                callback=self.audio_callback):
                logging.info("Recording started. Press Ctrl+C to stop.")
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            logging.error(f"An error occurred in the recording thread: {e}")

    def _map_channels(self, indata):
        """
        Maps the input stereo data to 13 output channels based on the selected scheme
        and adds a 13th channel as the sum of left and right.

        Parameters:
        - indata (numpy.ndarray): Input audio data with shape (chunk_size, 2).

        Returns:
        - numpy.ndarray: Mapped audio data with shape (chunk_size, 13).
        """
        if indata.shape[1] != 2:
            raise ValueError("Input data must have exactly 2 channels (stereo).")

        left = indata[:, 0]  # Left channel
        right = indata[:, 1]  # Right channel

        mapped = np.zeros((indata.shape[0], 13), dtype=np.float32)
        if self.mapping_scheme == 'stereo':
            # Stereo (grouped) mapping: first 6 channels are left, next 6 channels are right
            for i in range(6):
                mapped[:, i] = left
            for i in range(6, 12):
                mapped[:, i] = right
        elif self.mapping_scheme == 'alternating':
            # Alternating mapping: even channels left, odd channels right
            for i in range(12):
                if i % 2 == 0:
                    mapped[:, i] = left
                else:
                    mapped[:, i] = right
        elif self.mapping_scheme == 'mono':
            # Mono mapping: averaged left+right signal to all channels
            mono_signal = (left + right) / 2
            for i in range(12):
                mapped[:, i] = mono_signal
        else:
            raise ValueError("Invalid mapping_scheme.")

        # 13th channel is always the sum of left and right (or mono signal for mono mode)
        mapped[:, 12] = (left + right) / 2

        # Apply individual channel volumes
        for i in range(13):
            mapped[:, i] *= self.channel_volumes[i]

        # Apply solo logic - if any channel is soloed, mute all non-soloed channels
        if any(self.channel_solo):
            for i in range(13):
                if not self.channel_solo[i]:
                    mapped[:, i] = 0.0

        return mapped

    def handle_key_press(self, key: str):
        """
        Handles key press events. When the 't' key is pressed, toggles the playback mode 
        between 'alternating' and 'stereo' modes.
        
        Parameters:
        - key (str): The key that was pressed.
        """
        if key.lower() == 't':
            previous_mode = self.mapping_scheme
            if self.mapping_scheme == 'alternating':
                self.mapping_scheme = 'stereo'
            else:
                self.mapping_scheme = 'alternating'
            print(f"Playback mode toggled from {previous_mode} to {self.mapping_scheme}")

    def start(self):
        """
        Starts the recording thread and begins processing audio chunks.
        """
        self._recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._recording_thread.start()

        try:
            while not self._stop_event.is_set():
                if self.audio_deque:
                    latest_chunk = self.audio_deque.popleft()

                    if not isinstance(latest_chunk, np.ndarray):
                        latest_chunk = np.array(latest_chunk)

                    processed_chunk = latest_chunk.T * self.stream_volume

                    self.sound_streamer.send(processed_chunk)

                    logging.info(f"Processed and sent a new audio chunk of shape {processed_chunk.shape}.")
                else:
                    logging.debug("No audio data available yet.")
                time.sleep(0.01)
        except KeyboardInterrupt:
            logging.info("\nRecording stopped by user.")
            self.stop()
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            self.stop()

    def stop(self):
        """
        Stops the recording and processing.
        """
        self._stop_event.set()
        if self._recording_thread is not None:
            self._recording_thread.join()
        logging.info("Recording and processing have been stopped.")

    def update_master_volume(self, volume):
        """Update master volume."""
        self.stream_volume = volume
        
    def update_channel_volume(self, channel_idx, volume):
        """Update individual channel volume."""
        if 0 <= channel_idx < 13:
            self.channel_volumes[channel_idx] = volume
            
    def update_channel_solo(self, channel_idx, solo_state):
        """Update individual channel solo state."""
        if 0 <= channel_idx < 13:
            self.channel_solo[channel_idx] = solo_state
            
    def update_mapping_scheme(self, scheme):
        """Update mapping scheme."""
        if scheme.lower() in ['stereo', 'alternating', 'mono']:
            self.mapping_scheme = scheme.lower()


def create_gradio_interface(relayer):
    """Create Gradio interface for controlling the audio relayer."""
    
    def update_master_vol(vol):
        relayer.update_master_volume(vol)
        return f"Master volume: {vol:.2f}"
    
    def update_mapping(scheme):
        relayer.update_mapping_scheme(scheme)
        return f"Mapping scheme: {scheme}"
    
    def update_ch_vol(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12):
        volumes = [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12]
        for i, vol in enumerate(volumes):
            relayer.update_channel_volume(i, vol)
        return f"Channel volumes updated"
    
    def update_ch_solo(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12):
        solo_states = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
        
        # Count how many channels are currently soloed
        current_solo_count = sum(relayer.channel_solo)
        new_solo_count = sum(solo_states)
        
        if new_solo_count == 0:
            # All checkboxes unchecked - turn off all solos
            for i in range(13):
                relayer.update_channel_solo(i, False)
            return ("No channels soloed", *[False] * 13)
        elif new_solo_count == 1:
            # Exactly one checkbox is checked - find it and solo only that one
            newly_soloed = solo_states.index(True)
            
            # Reset all solo states
            for i in range(13):
                relayer.update_channel_solo(i, False)
            # Set only the newly soloed channel
            relayer.update_channel_solo(newly_soloed, True)
            
            # Return updated checkbox states for UI
            updated_states = [False] * 13
            updated_states[newly_soloed] = True
            return (f"Solo active on channel: {newly_soloed + 1}", *updated_states)
        else:
            # Multiple checkboxes are checked - find the most recently clicked one
            # This happens when user clicks a second checkbox while first is still checked
            if current_solo_count == 1:
                # Find which channel was previously soloed
                prev_soloed = relayer.channel_solo.index(True)
                # Find the new one (the one that's not the previous one)
                for i, solo in enumerate(solo_states):
                    if solo and i != prev_soloed:
                        newly_soloed = i
                        break
                else:
                    # Fallback: use the first True one
                    newly_soloed = solo_states.index(True)
                
                # Reset all solo states
                for i in range(13):
                    relayer.update_channel_solo(i, False)
                # Set only the newly soloed channel
                relayer.update_channel_solo(newly_soloed, True)
                
                # Return updated checkbox states for UI
                updated_states = [False] * 13
                updated_states[newly_soloed] = True
                return (f"Solo active on channel: {newly_soloed + 1}", *updated_states)
            else:
                # Fallback: solo the first checked channel
                newly_soloed = solo_states.index(True)
                
                # Reset all solo states
                for i in range(13):
                    relayer.update_channel_solo(i, False)
                # Set only the newly soloed channel
                relayer.update_channel_solo(newly_soloed, True)
                
                # Return updated checkbox states for UI
                updated_states = [False] * 13
                updated_states[newly_soloed] = True
                return (f"Solo active on channel: {newly_soloed + 1}", *updated_states)
    
    with gr.Blocks(title="Spatial Audio Control") as interface:
        gr.Markdown("# Spatial Audio Control Interface")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Master Controls")
                master_volume = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.01,
                    label="Master Volume"
                )
                mapping_scheme = gr.Radio(
                    choices=["alternating", "stereo", "mono"], 
                    value="alternating",
                    label="Mapping Scheme"
                )
                
                master_status = gr.Textbox(label="Master Status", interactive=False)
                mapping_status = gr.Textbox(label="Mapping Status", interactive=False)
        
        gr.Markdown("## Individual Channel Volumes")
        
        # Initialize lists for all sliders and checkboxes (in correct order 1-13)
        ch_sliders = [None] * 13
        ch_solo_checkboxes = [None] * 13
            
        # Row 1: Channels 1-5
        with gr.Row():
            for i in range(5):
                with gr.Column(min_width=150):
                    slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                        label=f"Ch {i+1}"
                    )
                    ch_sliders[i] = slider  # Store in correct index
                    
                    solo_checkbox = gr.Checkbox(
                        value=False,
                        label="Solo"
                    )
                    ch_solo_checkboxes[i] = solo_checkbox  # Store in correct index
        
        # Row 2: Channel 12 (left) and Channel 6 (right)
        with gr.Row():
            # Ch 12 on the left
            with gr.Column(min_width=150):
                slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                    label="Ch 12"
                )
                ch_sliders[11] = slider  # Channel 12 goes to index 11
                
                solo_checkbox = gr.Checkbox(
                    value=False,
                    label="Solo"
                )
                ch_solo_checkboxes[11] = solo_checkbox
            
            # Empty space in middle
            with gr.Column(min_width=150):
                gr.HTML("")
            with gr.Column(min_width=150):
                gr.HTML("")
            with gr.Column(min_width=150):
                gr.HTML("")
            
            # Ch 6 on the right
            with gr.Column(min_width=150):
                slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                    label="Ch 6"
                )
                ch_sliders[5] = slider  # Channel 6 goes to index 5
                
                solo_checkbox = gr.Checkbox(
                    value=False,
                    label="Solo"
                )
                ch_solo_checkboxes[5] = solo_checkbox
        
        # Row 3: Channels 11, 10, 9, 8, 7 (in that order)
        with gr.Row():
            for visual_pos, ch_num in enumerate([11, 10, 9, 8, 7]):
                with gr.Column(min_width=150):
                    slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                        label=f"Ch {ch_num}"
                    )
                    ch_sliders[ch_num - 1] = slider  # Store in correct index (ch_num - 1)
                    
                    solo_checkbox = gr.Checkbox(
                        value=False,
                        label="Solo"
                    )
                    ch_solo_checkboxes[ch_num - 1] = solo_checkbox
        
        # Row 4: Channel 13 alone
        with gr.Row():
            # Empty space for centering
            with gr.Column(min_width=150):
                gr.HTML("")
            with gr.Column(min_width=150):
                gr.HTML("")
            
            with gr.Column(min_width=150):
                slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                    label="Ch 13"
                )
                ch_sliders[12] = slider  # Channel 13 goes to index 12
                
                solo_checkbox = gr.Checkbox(
                    value=False,
                    label="Solo"
                )
                ch_solo_checkboxes[12] = solo_checkbox
            
            # Empty space for centering
            with gr.Column(min_width=150):
                gr.HTML("")
            with gr.Column(min_width=150):
                gr.HTML("")
        
        ch_status = gr.Textbox(label="Channel Status", interactive=False)
        solo_status = gr.Textbox(label="Solo Status", interactive=False)
        
        # Event handlers
        master_volume.change(
            fn=update_master_vol,
            inputs=[master_volume],
            outputs=[master_status]
        )
        
        mapping_scheme.change(
            fn=update_mapping,
            inputs=[mapping_scheme],
            outputs=[mapping_status]
        )
        
        for slider in ch_sliders:
            slider.change(
                fn=update_ch_vol,
                inputs=ch_sliders,
                outputs=[ch_status]
            )
        
        for checkbox in ch_solo_checkboxes:
            checkbox.change(
                fn=update_ch_solo,
                inputs=ch_solo_checkboxes,
                outputs=[solo_status] + ch_solo_checkboxes
            )
    
    return interface


if __name__ == "__main__":
    # Optional: Print available devices for verification
    print("Available audio devices:")
    print(sd.query_devices())

    # Example usage:
    mapping_scheme = 'alternating'
    relayer = BlackHoleStereoRelayer(mapping_scheme=mapping_scheme)
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(relayer)
    
    # Start relayer in a separate thread
    relayer_thread = threading.Thread(target=relayer.start, daemon=True)
    relayer_thread.start()
    
    # Launch Gradio interface
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__x":
    sound_streamer = SoundNetworkStreamer()
    nmb_blocks = 200
    num_channels = 13
    noise_array = np.random.uniform(low=-1.0, high=1.0, size=(int(BLOCKSIZE * nmb_blocks), num_channels))
    noise_array = np.clip(noise_array, -1, 1)
    noise_array *= 0.3
    x = sound_streamer.send_and_receive(noise_array.T)
    print(x)
    # sound_streamer.close()