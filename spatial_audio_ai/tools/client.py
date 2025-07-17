import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
import soundfile as sf
import time
from spatial_audio_ai.tools.numpysocket import FastNumpySocket
import os
import random
from collections import deque
import threading
import logging
import socket  # Add socket import for UDP support
import gradio as gr
import json
import uuid
from spatial_audio_ai.tools.tools import generate_random_noise
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE

try:
    import lunar_tools as lt
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("Warning: lunar_tools not available. ZMQ support disabled.")

CHUNKSIZE = BLOCKSIZE * 4

# Control message magic for profile selection
CONTROL_MAGIC = b'NPCC'
# Allowed queue-depth profiles (clearer names)
ALLOWED_PROFILES = {'ultra_low_latency', 'low_latency', 'balanced', 'high_buffer', 'super_buffer', 'stable'}

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
    def __init__(self, host: str = "10.40.49.47", port: int = 9999, zmq_port: int = 5556, simulate: bool = False, profile: str = None):
        self.host = host
        self.port = port
        self.zmq_port = zmq_port
        self.simulate = simulate
        self.profile = profile
        
        # Validate profile selection
        if self.profile is not None and self.profile not in ALLOWED_PROFILES:
            raise ValueError(f"Invalid profile '{self.profile}'. Allowed profiles: {sorted(ALLOWED_PROFILES)}")
        
        # Auto-select protocol based on profile
        self.use_zmq = self.profile == 'stable' and ZMQ_AVAILABLE and not simulate
        
        if self.use_zmq:
            # Use ZMQ for stable profiles
            self.streamer = SoundNetworkStreamerZMQ(host=host, zmq_port=zmq_port, profile=profile or 'stable')
        else:
            # Use UDP for low-latency profiles or fallback
            if self.simulate:
                self.socket = SimulatedSocket()
            else:
                self.socket = FastNumpySocket(type=socket.SOCK_DGRAM)  # Use UDP for streaming
            self.socket_connected = False
            self.lock = threading.Lock()  # To ensure thread safety if needed
            self.__enter__()

    def __enter__(self):
        if self.use_zmq:
            # ZMQ streamer handles its own connection
            return self
        else:
            # UDP connection logic
            self.socket.connect((self.host, self.port))
            self.socket_connected = True
            if self.simulate:
                print(f"[Simulation] Connected to simulated server at {self.host}:{self.port}")
            else:
                print(f"[UDP] Connected to server at {self.host}:{self.port}")
            # Send profile control on connect if provided
            if self.profile:
                try:
                    ctrl = CONTROL_MAGIC + self.profile.encode()
                    self.socket.send(ctrl)
                    print(f"[UDP][PROFILE] sent profile={self.profile}")
                except Exception as e:
                    print(f"[UDP][PROFILE] failed to send profile: {e}")
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.use_zmq:
            self.streamer.close()
        else:
            if self.socket_connected:
                self.socket.close()
                self.socket_connected = False
                print("[UDP] Connection closed.")

    def send(self, data: np.ndarray):
        if self.use_zmq:
            self.streamer.send(data)
        else:
            if data.ndim == 2:
                if data.shape[1] % BLOCKSIZE == 0:
                    try:
                        self.socket.sendall(data)
                        # print(f"Sent data with shape {data.shape} to the server.")
                    except Exception as e:
                        print(f"[UDP] Failed to send data: {e}")
                else:
                    print(f"[UDP] Data shape[1] must be divisible by blocksize. Current shape[1]: {data.shape[1]}, blocksize: {BLOCKSIZE}")
            else:
                print("[UDP] Data must be a 2-dimensional numpy array")

    def receive(self) -> np.ndarray:
        if self.use_zmq:
            return self.streamer.receive()
        else:
            try:
                response = self.socket.recv()
                if response is not None:
                    print(f"[UDP] Received data from server: {response}")
                else:
                    print("[UDP] No data received. The connection might be closed.")
                return response
            except Exception as e:
                print(f"[UDP] Failed to receive data: {e}")
                return None

    def send_and_receive(self, data: np.ndarray) -> np.ndarray:
        """
        Sends data to the server and waits to receive a response.
        """
        if self.use_zmq:
            self.send(data)
            print("[ZMQ] waiting to receive...")
            return self.receive()
        else:
            with self.lock:
                self.send(data)
                print("[UDP] waiting to receive...")
                return self.receive()


class SoundNetworkStreamerZMQ:
    """ZMQ-based audio streamer for stable connections with high latency tolerance"""
    
    def __init__(self, host: str = "10.40.49.47", zmq_port: int = 5556, profile: str = "stable"):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("lunar_tools not available. Cannot use ZMQ streamer.")
        
        self.host = host
        self.zmq_port = zmq_port
        self.profile = profile
        self.client_id = str(uuid.uuid4())[:8]  # Short unique ID
        self._seq = 0
        self.zmq_client = None
        self.connected = False
        
        # Validate profile
        if self.profile not in ALLOWED_PROFILES:
            raise ValueError(f"Invalid profile '{self.profile}'. Allowed profiles: {sorted(ALLOWED_PROFILES)}")
        
        self.__enter__()
    
    def __enter__(self):
        try:
            print(f"[ZMQ][CLIENT] Attempting to create ZMQ client endpoint...")
            print(f"[ZMQ][CLIENT] Host: {self.host}, Port: {self.zmq_port}")
            print(f"[ZMQ][CLIENT] lunar_tools available: {ZMQ_AVAILABLE}")
            
            self.zmq_client = lt.ZMQPairEndpoint(is_server=False, ip=self.host, port=str(self.zmq_port))
            self.connected = True
            print(f"[ZMQ][CLIENT] ✅ Connected to server at {self.host}:{self.zmq_port}")
            print(f"[ZMQ][CLIENT] ZMQ client object: {self.zmq_client}")
            
            # Send profile control message
            control_msg = {
                "client_id": self.client_id,
                "control": {
                    "profile": self.profile
                }
            }
            print(f"[ZMQ][CLIENT] Sending control message: {control_msg}")
            self.zmq_client.send_json(control_msg)
            print(f"[ZMQ][CLIENT] ✅ Profile message sent: profile={self.profile} client_id={self.client_id}")
            
            # Small delay to ensure message is sent
            time.sleep(0.1)
            
        except Exception as e:
            print(f"[ZMQ][CLIENT] ❌ Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            self.connected = False
            raise
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def close(self):
        if self.connected and self.zmq_client:
            self.zmq_client = None
            self.connected = False
            print(f"[ZMQ] Connection closed for client {self.client_id}")
    
    def send(self, data: np.ndarray):
        """Send numpy array via ZMQ using JSON format"""
        if not self.connected:
            print("[ZMQ] Not connected - cannot send data")
            return
        
        if data.ndim != 2:
            print("[ZMQ] Data must be a 2-dimensional numpy array")
            return
        
        if data.shape[1] % BLOCKSIZE != 0:
            print(f"[ZMQ] Data shape[1] must be divisible by blocksize. Current shape[1]: {data.shape[1]}, blocksize: {BLOCKSIZE}")
            return
        
        try:
            # Prepare audio data message
            audio_msg = {
                "client_id": self.client_id,
                "audio_data": {
                    "seq": self._seq,
                    "timestamp": time.perf_counter(),
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "data": data.tolist()  # Convert numpy array to list for JSON
                }
            }
            
            # Send via ZMQ
            self.zmq_client.send_json(audio_msg)
            self._seq += 1
            
            # Log occasionally for debugging
            if self._seq % 50 == 0:
                print(f"[ZMQ] Sent audio seq={self._seq} shape={data.shape}")
                
        except Exception as e:
            print(f"[ZMQ] Failed to send data: {e}")
    
    def receive(self) -> np.ndarray:
        """Receive data from server (if server sends responses)"""
        if not self.connected:
            print("[ZMQ] Not connected - cannot receive data")
            return None
        
        try:
            messages = self.zmq_client.get_messages()
            if messages:
                # Return the latest message's audio data if available
                for msg in messages:
                    if 'audio_data' in msg:
                        audio_info = msg['audio_data']
                        array_data = np.array(audio_info['data'], dtype=audio_info['dtype'])
                        return array_data.reshape(audio_info['shape'])
            return None
        except Exception as e:
            print(f"[ZMQ] Failed to receive data: {e}")
            return None


class BlackHoleStereoRelayer:
    """
    A class to record system audio from BlackHole and relay it using SoundNetworkStreamer with advanced channel mapping.
    """

    def __init__(self,
                 sample_rate=None,
                 channels=2,
                 chunk_size=None,
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
        self.logger = logging.getLogger(__name__)

        if sample_rate is None:
            sample_rate = SAMPLING_RATE
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Calculate optimal chunk size based on sample rate if not provided
        if chunk_size is None:
            # Use CHUNKSIZE to match what the system expects (21.3ms chunks)
            chunk_size = CHUNKSIZE
        self.chunk_size = chunk_size
        self.device_name = device_name
        self.max_queue_size = max_queue_size
        self.stream_volume = stream_volume
        self.mapping_scheme = mapping_scheme.lower()
        
        # Individual channel volumes (13 channels)
        self.channel_volumes = [1.0] * 13
        
        # Solo states for each channel (13 channels)
        self.channel_solo = [False] * 13
        
        # Previous volume states for smoothing (prevents clicks)
        self._prev_effective_volumes = [1.0] * 13

        # Validate mapping scheme
        if self.mapping_scheme not in ['stereo', 'alternating', 'mono']:
            raise ValueError("Invalid mapping_scheme. Choose 'stereo', 'alternating', or 'mono'.")

        # Initialize the deque to store audio chunks
        self.audio_deque = deque(maxlen=self.max_queue_size)

        # Initialize the SoundNetworkStreamer (using real connection)
        self.sound_streamer = SoundNetworkStreamer(profile='stable')

        # Thread control
        self._recording_thread = None
        self._stop_event = threading.Event()

        # Get device index dynamically
        self.device_index = self.get_device_index_by_name(self.device_name)
        
        # Log configuration for debugging
        self.logger.info(f"BlackHole Relayer configured:")
        self.logger.info(f"  Sample Rate: {self.sample_rate} Hz")
        self.logger.info(f"  Chunk Size: {self.chunk_size} samples (~{self.chunk_size/self.sample_rate*1000:.1f}ms)")
        self.logger.info(f"  BLOCKSIZE: {BLOCKSIZE} samples (~{BLOCKSIZE/self.sample_rate*1000:.1f}ms)")
        self.logger.info(f"  Device: {self.device_name}")
        self.logger.info(f"  Fixed: Using CHUNKSIZE-aligned chunks for smooth streaming")

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

        # Apply individual channel volumes and solo logic with smoothing
        for i in range(13):
            # Calculate target effective volume
            if any(self.channel_solo):
                # Solo mode: only soloed channels get volume
                target_volume = self.channel_volumes[i] if self.channel_solo[i] else 0.0
            else:
                # Normal mode: all channels get their individual volume
                target_volume = self.channel_volumes[i]
            
            # Smooth volume transitions to prevent clicks (simple linear interpolation)
            smoothing_factor = 0.1  # Adjust for smoother/faster transitions
            effective_volume = (1 - smoothing_factor) * self._prev_effective_volumes[i] + smoothing_factor * target_volume
            self._prev_effective_volumes[i] = effective_volume
            
            mapped[:, i] *= effective_volume

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

        # Pre-buffer chunks for smooth playback
        min_buffer_chunks = 8  # Increased for smaller chunks (8 * 21.3ms = ~170ms buffer)
        chunk_duration = self.chunk_size / self.sample_rate
        
        try:
            # Wait for initial buffer to fill
            logging.info(f"Building initial buffer ({min_buffer_chunks} chunks)...")
            while len(self.audio_deque) < min_buffer_chunks and not self._stop_event.is_set():
                time.sleep(0.01)
            
            if self._stop_event.is_set():
                return
                
            logging.info("Initial buffer ready, starting stream...")
            start_time = time.perf_counter()
            chunk_counter = 0

            while not self._stop_event.is_set():
                # Maintain buffer - only send if we have enough chunks ahead
                if len(self.audio_deque) >= min_buffer_chunks:
                    latest_chunk = self.audio_deque.popleft()

                    if not isinstance(latest_chunk, np.ndarray):
                        latest_chunk = np.array(latest_chunk)

                    processed_chunk = latest_chunk.T * self.stream_volume
                    
                    # Ensure no clipping that could cause clicks
                    processed_chunk = np.clip(processed_chunk, -1.0, 1.0)

                    self.sound_streamer.send(processed_chunk)
                    chunk_counter += 1

                    logging.debug(f"Sent chunk {chunk_counter}, buffer size: {len(self.audio_deque)}")
                    
                    # Schedule next chunk send time (similar to playback.py)
                    next_time = start_time + chunk_counter * chunk_duration
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -chunk_duration:
                        # If we're more than one chunk behind, reset timing
                        start_time = time.perf_counter() - chunk_counter * chunk_duration
                        logging.warning("Timing reset due to large delay")
                else:
                    # Buffer underrun - wait for more data
                    logging.debug(f"Buffer underrun, waiting... (buffer size: {len(self.audio_deque)})")
                    time.sleep(0.001)  # Very short sleep when waiting for buffer
                    
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


def main():
    """Main entry point for the bh command"""
    import argparse
    from spatial_audio_ai.tools.tools import generate_random_noise
    
    parser = argparse.ArgumentParser(description='Spatial Audio Client')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Send test audio to spatial audio server')
    test_parser.add_argument('--speaker', type=int, default=1, help='Speaker number (1-13, default: 1)')
    test_parser.add_argument('--amplitude', type=float, default=0.1, help='Amplitude of the audio (default: 0.1)')
    test_parser.add_argument('--duration', type=float, default=1.0, help='Duration in seconds (default: 1.0)')
    test_parser.add_argument('--host', default="10.40.49.47", help='Server host address')
    test_parser.add_argument('--port', type=int, default=9999, help='Server port number')
    
    # BlackHole command
    blackhole_parser = subparsers.add_parser('blackhole', help='Start BlackHole audio relayer with Gradio interface')
    blackhole_parser.add_argument('--mapping', default='alternating', choices=['alternating', 'stereo', 'mono'], help='Mapping scheme (default: alternating)')
    
    args = parser.parse_args()
    
    # Default to blackhole command if no command is specified
    if args.command is None:
        args.command = 'blackhole'
        args.mapping = 'alternating'
    
    if args.command == 'test':
        # Validate speaker number
        if not 1 <= args.speaker <= 13:
            raise ValueError(f"Speaker number must be between 1 and 13")
        
        sound_streamer = SoundNetworkStreamer(host=args.host, port=args.port)
        
        # Generate random audio data for the specified speaker
        sound_file, actual_duration = generate_random_noise(
            duration=args.duration,
            sampling_rate=SAMPLING_RATE,
            blocksize=BLOCKSIZE,
            n_speakers=13,
            speaker_id=args.speaker,
            amplitude=args.amplitude
        )
        
        print(f'Sending random noise through speaker {args.speaker} with duration {actual_duration:.2f} seconds')
        print(f"Sound array shape: {sound_file.shape}")
        print(f"Data stats - Min: {np.min(sound_file)}, Max: {np.max(sound_file)}, Mean: {np.mean(sound_file)}")
        
        # Send the audio data
        sound_streamer.send(sound_file)
        print("Data sent to server")
        
        print(f"Audio chunk sent. Keeping connection alive for {actual_duration:.2f} seconds...")
        time.sleep(actual_duration)
        
        sound_streamer.close()
    
    elif args.command == 'blackhole':
        # Optional: Print available devices for verification
        print("Available audio devices:")
        print(sd.query_devices())

        # Create BlackHole relayer
        relayer = BlackHoleStereoRelayer(mapping_scheme=args.mapping)
        
        # Create and launch Gradio interface
        interface = create_gradio_interface(relayer)
        
        # Start relayer in a separate thread
        relayer_thread = threading.Thread(target=relayer.start, daemon=True)
        relayer_thread.start()
        
        # Launch Gradio interface
        interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
    
    else:
        # No command provided - do nothing
        print("No command specified. Use 'test' or 'blackhole' commands.")
        parser.print_help()


if __name__ == "__main_x_":
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

if __name__ == "__main__":
    main()