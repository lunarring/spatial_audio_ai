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

        # Validate mapping scheme
        if self.mapping_scheme not in ['stereo', 'alternating']:
            raise ValueError("Invalid mapping_scheme. Choose 'stereo' or 'alternating'.")

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
        else:
            raise ValueError("Invalid mapping_scheme.")

        mapped[:, 12] = (left + right) / 2

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


if __name__ == "__main__":
    # Optional: Print available devices for verification
    print("Available audio devices:")
    print(sd.query_devices())

    # Example usage:
    mapping_scheme = 'alternating'
    relayer = BlackHoleStereoRelayer(mapping_scheme=mapping_scheme)
    relayer.start()

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