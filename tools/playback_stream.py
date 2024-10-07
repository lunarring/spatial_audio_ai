import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import soundfile as sf
import time
from numpysocket import NumpySocket
import os
import random
import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time
import logging


BLOCKSIZE = 1024
CHUNKSIZE = BLOCKSIZE*4
SAMPLING_RATE = 44100

sd.default.blocksize = BLOCKSIZE
        
class SoundNetworkStreamer:
    def __init__(self, host: str = "10.40.49.21", port: int = 9999):
        self.host = host
        self.port = port
        self.socket = NumpySocket()
        self.__enter__()

    def __enter__(self):
        self.socket.connect((self.host, self.port))
        return self

    def close(self):
        self.socket.close()

    def send(self, data: np.ndarray):
        if data.ndim == 2:
            if data.shape[1] % BLOCKSIZE == 0:
                # print(data.shape)
                self.socket.sendall(data)
            else:
                print(f"Data shape[1] must be divisible by blocksize. Current shape[1]: {data.shape[1]}, blocksize: {BLOCKSIZE}")
        else:
            print("Data must be a 2-dimensional numpy array")


class BlackHoleStereoRelayer:
    """
    A class to record system audio from BlackHole and relay it using SoundNetworkStreamer with advanced channel mapping.
    """

    def __init__(self,
                 sample_rate=44100,
                 channels=2,
                 chunk_size=1024*10,
                 device_name="BlackHole 64ch",
                 max_queue_size=1000,
                 stream_volume=0.1,
                 mapping_scheme='grouped'):
        """
        Initializes the BlackHoleStereoRelayer.

        Parameters:
        - sample_rate (int): Sampling rate in Hz.
        - channels (int): Number of audio channels (stereo).
        - chunk_size (int): Number of samples per audio chunk.
        - device_name (str): Name of the BlackHole device.
        - max_queue_size (int): Maximum number of chunks to store in the deque.
        - stream_volume (float): Volume scaling factor for the audio stream.
        - mapping_scheme (str): Channel mapping scheme ('grouped' or 'alternating').
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
        if self.mapping_scheme not in ['grouped', 'alternating']:
            raise ValueError("Invalid mapping_scheme. Choose 'grouped' or 'alternating'.")

        # Initialize the deque to store audio chunks
        self.audio_deque = deque(maxlen=self.max_queue_size)

        # Initialize the SoundNetworkStreamer
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
                    time.sleep(0.1)  # Short sleep to reduce CPU usage
        except Exception as e:
            logging.error(f"An error occurred in the recording thread: {e}")

    def _map_channels(self, indata):
        """
        Maps the input stereo data to 12 output channels based on the selected scheme
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

        if self.mapping_scheme == 'grouped':
            # Grouped Scheme: Channels 1-6 map to left, 7-12 map to right
            mapped = np.zeros((indata.shape[0], 13), dtype=np.float32)
            # Map left channel to channels 1-6
            for i in range(6):
                mapped[:, i] = left
            # Map right channel to channels 7-12
            for i in range(6, 12):
                mapped[:, i] = right
        elif self.mapping_scheme == 'alternating':
            # Alternating Scheme: Alternate between left and right for channels 1-12
            mapped = np.zeros((indata.shape[0], 13), dtype=np.float32)
            for i in range(12):
                if i % 2 == 0:
                    mapped[:, i] = left
                else:
                    mapped[:, i] = right
        else:
            # This block should never be reached due to validation in __init__
            raise ValueError("Invalid mapping_scheme.")

        # Add the 13th channel as the sum of left and right
        mapped[:, 12] = (left + right) / 2

        return mapped

    def start(self):
        """
        Starts the recording thread and begins processing audio chunks.
        """
        # Start the recording thread
        self._recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._recording_thread.start()

        try:
            while not self._stop_event.is_set():
                if self.audio_deque:
                    # Retrieve the latest audio chunk
                    latest_chunk = self.audio_deque.popleft()

                    # Ensure it's a NumPy array
                    if not isinstance(latest_chunk, np.ndarray):
                        latest_chunk = np.array(latest_chunk)

                    # # Map the channels to 13 channels
                    # try:
                    #     mapped_chunk = self._map_channels(latest_chunk)
                    # except ValueError as ve:
                    #     logging.error(f"Channel mapping error: {ve}")
                    #     continue

                    # Transpose and scale the audio chunk
                    processed_chunk = latest_chunk.T * self.stream_volume

                    # Send the processed chunk to the SoundNetworkStreamer
                    self.sound_streamer.send(processed_chunk)

                    logging.info(f"Processed and sent a new audio chunk of shape {processed_chunk.shape}.")
                else:
                    logging.debug("No audio data available yet.")
                # Sleep briefly to prevent a tight loop
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
    # Choose the mapping scheme: 'grouped' or 'alternating'
    mapping_scheme = 'alternating'  # Change to 'alternating' as needed

    # Initialize the relayer with desired parameters
    relayer = BlackHoleStereoRelayer(mapping_scheme=mapping_scheme)

    # Start the relayer
    relayer.start()
