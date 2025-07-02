import logging
from enum import IntEnum
import time
from collections import deque
import numpy as np
import sounddevice as sd
import argparse
from spatial_audio_ai.tools.tools import generate_random_noise
from spatial_audio_ai.config import (
    SAMPLING_RATE, BLOCKSIZE, N_SPEAKERS, MAX_QUEUE_SIZE
)

sd.default.blocksize = BLOCKSIZE

# Configuration
# Speaker id to soundcard cluster stereo_channel_idx mapping  
SPEAKER_SOUNDCARD_CLUSTER_MAPPING = {
    f"speaker{i}": {"soundcard_cluster_channel": i} 
    for i in range(1, N_SPEAKERS + 1)
}
class StereoChannels(IntEnum):
    "Stereo channel mapping to virtual soundcard stereo channel id"
    LEFT = 0
    RIGHT = 1


class StreamManager():
    def __init__(self, device: int, samplerate: int, stereo_channel_idx: int) -> None:
        self.device = device
        self.samplerate = samplerate
        self.stereo_channel_idx = stereo_channel_idx
        self.queue = deque(maxlen=MAX_QUEUE_SIZE)  # Limit queue size
        self.index = 0
        self.queue_index = 0
        self.dropped_frames = 0

    def callback(self, outdata : np.array, frames : int, time : float, status : sd.CallbackFlags) -> None:
        if status:
            print(f"Status: {status}")
        
        if len(self.queue) > 0:
            audio_to_play = np.zeros((len(audio := self.queue.popleft()), 2), dtype=np.float32)
            audio_to_play[:, 1 - self.stereo_channel_idx] = audio
            remaining_samples_2 = len(audio) - self.index
            outdata[:] = audio_to_play
        else:
            audio_to_play = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
            outdata[:] = audio_to_play

    def start(self) -> None:
        self.stream = sd.OutputStream(device=self.device, samplerate=self.samplerate, channels=2, callback=self.callback)
        self.stream.start()


class MockStreamManager():
    """A mock version of StreamManager that doesn't use actual audio devices"""
    def __init__(self, samplerate : int = SAMPLING_RATE) -> None:
        self.samplerate = samplerate
        self.queue = deque()
    
    def start(self) -> None:
        pass  # No actual stream to start


class SoundSystem():

    def __init__(self, log_level=logging.INFO, mock_mode=False) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=log_level)
        self.mock_mode = mock_mode
        
        if not mock_mode:
            try:
                self.config = self._make_config()
                self.streams = self._start_streams()
            except Exception as e:
                self.logger.warning(f"Failed to initialize real sound system: {e}")
                self.logger.warning("Falling back to mock mode")
                self.mock_mode = True
        
        if self.mock_mode:
            self.streams = self._start_mock_streams()

    def add_to_playback_queue(self, data : np.array) -> None:
        assert len(data.shape) == 2, f"data.shape must have 2 elements (data.shape is {data.shape})"
        assert data.shape[1] % BLOCKSIZE == 0, f"Data length must be divisible by BLOCKSIZE ({BLOCKSIZE})"
        "Play an audio file with distinct audi data per channel."
        
        if self.mock_mode:
            self.logger.info("Mock mode: audio would be played")
            return
            
        for speaker_id, speaker_audio in enumerate(data):
            speaker = list(self.streams.keys())[speaker_id]
            # Chunk into BLOCKSIZE
            for i in range(0, len(speaker_audio), BLOCKSIZE):
                chunk = speaker_audio[i:i + BLOCKSIZE]
                self.streams[speaker].queue.append(chunk)

        self.logger.info("Playing")

    def get_current_buffer_time(self) -> int:
        "Return the length of the queue of the first stream."
        if self.mock_mode:
            return 0
            
        first_stream_key = next(iter(self.streams))
        nmb_blocks = len(self.streams[first_stream_key].queue)
        nmb_samples = nmb_blocks * BLOCKSIZE
        remaining_time = nmb_samples / SAMPLING_RATE
        return remaining_time

    def _start_mock_streams(self) -> dict:
        """Initialize mock streams for testing without hardware"""
        streams = {}
        for speaker in SPEAKER_SOUNDCARD_CLUSTER_MAPPING.keys():
            streams[speaker] = MockStreamManager()
            streams[speaker].start()
        return streams

    def _start_streams(self) -> None:
        "Initalize and start streams for each speaker in the config."
        streams = {}
        for speaker, scfg in self.config.items():
            streams[speaker] = StreamManager(scfg["vistual_sound_card_id"], samplerate=SAMPLING_RATE, stereo_channel_idx=int(scfg["stereo_channel"]))
            streams[speaker].start()
        return streams        

    def _make_config(self) -> dict:
        "Generate the config for all speakers."
        cfg = {}
        for speaker in SPEAKER_SOUNDCARD_CLUSTER_MAPPING.keys():
            soundcard_cluster_channel = SPEAKER_SOUNDCARD_CLUSTER_MAPPING[speaker]["soundcard_cluster_channel"]
            virtual_soundcard_suffix = self.__derive_virtual_soundcard_suffix(soundcard_cluster_channel)
            vistual_sound_card_id = self.__derive_virtual_soundcard_id(virtual_soundcard_suffix)
            stereo_channel = self.__derive_stereo_channel(soundcard_cluster_channel)
            cfg = {
                **cfg,
                **{speaker: {
                        "soundcard_cluster_channel": soundcard_cluster_channel,
                        "virtual_soundcard_suffix": virtual_soundcard_suffix,
                        "vistual_sound_card_id": vistual_sound_card_id,
                        "stereo_channel": stereo_channel
                    }
                }
            }
        return cfg

    @staticmethod
    def __derive_virtual_soundcard_suffix(soundcard_cluster_channel: int) -> str:
        "Maps, e.g. 1 or 2 to '1-2', and so on."
        even = soundcard_cluster_channel - (soundcard_cluster_channel + 1) % 2
        odd = even + 1
        return f"{even}-{odd}"

    @staticmethod
    def __derive_virtual_soundcard_id(virtual_soundcard_suffix):
        for device in sd.query_devices():
            if " (Dante Virtu" in device["name"] and "DVS Transmit" in device["name"] and "(Dante Virtual Soundcard)" not in device["name"]:
                if virtual_soundcard_suffix == SoundSystem.__extract_device_channels(device["name"]):
                    return device["index"]
        raise Exception(f"No valid Dante Virtual soundcard found with suffix {virtual_soundcard_suffix}")

    @staticmethod
    def __derive_stereo_channel(soundcard_cluster_channel) -> StereoChannels:
        "Maps, e.g. even to left and odd to right."
        return StereoChannels(soundcard_cluster_channel % 2)

    @staticmethod
    def __extract_device_channels(name : str) -> str:
        "Extracts the sound card cluster channels (e.g. 1-2) part from a Dante Virtual sound card name."
        name = name.replace("  ", " ")
        name = name.split(" ")
        name = name[2]
        return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play audio through specified speaker with given amplitude')
    parser.add_argument('--speaker', type=int, default=1, help=f'Speaker number (1-{N_SPEAKERS}, default: 1)')
    parser.add_argument('--amplitude', type=float, default=0.1, help='Amplitude of the audio (default: 0.1)')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration in seconds (default: 1.0)')
    parser.add_argument('--mock', action='store_true', help='Use mock mode (no hardware required)')
    args = parser.parse_args()

    # Validate speaker number
    if not 1 <= args.speaker <= N_SPEAKERS:
        raise ValueError(f"Speaker number must be between 1 and {N_SPEAKERS}")

    sound_sys = SoundSystem(logging.INFO, mock_mode=args.mock)
    
    # Generate random audio data
    sound_file, actual_duration = generate_random_noise(
        duration=args.duration,
        sampling_rate=SAMPLING_RATE,
        blocksize=BLOCKSIZE,
        n_speakers=N_SPEAKERS,
        speaker_id=args.speaker,
        amplitude=args.amplitude
    )
    
    print(f'Playing random noise through speaker {args.speaker} with duration {actual_duration:.2f} seconds')
    sound_sys.add_to_playback_queue(sound_file)
    time.sleep(1.1 * actual_duration)


