import logging
from enum import IntEnum
import time
from collections import deque
from typing import Dict, Tuple
import asyncio
import threading
import numpy as np
import sounddevice as sd

BLOCKSIZE = 1024
sd.default.blocksize = BLOCKSIZE

SAMPLING_RATE = 44100

## Configuration
# The number of speakers, numbered consecutively
N_SPEAKERS = 13

# Speaker id to soundcard cluster stereo_channel_idx mapping
# usually maps 1 to 1, but can be different; defined in the Dante Cntroller
# Note: Speaker ID is once defined names of speaker in the room
# Note: Soundcard cluster channel is the channel no of the virtual 16 
SPEAKER_SOUNDCARD_CLUSTER_MAPPING = {f"speaker{i}": {"soundcard_cluster_channel": i} for i in range(1, N_SPEAKERS + 1)}

## Code
class StereoChannels(IntEnum):
    "Stereo channel mapping to virtual soundcard stereo channel id"
    LEFT = 0
    RIGHT = 1

class StreamManager():
    def __init__(self,device : int, samplerate : int, stereo_channel_idx : int) -> None:
        self.device = device
        self.samplerate = samplerate
        self.stereo_channel_idx = stereo_channel_idx
        self.queue = deque()
        self.index = 0
        self.queue_index = 0

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


class SoundSystem():

    def __init__(self, log_level = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)  # not sure if the correct way
        self.logger.setLevel(level=log_level)
        self.config = self._make_config()
        self.streams = self._start_streams()

    def add_to_playback_queue(self, data : np.array, duration : float = 0.01) -> None:
        assert len(data.shape) == 2, f"data.shape must have 2 elements (data.shape is {data.shape})"
        assert data.shape[1] % BLOCKSIZE == 0, f"Data length must be divisible by BLOCKSIZE ({BLOCKSIZE})"
        "Play an audio file with distinct audi data per channel."
        for speaker_id, speaker_audio in enumerate(data):
            speaker = list(self.streams.keys())[speaker_id]
            # Chunk into BLOCKSIZE
            for i in range(0, len(speaker_audio), BLOCKSIZE):
                chunk = speaker_audio[i:i + BLOCKSIZE]
                self.streams[speaker].queue.append(chunk)
            # self.streams[speaker].queue.append(speaker_audio)

        self.logger.info("Playing")

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
    # # GOOD PLAYBACK
    if False:
        sound_sys = SoundSystem(logging.INFO)
        file_name = "latest.npy"
        amplitude = 0.3
        sound_file = np.load(file_name)
        sound_file *= amplitude
        duration = sound_file.shape[1] / SAMPLING_RATE
        print(f'Playing {file_name} with duration {duration:.2f} seconds')
        sound_sys.play(sound_file, duration)

    # Chunking Test
    if True:
        sound_sys = SoundSystem(logging.INFO)
        file_name = "latest.npy"
        amplitude = 0.2
        sound_file = np.load(file_name)
        sound_file *= amplitude

        sound_sys.add_to_playback_queue(sound_file[:,0:500*BLOCKSIZE], 0)
        time.sleep(1.1*sound_file.shape[1] / SAMPLING_RATE)


