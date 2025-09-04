import logging
from enum import IntEnum
import time as time_module
from collections import deque
from typing import Dict, Tuple
import asyncio
import threading
import numpy as np
import sounddevice as sd
import argparse
from spatial_audio_ai.tools.tools import generate_random_noise
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE, N_SPEAKERS, MAX_AUDIO_QUEUE_DEPTH, AUDIO_LATENCY_MODE, UDP_BUFFER_DEPTH

sd.default.blocksize = BLOCKSIZE

## Configuration

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
    def __init__(self, device : int, samplerate : int, stereo_channel_idx : int, verbose : bool = False) -> None:
        self.device = device
        self.samplerate = samplerate
        self.stereo_channel_idx = stereo_channel_idx
        self.verbose = verbose
        self.queue = deque()
        self.index = 0
        self.queue_index = 0
        # Add minimum buffer enforcement for stable streaming
        self.min_buffer_blocks = 8  # Minimum buffer before starting playback (configurable)
        self.playback_started = False  # Track if we've started playing
        
        # Adaptive buffer management for ultra-low latency profiles
        self.target_buffer_blocks = 2  # Target buffer level for stable playback
        self.adaptive_mode = False  # Enable adaptive timing adjustments

    def set_min_buffer_blocks(self, min_blocks: int):
        """Set minimum buffer blocks required before starting playback"""
        self.min_buffer_blocks = min_blocks
        # For ultra-low latency profiles, enable adaptive mode
        if min_blocks <= 3:  # ultra_low_latency and experimental profiles
            self.adaptive_mode = True
            self.target_buffer_blocks = min_blocks + 1  # Target 1 block above minimum
        else:
            self.adaptive_mode = False
            
        if self.verbose:
            mode_str = " (adaptive mode)" if self.adaptive_mode else ""
            print(f"[STREAM] Minimum buffer set to {min_blocks} blocks (~{min_blocks * BLOCKSIZE / SAMPLING_RATE * 1000:.1f}ms){mode_str}")

    def get_buffer_health(self) -> str:
        """Get current buffer health status for adaptive management"""
        current_queue_len = len(self.queue)
        if not self.adaptive_mode:
            return "stable"
        
        if current_queue_len == 0:
            return "critical"  # Buffer underrun
        elif current_queue_len == 1:
            return "low"       # Close to underrun
        elif current_queue_len <= self.target_buffer_blocks:
            return "good"      # At target level
        else:
            return "high"      # Above target (risk of overflow)

    def callback(self, outdata : np.array, frames : int, time : float, status : sd.CallbackFlags) -> None:
        if status:
            print(f"Status: {status}")
        
        # Check if we have enough buffer to start/continue playback
        current_queue_len = len(self.queue)
        
        # If we haven't started playback yet, wait for minimum buffer
        if not self.playback_started:
            if current_queue_len >= self.min_buffer_blocks:
                self.playback_started = True
                if self.verbose:
                    print(f"[STREAM] Playback started with {current_queue_len} blocks buffer")
            else:
                # Not enough buffer yet - play silence and wait
                audio_to_play = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
                outdata[:] = audio_to_play
                if self.verbose and current_queue_len > 0:
                    print(f"[STREAM] Buffering: {current_queue_len}/{self.min_buffer_blocks} blocks")
                return
        
        # Check for buffer underrun during playback
        if current_queue_len == 0:
            # Buffer underrun - stop playback and require rebuilding buffer
            self.playback_started = False
            audio_to_play = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
            outdata[:] = audio_to_play
            if not getattr(self, '_underflow_logged', False):
                print(f"[SERVER][UNDERFLOW] Buffer underrun - stopping playback, will restart when {self.min_buffer_blocks} blocks available")
                self._underflow_logged = True
            return
        
        # Normal playback - we have buffer
        if current_queue_len > 0:
            # Reset underflow flag when data is available
            self._underflow_logged = False
            audio_to_play = np.zeros((len(audio := self.queue.popleft()), 2), dtype=np.float32)
            audio_to_play[:, 1 - self.stereo_channel_idx] = audio
            remaining_samples_2 = len(audio) - self.index
            outdata[:] = audio_to_play
            
            # Only log when queue length changes significantly (to avoid spam)
            if not hasattr(self, '_last_logged_queue_len'):
                self._last_logged_queue_len = -1
                self._callback_count = 0
            
            self._callback_count += 1
            
            # Log occasionally with buffer health info for adaptive mode
            queue_change = abs(current_queue_len - self._last_logged_queue_len)
            if self.verbose and (queue_change >= 2 or self._callback_count % 500 == 0):
                health = self.get_buffer_health()
                print(f"[AUDIO CB] Queue: {current_queue_len} blocks | Health: {health}")
                self._last_logged_queue_len = current_queue_len

    def start(self) -> None:
        # Configure stream based on latency mode
        stream_params = {
            'device': self.device,
            'samplerate': self.samplerate,
            'channels': 2,
            'callback': self.callback,
            'blocksize': BLOCKSIZE
        }
        
        if AUDIO_LATENCY_MODE == 'ultra':
            # Ultra-low latency: most aggressive settings
            stream_params.update({
                'latency': 'low',
                'clip_off': True,
                'dither_off': True,
                'never_drop_input': False,
                'prime_output_buffers_using_stream_callback': True
            })
        elif AUDIO_LATENCY_MODE == 'low':
            # Low latency: balanced settings
            stream_params.update({
                'latency': 'low',
                'clip_off': True,
                'dither_off': False  # Keep dithering for quality
            })
        # 'stable' mode uses default settings
        
        self.stream = sd.OutputStream(**stream_params)
        self.stream.start()


class MockStreamManager():
    """A mock version of StreamManager that doesn't use actual audio devices"""
    def __init__(self, samplerate : int = SAMPLING_RATE) -> None:
        self.samplerate = samplerate
        self.queue = deque()
    
    def start(self) -> None:
        pass  # No actual stream to start


class SoundSystem():

    def __init__(self, log_level=logging.INFO, mock_mode=False, verbose=False) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=log_level)
        self.mock_mode = mock_mode
        self.verbose = verbose
        # Dynamic max queue depth for UDP profiles
        self.max_queue_depth = UDP_BUFFER_DEPTH
        # Mono restream queue (BLOCKSIZE-sized mono chunks)
        self.return_queue = deque()
        
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

    def add_to_playback_queue(self, data : np.array, seq : int = None) -> None:
        assert len(data.shape) == 2, f"data.shape must have 2 elements (data.shape is {data.shape})"
        assert data.shape[1] % BLOCKSIZE == 0, f"Data length must be divisible by BLOCKSIZE ({BLOCKSIZE})"
        "Play an audio file with distinct audi data per channel."
        
        if self.mock_mode:
            self.logger.info("Mock mode: audio would be played")
            return
            
        t0 = time_module.perf_counter()
        
        # Check queue depth and implement adaptive buffering
        first_stream_key = next(iter(self.streams))
        current_queue_len = len(self.streams[first_stream_key].queue)
        MAX_QUEUE_DEPTH = self.max_queue_depth  # Dynamic based on client profile
        
        # Get adaptive mode status from streams
        adaptive_mode = getattr(self.streams[first_stream_key], 'adaptive_mode', False)
        
        # Smart queue management for ultra-low latency profiles
        if adaptive_mode and current_queue_len >= MAX_QUEUE_DEPTH:
            # For ultra-low latency profiles, be more intelligent about drops
            target_buffer = getattr(self.streams[first_stream_key], 'target_buffer_blocks', 2)
            
            if current_queue_len > target_buffer + 1:
                # Drop oldest chunk instead of newest to maintain responsiveness
                for stream in self.streams.values():
                    if len(stream.queue) > 0:
                        stream.queue.popleft()  # Remove oldest chunk
                # Keep the mono restream queue aligned
                if len(self.return_queue) > 0:
                    self.return_queue.popleft()
                print(f"[SERVER][ADAPTIVE-DROP] dropped oldest chunk, queue reduced to {current_queue_len-1} (seq={seq if seq is not None else '?'})")
            else:
                # Still at max, drop this new chunk
                print(f"[SERVER][DROP][QUEUE-OVERFLOW] dropping seq={seq if seq is not None else '?'} (queue depth={current_queue_len}/{MAX_QUEUE_DEPTH})")
                return
        elif current_queue_len >= MAX_QUEUE_DEPTH:
            # Standard drop for non-adaptive profiles
            print(f"[SERVER][DROP][QUEUE-OVERFLOW] dropping seq={seq if seq is not None else '?'} (queue depth={current_queue_len}/{MAX_QUEUE_DEPTH})")
            return
        
        # Enqueue mono restream (mean over all channels), chunked to BLOCKSIZE
        try:
            mono_series = np.mean(data, axis=0).astype(np.float32)
            for i in range(0, len(mono_series), BLOCKSIZE):
                mono_chunk = mono_series[i:i + BLOCKSIZE]
                self.return_queue.append(mono_chunk)
        except Exception as e:
            print(f"[SERVER][RESTREAM] Failed to enqueue mono restream: {e}")
        
        for speaker_id, speaker_audio in enumerate(data):
            speaker = list(self.streams.keys())[speaker_id]
            # Chunk into BLOCKSIZE
            for i in range(0, len(speaker_audio), BLOCKSIZE):
                chunk = speaker_audio[i:i + BLOCKSIZE]
                self.streams[speaker].queue.append(chunk)
        
        self.logger.info("Playing")
        t1 = time_module.perf_counter()
        buf_secs = self.get_current_buffer_time()
        
        # Only log occasionally to avoid spam
        if not hasattr(self, '_enqueue_counter'):
            self._enqueue_counter = 0
            self._last_logged_buffer = -1
        
        self._enqueue_counter += 1
        
        # Log only when buffer time changes significantly OR every 500 enqueues (≈10 seconds)
        buffer_change = abs(buf_secs - self._last_logged_buffer)
        if buffer_change > 0.1 or self._enqueue_counter % 500 == 0:  # 100ms buffer change or every 10s
            # Get buffer status info
            buffer_ms = buf_secs * 1000
            min_buffer_ms = getattr(self, '_min_buffer_ms', 42.7)  # Default 8 blocks at 48kHz (~42.7ms)
            
            # Check if any stream is currently playing
            playback_active = False
            if hasattr(self, 'streams') and self.streams:
                first_stream = next(iter(self.streams.values()))
                playback_active = getattr(first_stream, 'playback_started', False)
            
            status = "PLAYING" if playback_active else "BUFFERING"
            buffer_health = "HEALTHY" if buffer_ms >= min_buffer_ms else "LOW"
            
            # Add adaptive mode info to logging
            mode_info = " [ADAPTIVE]" if adaptive_mode else ""
            print(f"[SERVER][BUFFER] {status} | {buffer_ms:.1f}ms queued (min: {min_buffer_ms:.1f}ms) | Health: {buffer_health}{mode_info} | Enqueue: {(t1-t0)*1000:.1f}ms")
            self._last_logged_buffer = buf_secs

    def get_current_buffer_time(self) -> int:
        "Return the length of the queue of the first stream."
        if self.mock_mode:
            return 0
            
        first_stream_key = next(iter(self.streams))
        nmb_blocks = len(self.streams[first_stream_key].queue)
        nmb_samples = nmb_blocks * BLOCKSIZE
        remaining_time = nmb_samples / SAMPLING_RATE
        return remaining_time

    def set_max_queue_depth(self, depth: int) -> None:
        """Set a new maximum queue depth for adaptive buffering."""
        self.max_queue_depth = depth

    def set_min_buffer_blocks(self, min_blocks: int) -> None:
        """Set minimum buffer blocks for all streams to prevent underflows on unstable networks."""
        if self.mock_mode:
            return
            
        # Store minimum buffer in milliseconds for logging
        self._min_buffer_ms = (min_blocks * BLOCKSIZE / SAMPLING_RATE) * 1000
            
        if hasattr(self, 'streams') and self.streams:
            for stream_name, stream in self.streams.items():
                if hasattr(stream, 'set_min_buffer_blocks'):
                    stream.set_min_buffer_blocks(min_blocks)
            if self.verbose:
                print(f"[SOUND_SYSTEM] Set minimum buffer to {min_blocks} blocks ({self._min_buffer_ms:.1f}ms) for all {len(self.streams)} streams")

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
            streams[speaker] = StreamManager(scfg["vistual_sound_card_id"], samplerate=SAMPLING_RATE, stereo_channel_idx=int(scfg["stereo_channel"]), verbose=self.verbose)
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
        # Device selection based on latency mode
        if AUDIO_LATENCY_MODE in ['ultra', 'low']:
            # First try to find low-latency WASAPI drivers (3-10ms latency)
            for device in sd.query_devices():
                if ("DVS Transmit" in device["name"] and 
                    "(Dante Virtual Soundcard)" in device["name"] and  # Use WASAPI drivers (low latency)
                    device["default_low_output_latency"] < 0.02):  # Less than 20ms latency
                    if virtual_soundcard_suffix == SoundSystem.__extract_device_channels(device["name"]):
                        print(f"[{AUDIO_LATENCY_MODE.upper()} LATENCY] Using device {device['index']}: {device['name']} "
                              f"(latency: {device['default_low_output_latency']*1000:.1f}ms)")
                        return device["index"]
        
        # Fallback to any Dante device (for 'stable' mode or if low-latency not found)
        for device in sd.query_devices():
            if " (Dante Virtu" in device["name"] and "DVS Transmit" in device["name"]:
                if virtual_soundcard_suffix == SoundSystem.__extract_device_channels(device["name"]):
                    mode_label = "STABLE" if AUDIO_LATENCY_MODE == 'stable' else "FALLBACK"
                    print(f"[{mode_label}] Using device {device['index']}: {device['name']} "
                          f"(latency: {device['default_low_output_latency']*1000:.1f}ms)")
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
    time_module.sleep(1.1 * actual_duration)


