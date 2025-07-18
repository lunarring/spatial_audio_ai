"""
Buffered Audio Streaming for Spatial Audio AI

Provides buffering capabilities for irregular audio sources like real-time generation APIs.
Decouples irregular input timing from consistent spatial audio output timing.
"""

import asyncio
import time
import numpy as np
from collections import deque
from typing import AsyncGenerator, Optional
import logging

from spatial_audio_ai.tools.client import SoundNetworkStreamer
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE


class AudioBuffer:
    """
    Thread-safe audio buffer for decoupling irregular audio input from consistent output.
    
    Maintains a sliding window buffer with configurable maximum duration.
    Handles partial blocks and provides precise sample management.
    """
    
    def __init__(self, max_seconds: float = 2.0, sample_rate: int = SAMPLING_RATE):
        """
        Initialize the audio buffer.
        
        Args:
            max_seconds: Maximum buffer duration in seconds (sliding window)
            sample_rate: Audio sample rate in Hz
        """
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.channels = 2  # Assume stereo
        self.max_samples = int(max_seconds * sample_rate)
        
        # Buffer storage
        self.buffer = deque()  # List of audio chunks
        self.total_samples = 0  # Total samples currently buffered
        self.partial_block = np.array([], dtype=np.float32).reshape(0, self.channels)
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        # Statistics
        self.chunks_added = 0
        self.blocks_extracted = 0
        self.underruns = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Add audio chunk to buffer. Maintains sliding window by removing old audio.
        
        Args:
            audio_chunk: Audio data with shape (samples, channels) as float32
        """
        async with self.lock:
            # Ensure correct format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Ensure 2D array (samples, channels)
            if audio_chunk.ndim == 1:
                audio_chunk = audio_chunk.reshape(-1, 1)
            if audio_chunk.shape[1] == 1:
                # Convert mono to stereo
                audio_chunk = np.hstack([audio_chunk, audio_chunk])
            
            # Add to buffer
            self.buffer.append(audio_chunk)
            self.total_samples += len(audio_chunk)
            self.chunks_added += 1
            
            # Maintain sliding window - remove old chunks if exceeding max_seconds
            while self.total_samples > self.max_samples and len(self.buffer) > 1:
                old_chunk = self.buffer.popleft()
                self.total_samples -= len(old_chunk)
            
            # Log occasionally
            if self.chunks_added % 50 == 0:
                buffer_seconds = self.get_buffer_seconds()
                self.logger.debug(f"[BUFFER] Added chunk {self.chunks_added}, "
                                f"buffer: {buffer_seconds:.2f}s ({self.total_samples} samples)")
    
    async def get_block(self, block_size: int = BLOCKSIZE) -> Optional[np.ndarray]:
        """
        Extract exactly block_size samples from buffer.
        
        Args:
            block_size: Number of samples to extract
            
        Returns:
            Audio block with shape (block_size, channels) or None if insufficient data
        """
        async with self.lock:
            # Check if we have enough samples (including partial block)
            available_samples = len(self.partial_block) + self.total_samples
            if available_samples < block_size:
                self.underruns += 1
                return None
            
            # Accumulate samples until we have enough
            accumulated = self.partial_block.copy()
            
            while len(accumulated) < block_size and len(self.buffer) > 0:
                chunk = self.buffer.popleft()
                self.total_samples -= len(chunk)
                accumulated = np.vstack([accumulated, chunk]) if len(accumulated) > 0 else chunk
            
            # Extract exactly block_size samples
            block = accumulated[:block_size]
            self.partial_block = accumulated[block_size:] if len(accumulated) > block_size else np.array([], dtype=np.float32).reshape(0, self.channels)
            
            self.blocks_extracted += 1
            
            # Log occasionally
            if self.blocks_extracted % 200 == 0:  # Every ~1 second at 48kHz
                buffer_seconds = self.get_buffer_seconds()
                health = "HEALTHY" if buffer_seconds > 0.5 else "LOW" if buffer_seconds > 0.1 else "CRITICAL"
                self.logger.info(f"[BUFFER] Block {self.blocks_extracted}, "
                               f"buffer: {buffer_seconds:.2f}s, health: {health}, "
                               f"underruns: {self.underruns}")
            
            return block
    
    def get_buffer_seconds(self) -> float:
        """Get current buffer level in seconds."""
        total_buffered = len(self.partial_block) + self.total_samples
        return total_buffered / self.sample_rate
    
    def has_minimum_buffer(self, min_seconds: float = 0.5) -> bool:
        """Check if buffer has at least min_seconds of audio."""
        return self.get_buffer_seconds() >= min_seconds
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'buffer_seconds': self.get_buffer_seconds(),
            'chunks_added': self.chunks_added,
            'blocks_extracted': self.blocks_extracted,
            'underruns': self.underruns,
            'max_seconds': self.max_seconds
        }


async def buffered_stream_audio_generator(
    audio_generator: AsyncGenerator[np.ndarray, None],
    input_sr: int = SAMPLING_RATE,
    mapping_scheme: str = 'alternating',
    speaker_id: int = None,
    host: str = "10.40.49.47",
    port: int = 9999,
    volume: float = 1.0,
    auto_resample: bool = True,
    profile: str = 'stable',
    buffer_seconds: float = 2.0,
    min_buffer_start: float = 0.5
) -> None:
    """
    Stream audio from an async generator with client-side buffering.
    
    Decouples irregular audio generation from consistent spatial audio streaming.
    Uses a sliding window buffer to absorb timing irregularities.
    
    Args:
        audio_generator: Async generator yielding audio chunks
        input_sr: Input sample rate
        mapping_scheme: Spatial audio mapping scheme
        speaker_id: Speaker ID for single mapping
        host: Server host
        port: Server port  
        volume: Volume multiplier
        auto_resample: Whether to resample if needed
        profile: Audio profile for spatial streaming
        buffer_seconds: Maximum buffer duration (sliding window)
        min_buffer_start: Minimum buffer before starting playback
    """
    # Initialize buffer and streamer
    audio_buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=input_sr)
    streamer = SoundNetworkStreamer(host=host, port=port, profile=profile)
    
    logger = logging.getLogger(__name__)
    logger.info(f"[BUFFERED_STREAM] Starting with {buffer_seconds}s max buffer, "
               f"{min_buffer_start}s min start buffer")
    
    async def api_receiver_task():
        """Continuously receive from audio generator and buffer."""
        try:
            async for audio_chunk in audio_generator:
                # Resample if needed
                if auto_resample and input_sr != SAMPLING_RATE:
                    from spatial_audio_ai.tools.playback import resample_audio
                    audio_chunk = resample_audio(audio_chunk, input_sr, SAMPLING_RATE)
                
                # Apply volume
                audio_chunk = audio_chunk * volume
                
                # Add to buffer
                await audio_buffer.add_audio(audio_chunk)
                
        except Exception as e:
            logger.error(f"[BUFFERED_STREAM] API receiver error: {e}")
            raise
    
    async def spatial_sender_task():
        """Send at consistent rate to spatial audio."""
        try:
            # Wait for initial buffer
            logger.info(f"[BUFFERED_STREAM] Waiting for {min_buffer_start}s initial buffer...")
            while not audio_buffer.has_minimum_buffer(min_buffer_start):
                await asyncio.sleep(0.01)
            
            logger.info(f"[BUFFERED_STREAM] Starting playback with {audio_buffer.get_buffer_seconds():.2f}s buffered")
            
            # Prepare for streaming
            from spatial_audio_ai.tools.playback import prepare_for_streaming
            
            # Start precise timing loop
            start_time = time.perf_counter()
            block_count = 0
            block_duration = BLOCKSIZE / SAMPLING_RATE
            
            consecutive_underruns = 0
            max_consecutive_underruns = 10  # ~53ms of silence before warning
            
            with streamer:
                while True:
                    # Get audio block
                    audio_block = await audio_buffer.get_block(BLOCKSIZE)
                    
                    if audio_block is None:
                        # Buffer underrun
                        consecutive_underruns += 1
                        
                        if consecutive_underruns == 1:
                            logger.warning(f"[BUFFERED_STREAM] Buffer underrun! "
                                         f"Buffer: {audio_buffer.get_buffer_seconds():.3f}s")
                        elif consecutive_underruns >= max_consecutive_underruns:
                            logger.error(f"[BUFFERED_STREAM] Extended underrun ({consecutive_underruns} blocks), "
                                       f"may need larger buffer or slower generation")
                        
                        # Send silence and continue
                        audio_block = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
                    else:
                        # Reset underrun counter
                        if consecutive_underruns > 0:
                            logger.info(f"[BUFFERED_STREAM] Recovered from {consecutive_underruns} underruns")
                            consecutive_underruns = 0
                    
                    # Prepare for spatial audio
                    multi_channel = prepare_for_streaming(
                        audio_block,
                        mapping_scheme=mapping_scheme,
                        speaker_id=speaker_id
                    )
                    
                    # Send to spatial audio
                    streamer.send(multi_channel)
                    
                    # Precise timing
                    block_count += 1
                    next_time = start_time + block_count * block_duration
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
                    # Log stats occasionally
                    if block_count % 1000 == 0:  # Every ~5 seconds
                        stats = audio_buffer.get_stats()
                        logger.info(f"[BUFFERED_STREAM] Stats: {stats}")
                        
        except Exception as e:
            logger.error(f"[BUFFERED_STREAM] Spatial sender error: {e}")
            raise
    
    # Run both tasks concurrently
    async with asyncio.TaskGroup() as tg:
        tg.create_task(api_receiver_task())
        tg.create_task(spatial_sender_task())


# Convenience function with simpler interface
async def stream_audio_buffered(
    audio_generator: AsyncGenerator[np.ndarray, None],
    **kwargs
) -> None:
    """
    Simplified interface for buffered audio streaming.
    
    Args:
        audio_generator: Async generator yielding audio chunks
        **kwargs: Arguments passed to buffered_stream_audio_generator
    """
    await buffered_stream_audio_generator(audio_generator, **kwargs)


def setup_buffered_streaming_logging(level: str = "INFO") -> None:
    """
    Set up logging for buffered streaming debugging.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logger = logging.getLogger("spatial_audio_ai.tools.buffered_streaming")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.info(f"[BUFFERED_STREAM] Logging configured at {level} level") 