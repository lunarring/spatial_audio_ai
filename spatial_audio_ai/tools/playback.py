#!/usr/bin/env python3
"""
Audio file playback utility for the spatial audio AI system.
Loads .wav and .mp3 files and streams them over the network.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import time
from spatial_audio_ai.tools.client import SoundNetworkStreamer
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE


def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return the audio data and sample rate.
    
    Args:
        file_path: Path to the audio file (.wav or .mp3)
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check file extension
    ext = Path(file_path).suffix.lower()
    if ext not in ['.wav', '.mp3']:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Only .wav and .mp3 are supported."
        )
    
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(file_path)
        print(f"Loaded {file_path}: {audio_data.shape} samples "
              f"at {sample_rate} Hz")
        return audio_data, sample_rate
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")


def resample_audio(audio_data: np.ndarray, original_sr: int, 
                   target_sr: int) -> np.ndarray:
    """
    High-quality resampling using scipy with anti-aliasing.
    """
    if original_sr == target_sr:
        return audio_data
    
    try:
        from scipy import signal
        
        # Use scipy's resample_poly for better quality when possible
        # (works well for integer ratios)
        ratio = target_sr / original_sr
        if abs(ratio - round(ratio)) < 1e-6:  # Nearly integer ratio
            up = int(round(ratio)) if ratio >= 1 else 1
            down = 1 if ratio >= 1 else int(round(1/ratio))
            resampled = signal.resample_poly(audio_data, up, down)
        else:
            # Use standard resample with windowing for non-integer ratios
            num_samples = int(len(audio_data) * target_sr / original_sr)
            resampled = signal.resample(audio_data, num_samples, window='hann')
        
        print(f"Resampled from {original_sr} Hz to {target_sr} Hz (ratio: {ratio:.3f})")
        return resampled.astype(np.float32)
    except ImportError:
        print(f"Warning: scipy not available for resampling. "
              f"Playing at original rate {original_sr} Hz")
        return audio_data


def prepare_for_streaming(audio_data: np.ndarray, 
                         n_speakers: int = 13, 
                         mapping_scheme: str = 'alternating',
                         speaker_id: int = None) -> np.ndarray:
    """
    Prepare audio data for streaming by formatting it for the spatial system.
    
    Args:
        audio_data: Audio data (mono or stereo)
        n_speakers: Number of speakers in the system
        mapping_scheme: How to distribute audio ('alternating', 'stereo', 'mono', 'single')
        speaker_id: Which speaker to route audio to (1-based, only for 'single' mode)
        
    Returns:
        np.ndarray: Audio data formatted for the spatial system
    """
    # Get audio length
    audio_length = len(audio_data)
    
    # Pad to make divisible by BLOCKSIZE
    remainder = audio_length % BLOCKSIZE
    if remainder != 0:
        padding = BLOCKSIZE - remainder
        if len(audio_data.shape) == 2:
            audio_data = np.pad(audio_data, ((0, padding), (0, 0)), mode='constant')
        else:
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        audio_length = len(audio_data)
    
    # Handle different audio formats
    if len(audio_data.shape) == 1:
        # Mono - duplicate to stereo
        left = right = audio_data
    else:
        # Stereo or multi-channel
        left = audio_data[:, 0]
        right = audio_data[:, 1] if audio_data.shape[1] > 1 else audio_data[:, 0]
    
    # Create multi-channel array (shape: [n_speakers, audio_length])
    multi_channel_audio = np.zeros((n_speakers, audio_length), dtype=np.float32)
    
    if mapping_scheme == 'single' and speaker_id is not None:
        # Route to single speaker (legacy mode)
        speaker_idx = speaker_id - 1
        if 0 <= speaker_idx < n_speakers:
            mono_signal = (left + right) / 2
            multi_channel_audio[speaker_idx] = mono_signal
            print(
                f"Prepared audio for streaming: {multi_channel_audio.shape} "
                f"routed to speaker {speaker_id}"
            )
        else:
            raise ValueError(
                f"Speaker ID must be between 1 and {n_speakers}"
            )
    
    elif mapping_scheme == 'alternating':
        # Alternating mapping: even channels left, odd channels right
        for i in range(12):
            if i % 2 == 0:
                multi_channel_audio[i] = left
            else:
                multi_channel_audio[i] = right
        # 13th channel is sum of left and right
        multi_channel_audio[12] = (left + right) / 2
        print(
            f"Prepared audio for streaming: {multi_channel_audio.shape} "
            f"with alternating mapping"
        )
    
    elif mapping_scheme == 'stereo':
        # Stereo (grouped) mapping: first 6 channels are left, next 6 channels are right
        for i in range(6):
            multi_channel_audio[i] = left
        for i in range(6, 12):
            multi_channel_audio[i] = right
        # 13th channel is sum of left and right
        multi_channel_audio[12] = (left + right) / 2
        print(
            f"Prepared audio for streaming: {multi_channel_audio.shape} "
            f"with stereo mapping"
        )
    
    elif mapping_scheme == 'mono':
        # Mono mapping: averaged left+right signal to all channels
        mono_signal = (left + right) / 2
        for i in range(12):
            multi_channel_audio[i] = mono_signal
        # 13th channel is also the mono signal
        multi_channel_audio[12] = mono_signal
        print(
            f"Prepared audio for streaming: {multi_channel_audio.shape} "
            f"with mono mapping"
        )
    
    else:
        raise ValueError(
            f"Invalid mapping scheme: {mapping_scheme}. "
            f"Must be 'alternating', 'stereo', 'mono', or 'single'"
        )
    
    return multi_channel_audio


def stream_audio_file_buffered(file_path: str, 
                               mapping_scheme: str = 'alternating',
                               speaker_id: int = None,
                               host: str = "10.40.49.47", 
                               port: int = 9999,
                               volume: float = 1.0,
                               auto_resample: bool = True,
                               buffer_seconds: float = 2.0,
                               min_buffer_chunks: int = 5,
                               adaptive_buffering: bool = True):
    """
    Load and stream an audio file over the network with intelligent buffering
    for better WiFi stability.
    
    Args:
        file_path: Path to the audio file
        mapping_scheme: How to distribute audio ('alternating', 'stereo', 'mono', 'single')
        speaker_id: Speaker to route audio to (1-13, only for 'single' mode)
        host: Server host address
        port: Server port
        volume: Volume multiplier (0.0 to 2.0)
        auto_resample: Whether to resample to system rate
        buffer_seconds: Target buffer time on server (seconds)
        min_buffer_chunks: Minimum chunks to send before starting playback timing
        adaptive_buffering: Whether to adjust buffer size based on network performance
    """
    # Validate parameters
    if mapping_scheme == 'single':
        if speaker_id is None:
            raise ValueError("Speaker ID required for 'single' mapping scheme")
        if not 1 <= speaker_id <= 13:
            raise ValueError("Speaker ID must be between 1 and 13")
    
    if not 0.0 <= volume <= 2.0:
        raise ValueError("Volume must be between 0.0 and 2.0")
    
    if buffer_seconds < 0.5:
        raise ValueError("Buffer time must be at least 0.5 seconds")
    
    # Load and prepare audio
    audio_data, sample_rate = load_audio_file(file_path)
    
    if sample_rate != SAMPLING_RATE:
        if auto_resample:
            print(f"Warning: File is {sample_rate}Hz but system expects {SAMPLING_RATE}Hz")
            print("Resampling for compatibility (may affect quality)")
            audio_data = resample_audio(audio_data, sample_rate, SAMPLING_RATE)
            sample_rate = SAMPLING_RATE
        else:
            raise ValueError(f"Sample rate mismatch: file is {sample_rate}Hz, system expects {SAMPLING_RATE}Hz")
    
    audio_data = audio_data * volume
    stream_data = prepare_for_streaming(
        audio_data,
        mapping_scheme=mapping_scheme,
        speaker_id=speaker_id
    )
    
    # Calculate streaming parameters
    n_speakers, total_samples = stream_data.shape
    n_blocks = total_samples // BLOCKSIZE
    chunk_duration = BLOCKSIZE / sample_rate
    duration = total_samples / sample_rate
    
    # Calculate buffer parameters
    buffer_chunks = max(int(buffer_seconds / chunk_duration), min_buffer_chunks)
    print(f"Target buffer: {buffer_chunks} chunks ({buffer_chunks * chunk_duration:.2f}s)")
    
    # Connect and start buffered streaming
    print(f"Connecting to server at {host}:{port}")
    streamer = SoundNetworkStreamer(host=host, port=port)
    
    try:
        if mapping_scheme == 'single':
            print(f"Streaming {file_path} to speaker {speaker_id} with buffering (duration: {duration:.2f}s)")
        else:
            print(f"Streaming {file_path} to all speakers with {mapping_scheme} mapping and buffering (duration: {duration:.2f}s)")
        
        # Phase 1: Pre-fill buffer
        print(f"Pre-filling buffer with {buffer_chunks} chunks...")
        pre_fill_start = time.perf_counter()
        
        for i in range(min(buffer_chunks, n_blocks)):
            chunk = stream_data[:, i*BLOCKSIZE:(i+1)*BLOCKSIZE]
            send_start = time.perf_counter()
            streamer.send(chunk)
            send_time = time.perf_counter() - send_start
            
            # Monitor send times for adaptive buffering
            if adaptive_buffering and send_time > chunk_duration * 0.5:
                print(f"Warning: Slow send detected ({send_time:.3f}s), may need larger buffer")
        
        pre_fill_duration = time.perf_counter() - pre_fill_start
        print(f"Buffer pre-filled in {pre_fill_duration:.2f}s (avg: {pre_fill_duration/min(buffer_chunks, n_blocks):.3f}s/chunk)")
        
        # Phase 2: Timed streaming for remaining chunks
        if n_blocks > buffer_chunks:
            print("Starting timed streaming for remaining audio...")
            
            # Start timing from when the first buffered chunk should be playing
            playback_start_time = time.perf_counter()
            
            # Stream remaining chunks with timing
            for i in range(buffer_chunks, n_blocks):
                chunk = stream_data[:, i*BLOCKSIZE:(i+1)*BLOCKSIZE]
                
                # Calculate when this chunk should be sent
                # (accounting for the buffer we already sent)
                chunk_play_time = (i - buffer_chunks) * chunk_duration
                target_send_time = playback_start_time + chunk_play_time
                
                # Wait until it's time to send this chunk
                current_time = time.perf_counter()
                sleep_time = target_send_time - current_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif adaptive_buffering and sleep_time < -chunk_duration:
                    print(f"Warning: Running {-sleep_time:.3f}s behind schedule")
                
                # Send the chunk
                send_start = time.perf_counter()
                streamer.send(chunk)
                send_time = time.perf_counter() - send_start
                
                # Adaptive buffering: monitor performance
                if adaptive_buffering and i % 50 == 0:  # Check every ~1 second
                    if send_time > chunk_duration * 0.8:
                        print(f"Network performance warning: send took {send_time:.3f}s")
        
        print("Audio streaming completed. Buffer should continue playing...")
        
        # Calculate total buffer time remaining
        remaining_buffer_time = buffer_chunks * chunk_duration
        print(f"Remaining buffer: ~{remaining_buffer_time:.1f}s")
        
    except Exception as e:
        print(f"Streaming error: {e}")
        raise
    finally:
        streamer.close()


def stream_audio_file(file_path: str, 
                     mapping_scheme: str = 'alternating',
                     speaker_id: int = None,
                     host: str = "10.40.49.47", 
                     port: int = 9999,
                     volume: float = 1.0,
                     auto_resample: bool = True):
    """
    Load and stream an audio file over the network (legacy real-time mode).
    
    Args:
        file_path: Path to the audio file
        mapping_scheme: How to distribute audio ('alternating', 'stereo', 'mono', 'single')
        speaker_id: Speaker to route audio to (1-13, only for 'single' mode)
        host: Server host address
        port: Server port
        volume: Volume multiplier (0.0 to 2.0)
        auto_resample: Whether to resample to system rate
    """
    # Validate speaker ID if using single mode
    if mapping_scheme == 'single':
        if speaker_id is None:
            raise ValueError("Speaker ID required for 'single' mapping scheme")
        if not 1 <= speaker_id <= 13:
            raise ValueError("Speaker ID must be between 1 and 13")
    
    # Validate volume
    if not 0.0 <= volume <= 2.0:
        raise ValueError("Volume must be between 0.0 and 2.0")
    
    # Load audio file
    audio_data, sample_rate = load_audio_file(file_path)
    
    # Check sample rate compatibility
    if sample_rate != SAMPLING_RATE:
        if auto_resample:
            print(f"Warning: File is {sample_rate}Hz but system expects {SAMPLING_RATE}Hz")
            print("Resampling for compatibility (may affect quality)")
            audio_data = resample_audio(audio_data, sample_rate, SAMPLING_RATE)
            sample_rate = SAMPLING_RATE
        else:
            raise ValueError(f"Sample rate mismatch: file is {sample_rate}Hz, system expects {SAMPLING_RATE}Hz. "
                           f"Use --no-resample flag or convert file to {SAMPLING_RATE}Hz")
    
    # Apply volume
    audio_data = audio_data * volume
    
    # Prepare for streaming
    stream_data = prepare_for_streaming(
        audio_data,
        mapping_scheme=mapping_scheme,
        speaker_id=speaker_id
    )
    
    # Calculate duration
    duration = len(audio_data) / sample_rate

    # Stream the audio in chunks
    print(f"Connecting to server at {host}:{port}")
    streamer = SoundNetworkStreamer(host=host, port=port)
    try:
        if mapping_scheme == 'single':
            print(
                f"Streaming {file_path} to speaker {speaker_id} "
                f"(duration: {duration:.2f}s)"
            )
        else:
            print(
                f"Streaming {file_path} to all speakers with "
                f"{mapping_scheme} mapping (duration: {duration:.2f}s)"
            )

        n_speakers, total_samples = stream_data.shape
        n_blocks = total_samples // BLOCKSIZE
        chunk_duration = BLOCKSIZE / sample_rate
        start_time = time.perf_counter()
        for i in range(n_blocks):
            chunk = stream_data[:, i*BLOCKSIZE:(i+1)*BLOCKSIZE]
            streamer.send(chunk)
            # Schedule so that chunk i is sent at (i+1)*chunk_duration
            next_time = start_time + (i + 1) * chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Audio sent. Streaming completed.")
    finally:
        streamer.close()


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description='Stream audio files (.wav, .mp3) over network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  playback song.wav                           # Stream to all speakers (buffered)
  playback song.mp3 --mapping stereo          # Stream with stereo mapping (buffered)
  playback song.wav --mapping single --speaker 5  # Stream to speaker 5 only (buffered)
  playback song.wav --volume 0.5              # Stream at 50% volume (buffered)
  playback song.wav --realtime                # Use legacy real-time mode
  playback song.wav --buffer 3.0              # Use 3-second buffer
        """
    )
    
    parser.add_argument('file', help='Audio file to stream (.wav or .mp3)')
    parser.add_argument('--mapping', '-m', default='alternating',
                       choices=['alternating', 'stereo', 'mono', 'single'],
                       help='Mapping scheme (default: alternating)')
    parser.add_argument('--speaker', '-s', type=int, default=None,
                       help='Speaker number (1-13, required for single mapping)')
    parser.add_argument('--volume', '-v', type=float, default=0.1,
                       help='Volume level (0.0 to 2.0, default: 0.1)')
    parser.add_argument('--host', default="10.40.49.47",
                       help='Server host address (default: 10.40.49.47)')
    parser.add_argument('--port', type=int, default=9999,
                       help='Server port (default: 9999)')
    parser.add_argument('--no-resample', action='store_true',
                       help='Do not resample audio to system rate')
    parser.add_argument('--realtime', action='store_true',
                       help='Use legacy real-time streaming (no buffering)')
    parser.add_argument('--buffer', type=float, default=2.0,
                       help='Buffer size in seconds (default: 2.0, min: 0.5)')
    parser.add_argument('--min-buffer-chunks', type=int, default=5,
                       help='Minimum buffer chunks (default: 5)')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive buffering')
    
    args = parser.parse_args()
    
    try:
        if args.realtime:
            # Use legacy real-time streaming
            stream_audio_file(
                file_path=args.file,
                mapping_scheme=args.mapping,
                speaker_id=args.speaker,
                host=args.host,
                port=args.port,
                volume=args.volume,
                auto_resample=not args.no_resample
            )
        else:
            # Use new buffered streaming (default)
            stream_audio_file_buffered(
                file_path=args.file,
                mapping_scheme=args.mapping,
                speaker_id=args.speaker,
                host=args.host,
                port=args.port,
                volume=args.volume,
                auto_resample=not args.no_resample,
                buffer_seconds=args.buffer,
                min_buffer_chunks=args.min_buffer_chunks,
                adaptive_buffering=not args.no_adaptive
            )
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 