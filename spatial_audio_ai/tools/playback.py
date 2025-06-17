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
            print(f"Prepared audio for streaming: {multi_channel_audio.shape} "
                  f"routed to speaker {speaker_id}")
        else:
            raise ValueError(f"Speaker ID must be between 1 and {n_speakers}")
    
    elif mapping_scheme == 'alternating':
        # Alternating mapping: even channels left, odd channels right
        for i in range(12):
            if i % 2 == 0:
                multi_channel_audio[i] = left
            else:
                multi_channel_audio[i] = right
        # 13th channel is sum of left and right
        multi_channel_audio[12] = (left + right) / 2
        print(f"Prepared audio for streaming: {multi_channel_audio.shape} "
              f"with alternating mapping")
    
    elif mapping_scheme == 'stereo':
        # Stereo (grouped) mapping: first 6 channels are left, next 6 channels are right
        for i in range(6):
            multi_channel_audio[i] = left
        for i in range(6, 12):
            multi_channel_audio[i] = right
        # 13th channel is sum of left and right
        multi_channel_audio[12] = (left + right) / 2
        print(f"Prepared audio for streaming: {multi_channel_audio.shape} "
              f"with stereo mapping")
    
    elif mapping_scheme == 'mono':
        # Mono mapping: averaged left+right signal to all channels
        mono_signal = (left + right) / 2
        for i in range(12):
            multi_channel_audio[i] = mono_signal
        # 13th channel is also the mono signal
        multi_channel_audio[12] = mono_signal
        print(f"Prepared audio for streaming: {multi_channel_audio.shape} "
              f"with mono mapping")
    
    else:
        raise ValueError(f"Invalid mapping scheme: {mapping_scheme}. "
                        f"Must be 'alternating', 'stereo', 'mono', or 'single'")
    
    return multi_channel_audio


def stream_audio_file(file_path: str, 
                     mapping_scheme: str = 'alternating',
                     speaker_id: int = None,
                     host: str = "10.40.49.47", 
                     port: int = 9999,
                     volume: float = 1.0,
                     auto_resample: bool = True):
    """
    Load and stream an audio file over the network.
    
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
    stream_data = prepare_for_streaming(audio_data, 
                                       mapping_scheme=mapping_scheme,
                                       speaker_id=speaker_id)
    
    # Calculate duration
    duration = len(audio_data) / sample_rate
    
    # Stream the audio
    print(f"Connecting to server at {host}:{port}")
    streamer = SoundNetworkStreamer(host=host, port=port)
    try:
        if mapping_scheme == 'single':
            print(f"Streaming {file_path} to speaker {speaker_id} "
                  f"(duration: {duration:.2f}s)")
        else:
            print(f"Streaming {file_path} to all speakers with {mapping_scheme} mapping "
                  f"(duration: {duration:.2f}s)")
        
        streamer.send(stream_data)
        
        print(f"Audio sent. Keeping connection alive for {duration:.2f} seconds...")
        time.sleep(duration + 0.5)  # Small buffer
        
    finally:
        streamer.close()
        
    print("Streaming completed.")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description='Stream audio files (.wav, .mp3) over network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  playback song.wav                           # Stream to all speakers (alternating)
  playback song.mp3 --mapping stereo          # Stream with stereo mapping
  playback song.wav --mapping single --speaker 5  # Stream to speaker 5 only
  playback song.wav --volume 0.5              # Stream at 50% volume
  playback song.wav --host 192.168.1.100     # Stream to different host
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
    
    args = parser.parse_args()
    
    try:
        stream_audio_file(
            file_path=args.file,
            mapping_scheme=args.mapping,
            speaker_id=args.speaker,
            host=args.host,
            port=args.port,
            volume=args.volume,
            auto_resample=not args.no_resample
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