"""
Spatial Audio AI - A toolkit for audio processing, spatial sound control 
and AI-based audio generation.
"""

__version__ = "0.1.0"

# Import core components
from spatial_audio_ai.tools.sound_system import SoundSystem
from spatial_audio_ai.tools.spatializer import (
    Spatializer, 
    Scene,
    SO_Playback,
    SO_PlaybackSine
)
from spatial_audio_ai.tools.fast_network import (
    FastAudioStreamer,
    QueueManagedStreamer
)
from spatial_audio_ai.config import (
    get_sample_rate,
    set_sample_rate,
    SAMPLING_RATE,
    BLOCKSIZE
)

# Make commonly used items available at package level
__all__ = [
    'SoundSystem',
    'Spatializer',
    'Scene',
    'SO_Playback',
    'SO_PlaybackSine',
    'FastAudioStreamer',
    'QueueManagedStreamer',
    'get_sample_rate',
    'set_sample_rate',
    'SAMPLING_RATE',
    'BLOCKSIZE',
] 