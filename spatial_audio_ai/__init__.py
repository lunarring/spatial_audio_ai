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
    SO_PlaybackSine,
    SO_PlaybackMultiHarmonic
)
from spatial_audio_ai.tools.client import (
    SoundNetworkStreamer
)
from spatial_audio_ai.config import (
    get_sample_rate,
    set_sample_rate,
    SAMPLING_RATE
)

# Make commonly used items available at package level
__all__ = [
    'SoundSystem',
    'Spatializer',
    'Scene',
    'SO_Playback',
    'SO_PlaybackSine',
    'SO_PlaybackMultiHarmonic',
    'SoundNetworkStreamer',
    'get_sample_rate',
    'set_sample_rate',
    'SAMPLING_RATE',
] 