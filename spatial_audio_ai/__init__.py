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
    SO_Playback
)
from spatial_audio_ai.tools.playback_stream import (
    SoundNetworkStreamer
)

# Make commonly used items available at package level
__all__ = [
    'SoundSystem',
    'Spatializer',
    'Scene',
    'SO_Playback',
    'SoundNetworkStreamer',
] 