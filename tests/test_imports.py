"""
Test that all module imports work correctly
"""


def test_main_imports():
    """Test that the main package imports work."""
    import spatial_audio_ai
    
    # Test that imports don't raise errors
    from spatial_audio_ai import (
        SoundSystem,
        Spatializer,
        Scene,
        SO_Playback,
        SO_PlaybackSine,
        FastAudioStreamer,
        QueueManagedStreamer,
        BLOCKSIZE
    )
    
    assert spatial_audio_ai.__version__ == "0.1.0"
    # Just verify objects exist
    assert SoundSystem
    assert Spatializer
    assert Scene
    assert SO_Playback
    assert SO_PlaybackSine
    assert FastAudioStreamer
    assert QueueManagedStreamer
    assert BLOCKSIZE


def test_tools_imports():
    """Test that tool module imports work."""
    from spatial_audio_ai.tools import (
        SoundSystem,
        Spatializer,
        Scene,
        SO_Playback,
        SO_PlaybackSine,
        FastAudioStreamer,
        QueueManagedStreamer,
        apply_fade_in_out
    )
    
    # Just verify objects exist
    assert SoundSystem
    assert Spatializer
    assert Scene
    assert SO_Playback
    assert SO_PlaybackSine
    assert FastAudioStreamer
    assert QueueManagedStreamer
    assert apply_fade_in_out


def test_generators_imports():
    """Test that generator module imports work."""
    from spatial_audio_ai.generators import StableAudioDiffusion
    
    # Just verify object exists
    assert StableAudioDiffusion 