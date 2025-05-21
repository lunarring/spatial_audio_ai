import numpy as np
import time
from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE

# Generate a simple sine wave
sample_rate = 44100  # Standard sample rate
duration = 5  # Duration in seconds
frequency = 440  # A4 note frequency in Hz

# Create a sine wave
t = np.linspace(0, duration, int(sample_rate * duration), False)
# Amplitude 0.5 to avoid clipping
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Create a playback sound object
sound_object = SO_Playback(sine_wave)

# Set up the spatial audio scene
spatializer = Spatializer()
scene = Scene(spatializer)
scene.register(sound_object)

sound_streamer = SoundNetworkStreamer()
for j, chunk in enumerate(scene.run()):
    chunk = np.clip(chunk, -1, 1)
    sound_streamer.send(chunk)
    time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)