import numpy as np
import time
from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    QueueManagedStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE

# Generate a simple sine wave
from spatial_audio_ai import get_sample_rate
sample_rate = get_sample_rate()  # Use configured sample rate
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

sound_streamer = QueueManagedStreamer()
if sound_streamer.connect():
    for j, chunk in enumerate(scene.run()):
        chunk = np.clip(chunk, -1, 1)
        sound_streamer.send_with_queue_management(chunk)
    sound_streamer.disconnect()
else:
    print("Failed to connect to fast audio server")