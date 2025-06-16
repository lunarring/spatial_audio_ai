import numpy as np
import time
from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE

# Configuration
RADIUS = 5.0  # Radius of the circular path (in units used by the spatializer)
PERIOD = 10.0  # Period of rotation in seconds (adjustable parameter)

# Generate a simple sine wave
from spatial_audio_ai import get_sample_rate
sample_rate = get_sample_rate()  # Use configured sample rate
# Duration in seconds - enough time to complete multiple rotations
duration = 30  
frequency = 440  # A4 note frequency in Hz

# Create a sine wave
sample_count = int(sample_rate * duration)
t = np.linspace(0, duration, sample_count, False)
# Amplitude 0.5 to avoid clipping
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Create a playback sound object (starting at center)
sound_object = SO_Playback(sine_wave)

# Set up the spatial audio scene
spatializer = Spatializer()
scene = Scene(spatializer)
scene.register(sound_object)

# Start time to calculate position
start_time = time.time()

sound_streamer = SoundNetworkStreamer()
for j, chunk in enumerate(scene.run()):
    # Calculate the current angle based on time
    elapsed_time = time.time() - start_time
    # Convert elapsed time to angle in radians
    # 2π radians = one full rotation in PERIOD seconds
    angle = 2 * np.pi * (elapsed_time % PERIOD) / PERIOD
    
    # Calculate new position on circle with radius RADIUS
    x = RADIUS * np.cos(angle)
    y = RADIUS * np.sin(angle)
    
    # Update sound object position
    sound_object.set_position(np.array([x, y], dtype=float))
    
    # Print position information (optional)
    if j % 20 == 0:  # Print every 20th update to reduce console output
        print(f"Time: {elapsed_time:.2f}s, Position: ({x:.2f}, {y:.2f})")
    
    # Send audio to output
    chunk = np.clip(chunk, -1, 1)
    sound_streamer.send(chunk)
    
    # Sleep to maintain proper timing
    time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01) 