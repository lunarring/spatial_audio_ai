#!/usr/bin/env python3
"""
Test script for ZMQ audio streaming

This script demonstrates the ZMQ-based audio streaming functionality
for the 'stable' profile with high latency tolerance.
"""

import numpy as np
import time
from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE

def test_zmq_streaming():
    """Test ZMQ streaming with a simple sine wave"""
    print("Testing ZMQ audio streaming...")
    
    # Generate a simple sine wave
    duration = 10  # Duration in seconds
    frequency = 440  # A4 note frequency in Hz
    sample_rate = SAMPLING_RATE
    
    # Create a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Create a playback sound object
    sound_object = SO_Playback(sine_wave)
    
    # Set up the spatial audio scene
    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.volume = 0.5
    scene.register(sound_object)
    
    # Use ZMQ streamer with stable profile
    print("Connecting via ZMQ with stable profile...")
    sound_streamer = SoundNetworkStreamer(profile="stable_zmq")
    
    # Stream timing (more relaxed for stable profile)
    chunk_duration = CHUNKSIZE / SAMPLING_RATE
    start_time = time.perf_counter()
    chunks_sent = 0
    
    print(f"Starting to stream {duration}s sine wave at {frequency}Hz via ZMQ...")
    print("This uses JSON-based numpy array transmission for reliability.")
    
    try:
        for j, chunk in enumerate(scene.run()):
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            chunks_sent += 1
            
            # Less precise timing for stable profile (acceptable for 1s+ latency)
            if chunks_sent % 10 == 0:  # Every ~200ms
                elapsed = time.perf_counter() - start_time
                expected = chunks_sent * chunk_duration
                print(f"[ZMQ] Sent {chunks_sent} chunks, elapsed: {elapsed:.2f}s, expected: {expected:.2f}s")
            
            # Relaxed timing - just prevent overwhelming the network
            time.sleep(chunk_duration * 0.5)  # Half-speed for stable streaming
            
    except KeyboardInterrupt:
        print("\nStopping ZMQ stream...")
    finally:
        sound_streamer.close()
        print(f"ZMQ test completed. Sent {chunks_sent} chunks total.")

def test_protocol_comparison():
    """Compare UDP vs ZMQ streaming"""
    print("\n" + "="*60)
    print("PROTOCOL COMPARISON TEST")
    print("="*60)
    
    # Test data
    test_duration = 2  # seconds
    frequency = 880  # Hz
    sample_rate = SAMPLING_RATE
    
    t = np.linspace(0, test_duration, int(sample_rate * test_duration), False)
    sine_wave = 0.2 * np.sin(2 * np.pi * frequency * t)
    sound_object = SO_Playback(sine_wave)
    
    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.volume = 0.4
    scene.register(sound_object)
    
    # Test UDP (low latency)
    print("\n1. Testing UDP streaming (ultra_low_latency profile)...")
    try:
        udp_streamer = SoundNetworkStreamer(profile="ultra_low_latency")
        start_time = time.perf_counter()
        
        for j, chunk in enumerate(scene.run()):
            chunk = np.clip(chunk, -1, 1)
            udp_streamer.send(chunk)
            if j >= 50:  # Send ~1 second worth
                break
        
        udp_time = time.perf_counter() - start_time
        udp_streamer.close()
        print(f"   UDP completed in {udp_time:.3f}s")
        
    except Exception as e:
        print(f"   UDP test failed: {e}")
    
    # Reset scene
    scene.register(SO_Playback(sine_wave))
    
    # Test ZMQ (stable)
    print("\n2. Testing ZMQ streaming (stable_zmq profile)...")
    try:
        zmq_streamer = SoundNetworkStreamer(profile="stable_zmq")
        start_time = time.perf_counter()
        
        for j, chunk in enumerate(scene.run()):
            chunk = np.clip(chunk, -1, 1)
            zmq_streamer.send(chunk)
            if j >= 50:  # Send ~1 second worth
                break
        
        zmq_time = time.perf_counter() - start_time
        zmq_streamer.close()
        print(f"   ZMQ completed in {zmq_time:.3f}s")
        
    except Exception as e:
        print(f"   ZMQ test failed: {e}")
    
    print("\nProtocol comparison complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ZMQ audio streaming')
    parser.add_argument('--comparison', action='store_true', help='Run protocol comparison test')
    args = parser.parse_args()
    
    if args.comparison:
        test_protocol_comparison()
    else:
        test_zmq_streaming() 