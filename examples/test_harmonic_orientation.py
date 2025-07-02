#!/usr/bin/env python3
"""
Test script for orientation-controlled harmonics

This script demonstrates the SO_PlaybackMultiHarmonic class by simulating 
orientation changes and showing how they affect the harmonic content.
"""

import numpy as np
import time
import threading
from spatial_audio_ai.tools.spatializer import (
    SO_PlaybackMultiHarmonic, 
    Spatializer, 
    Scene,
    CHUNKSIZE,
    SAMPLING_RATE
)
from spatial_audio_ai.tools.client import SoundNetworkStreamer


def simulate_orientation_test():
    """Test the orientation-controlled harmonics with simulated data."""
    
    # Create harmonic object
    harmonic_obj = SO_PlaybackMultiHarmonic(
        fundamental_frequency=220.0,  # A3 note
        amplitude=0.4,
        num_harmonics=8,
        harmonic_decay=0.7
    )
    
    # Setup spatial audio
    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.volume = 0.3
    scene.register(harmonic_obj)
    
    sound_streamer = SoundNetworkStreamer()
    
    print("Starting orientation-controlled harmonic test...")
    print("The harmonics will change as we simulate different orientations")
    print("Listen for changes in timbre as the orientation changes")
    
    # Test different orientations
    test_orientations = [
        np.array([0.0, 0.0, 0.0, 1.0]),      # Identity (no rotation)
        np.array([1.0, 0.0, 0.0, 0.0]),      # 180° around X
        np.array([0.0, 1.0, 0.0, 0.0]),      # 180° around Y
        np.array([0.0, 0.0, 1.0, 0.0]),      # 180° around Z
        np.array([0.707, 0.707, 0.0, 0.0]),  # 90° around X
        np.array([0.0, 0.707, 0.707, 0.0]),  # Complex rotation
        np.array([0.5, 0.5, 0.5, 0.5]),      # Balanced quaternion
    ]
    
    orientation_names = [
        "Identity (no rotation)",
        "180° around X-axis", 
        "180° around Y-axis",
        "180° around Z-axis",
        "90° around X-axis",
        "Complex rotation",
        "Balanced quaternion"
    ]
    
    # Timing setup
    chunk_duration = CHUNKSIZE / SAMPLING_RATE
    start_time = time.perf_counter()
    chunk_counter = 0
    orientation_index = 0
    chunks_per_orientation = int(3.0 / chunk_duration)  # 3 seconds per orientation
    
    print(f"\nStarting with: {orientation_names[0]}")
    harmonic_obj.set_orientation(test_orientations[0])
    
    try:
        for chunk in scene.run():
            # Send audio
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            chunk_counter += 1
            
            # Change orientation every 3 seconds
            if chunk_counter % chunks_per_orientation == 0 and chunk_counter > 0:
                orientation_index = (orientation_index + 1) % len(test_orientations)
                new_orientation = test_orientations[orientation_index]
                harmonic_obj.set_orientation(new_orientation)
                
                # Print current harmonic amplitudes
                harmonic_amps = harmonic_obj.get_harmonic_amplitudes()
                harmonic_str = ", ".join([f"{amp:.3f}" for amp in harmonic_amps])
                
                print(f"\n→ Changed to: {orientation_names[orientation_index]}")
                print(f"  Orientation: [{new_orientation[0]:.3f}, {new_orientation[1]:.3f}, {new_orientation[2]:.3f}, {new_orientation[3]:.3f}]")
                print(f"  Harmonic amplitudes: [{harmonic_str}]")
            
            # Precise timing
            next_time = start_time + chunk_counter * chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Stop after testing all orientations twice
            if chunk_counter > len(test_orientations) * 2 * chunks_per_orientation:
                break
                
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    print("\nOrientation test completed!")


def compare_mapping_modes():
    """Compare all orientation mapping modes side by side."""
    
    print("\n" + "="*80)
    print("COMPARING ALL ORIENTATION MAPPING MODES")
    print("="*80)
    
    # Test orientation
    test_quat = np.array([0.04169543, 0.28970698, -0.09529953, -0.951446])  # From your data
    
    # All available mapping modes
    mapping_modes = [
        "quaternion_simple", "quaternion_complex", 
        "basis_fourier", "basis_chebyshev", "basis_legendre", 
        "basis_wavelets", "basis_radial"
    ]
    
    # Create objects for each mapping mode
    objects = {}
    for mode in mapping_modes:
        obj = SO_PlaybackMultiHarmonic(
            fundamental_frequency=220.0,
            num_harmonics=8,
            harmonic_decay=0.6
        )
        obj.set_orientation_mapping_mode(mode)
        obj.set_orientation(test_quat)
        objects[mode] = obj
    
    print(f"Test quaternion: [{test_quat[0]:.3f}, {test_quat[1]:.3f}, {test_quat[2]:.3f}, {test_quat[3]:.3f}]")
    print()
    
    # Show results for each mode
    for mode in mapping_modes:
        amps = objects[mode].get_harmonic_amplitudes()
        print(f"{mode.upper()} mapping:")
        harmonic_str = ", ".join([f"{amp:.3f}" for amp in amps])
        print(f"  [{harmonic_str}]")
        print()
    
    # Show which mode creates the most variation
    variations = {}
    for mode in mapping_modes:
        amps = objects[mode].get_harmonic_amplitudes()
        variation = np.std(amps)  # Standard deviation as measure of variation
        variations[mode] = variation
    
    print("Variation levels (higher = more dynamic):")
    for mode, var in sorted(variations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mode}: {var:.4f}")
    print()
    
    most_dynamic = max(variations.items(), key=lambda x: x[1])
    print(f"Most dynamic mode: {most_dynamic[0]} (std = {most_dynamic[1]:.4f})")


def interactive_orientation_demo():
    """Interactive demo where user can input orientations."""
    
    print("\n" + "="*60)
    print("INTERACTIVE ORIENTATION DEMO")
    print("="*60)
    print("Enter quaternion values to hear how they affect harmonics")
    print("Format: x y z w (e.g., 0.5 0.5 0.5 0.5)")
    print("Press Ctrl+C to stop")
    
    # Create harmonic object
    harmonic_obj = SO_PlaybackMultiHarmonic(
        fundamental_frequency=330.0,  # E4 note
        amplitude=0.4,
        num_harmonics=6,
        harmonic_decay=0.6
    )
    
    # Setup spatial audio
    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.volume = 0.3
    scene.register(harmonic_obj)
    
    sound_streamer = SoundNetworkStreamer()
    
    # Audio thread
    def audio_thread():
        chunk_duration = CHUNKSIZE / SAMPLING_RATE
        start_time = time.perf_counter()
        chunk_counter = 0
        
        try:
            for chunk in scene.run():
                chunk = np.clip(chunk, -1, 1)
                sound_streamer.send(chunk)
                chunk_counter += 1
                
                next_time = start_time + chunk_counter * chunk_duration
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except:
            pass
    
    # Start audio in background
    audio_thread_obj = threading.Thread(target=audio_thread, daemon=True)
    audio_thread_obj.start()
    
    try:
        while True:
            try:
                user_input = input("\nEnter quaternion (x y z w): ").strip()
                if not user_input:
                    continue
                
                parts = user_input.split()
                if len(parts) != 4:
                    print("Please enter exactly 4 numbers")
                    continue
                
                quat = np.array([float(p) for p in parts])
                
                # Normalize quaternion
                quat_norm = np.linalg.norm(quat)
                if quat_norm == 0:
                    print("Invalid quaternion (all zeros)")
                    continue
                
                quat = quat / quat_norm
                
                # Apply orientation
                harmonic_obj.set_orientation(quat)
                
                # Show results
                harmonic_amps = harmonic_obj.get_harmonic_amplitudes()
                harmonic_str = ", ".join([f"{amp:.3f}" for amp in harmonic_amps])
                
                print(f"Normalized quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                print(f"Harmonic amplitudes: [{harmonic_str}]")
                
            except ValueError:
                print("Please enter valid numbers")
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        pass
    
    print("\nInteractive demo ended")


if __name__ == "__main__":
    print("Multi-Harmonic Orientation Test")
    print("===============================")
    
    # Compare mapping modes first
    compare_mapping_modes()
    
    # Run automated orientation test
    simulate_orientation_test()
    
    # Interactive demo
    try:
        interactive_orientation_demo()
    except KeyboardInterrupt:
        print("\nAll tests completed!") 