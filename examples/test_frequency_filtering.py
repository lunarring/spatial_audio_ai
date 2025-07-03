#!/usr/bin/env python3
"""
Test script for frequency filtering capabilities

This script demonstrates how different filter types affect the harmonic content.
"""

import numpy as np
from spatial_audio_ai.tools.sound_objects import SO_PlaybackMultiHarmonic


def test_frequency_filters():
    """Test different frequency filter types and show their effects."""
    
    print("FREQUENCY FILTERING TEST")
    print("=" * 50)
    
    # Create a harmonic object with many harmonics for testing
    harmonic_obj = SO_PlaybackMultiHarmonic(
        fundamental_frequency=440.0,  # A4 - 440 Hz
        num_harmonics=8,
        harmonic_decay=0.7
    )
    
    # Set orientation to generate some harmonic content
    test_orientation = np.array([0.04169543, 0.28970698, -0.09529953, -0.951446])
    harmonic_obj.set_orientation(test_orientation)
    
    print(f"Base frequency: {harmonic_obj.current_frequency:.1f} Hz")
    print(f"Test orientation: [{test_orientation[0]:.3f}, {test_orientation[1]:.3f}, {test_orientation[2]:.3f}, {test_orientation[3]:.3f}]")
    print()
    
    # Calculate harmonic frequencies for reference
    harmonic_freqs = [harmonic_obj.current_frequency * (i + 1) for i in range(harmonic_obj.num_harmonics)]
    print("Harmonic frequencies:")
    for i, freq in enumerate(harmonic_freqs):
        print(f"  Harmonic {i+1}: {freq:.1f} Hz")
    print()
    
    # Test different filter types
    filter_configs = [
        {"enabled": False, "type": "none", "description": "No Filter"},
        {"enabled": True, "type": "lowpass", "cutoff_high": 1000.0, "description": "Low-pass (< 1000 Hz)"},
        {"enabled": True, "type": "lowpass", "cutoff_high": 2000.0, "description": "Low-pass (< 2000 Hz)"},
        {"enabled": True, "type": "highpass", "cutoff_low": 1000.0, "description": "High-pass (> 1000 Hz)"},
        {"enabled": True, "type": "bandpass", "cutoff_low": 800.0, "cutoff_high": 2000.0, "description": "Band-pass (800-2000 Hz)"},
        {"enabled": True, "type": "notch", "cutoff_low": 1200.0, "cutoff_high": 1800.0, "description": "Notch (remove 1200-1800 Hz)"},
    ]
    
    for config in filter_configs:
        print(f"Filter: {config['description']}")
        
        if config["enabled"]:
            harmonic_obj.set_frequency_filter(
                enabled=True,
                filter_type=config["type"],
                cutoff_low=config.get("cutoff_low", 1000.0),
                cutoff_high=config.get("cutoff_high", 4000.0),
                rolloff=12
            )
        else:
            harmonic_obj.set_frequency_filter(enabled=False)
        
        # Get filtered amplitudes
        amplitudes = harmonic_obj.get_harmonic_amplitudes()
        
        # Show which harmonics are affected
        print("  Harmonic amplitudes:")
        for i, (freq, amp) in enumerate(zip(harmonic_freqs, amplitudes)):
            status = "MUTED" if amp < 0.01 else f"{amp:.3f}"
            print(f"    H{i+1} ({freq:.0f}Hz): {status}")
        print()
    
    # Test custom harmonic mask
    print("Custom Harmonic Mask Test:")
    print("Keeping only harmonics 1, 3, and 5 (odd harmonics)")
    
    harmonic_obj.set_frequency_filter(enabled=True, filter_type="custom")
    custom_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Only odd harmonics
    harmonic_obj.set_custom_harmonic_mask(custom_mask)
    
    amplitudes = harmonic_obj.get_harmonic_amplitudes()
    
    print("  Custom mask applied:")
    for i, (freq, amp) in enumerate(zip(harmonic_freqs, amplitudes)):
        status = "ACTIVE" if amp > 0.01 else "MUTED"
        print(f"    H{i+1} ({freq:.0f}Hz): {status} ({amp:.3f})")


def demonstrate_filter_rolloff():
    """Demonstrate different filter rolloff rates."""
    
    print("\n" + "=" * 50)
    print("FILTER ROLLOFF DEMONSTRATION")
    print("=" * 50)
    
    harmonic_obj = SO_PlaybackMultiHarmonic(
        fundamental_frequency=220.0,  # Lower frequency for more harmonics in range
        num_harmonics=12,
        harmonic_decay=0.8
    )
    
    # Set orientation for consistent harmonics
    harmonic_obj.set_orientation(np.array([0.5, 0.5, 0.5, 0.5]))
    
    harmonic_freqs = [harmonic_obj.current_frequency * (i + 1) for i in range(harmonic_obj.num_harmonics)]
    
    # Test different rolloff rates with same cutoff
    cutoff_freq = 1000.0
    rolloff_rates = [6, 12, 18, 24]
    
    print(f"Low-pass filter at {cutoff_freq} Hz with different rolloff rates:")
    print()
    
    for rolloff in rolloff_rates:
        harmonic_obj.set_frequency_filter(
            enabled=True,
            filter_type="lowpass",
            cutoff_high=cutoff_freq,
            rolloff=rolloff
        )
        
        amplitudes = harmonic_obj.get_harmonic_amplitudes()
        
        print(f"Rolloff: {rolloff} dB/octave")
        for i, (freq, amp) in enumerate(zip(harmonic_freqs, amplitudes)):
            if freq > cutoff_freq:
                octaves_above = np.log2(freq / cutoff_freq)
                attenuation_db = -rolloff * octaves_above
                print(f"  H{i+1} ({freq:.0f}Hz): {amp:.3f} ({attenuation_db:.1f}dB)")
            else:
                print(f"  H{i+1} ({freq:.0f}Hz): {amp:.3f} (passband)")
        print()


if __name__ == "__main__":
    test_frequency_filters()
    demonstrate_filter_rolloff()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Frequency filtering options:")
    print("• LOWPASS: Remove high frequencies (keep only lower harmonics)")
    print("• HIGHPASS: Remove low frequencies (keep only higher harmonics)")  
    print("• BANDPASS: Keep only frequencies in a specific range")
    print("• NOTCH: Remove frequencies in a specific range")
    print("• CUSTOM: Manually set amplitude for each harmonic")
    print()
    print("This is perfect for:")
    print("• Removing harsh high frequencies")
    print("• Creating warm, mellow tones (lowpass)")
    print("• Creating bright, airy sounds (highpass)")
    print("• Isolating specific frequency ranges")
    print("• Creative sound design with custom masks") 