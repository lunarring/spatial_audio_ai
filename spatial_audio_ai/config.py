"""
Spatial Audio AI Configuration

Central configuration for audio settings including sample rates.
Easy switching between different audio configurations.
"""

import os
from typing import Literal

# Audio Configuration Options
SUPPORTED_SAMPLE_RATES = {
    "44100": 44100,  # CD quality
    "48000": 48000,  # Professional/broadcast standard
}

# Default sample rate - change this to switch between configurations
# Can also be overridden via environment variable SPATIAL_AUDIO_SAMPLE_RATE
DEFAULT_SAMPLE_RATE = "48000"  # Changed from 44100 to 48000

def get_sample_rate() -> int:
    """
    Get the current sample rate from config or environment variable.
    
    Returns:
        int: Sample rate in Hz (44100 or 48000)
        
    Environment Variables:
        SPATIAL_AUDIO_SAMPLE_RATE: Override sample rate ("44100" or "48000")
    """
    # Check environment variable first
    env_rate = os.getenv("SPATIAL_AUDIO_SAMPLE_RATE")
    if env_rate and env_rate in SUPPORTED_SAMPLE_RATES:
        return SUPPORTED_SAMPLE_RATES[env_rate]
    
    # Fall back to default
    return SUPPORTED_SAMPLE_RATES[DEFAULT_SAMPLE_RATE]

def set_sample_rate(rate: Literal["44100", "48000"]) -> None:
    """
    Set the sample rate for the current session.
    
    Args:
        rate: Sample rate ("44100" or "48000")
    """
    if rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate: {rate}. Must be one of {list(SUPPORTED_SAMPLE_RATES.keys())}")
    
    os.environ["SPATIAL_AUDIO_SAMPLE_RATE"] = rate

# Main configuration constants
SAMPLING_RATE = get_sample_rate()
BLOCKSIZE = 1024

# Audio processing settings
DEFAULT_FADE_DURATION = 0.4
DEFAULT_REVERB_DECAY = 0.5

# Network settings  
DEFAULT_HOST = "10.40.49.47"
DEFAULT_PORT = 9999

# Hardware settings
N_SPEAKERS = 13

print(f"Spatial Audio AI initialized with sample rate: {SAMPLING_RATE} Hz") 