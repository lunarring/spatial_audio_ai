"""
Spatial Audio AI Configuration

Central configuration for audio settings including sample rates.
Easy switching between different audio configurations.
"""

from typing import Literal

# Audio Configuration Options
SUPPORTED_SAMPLE_RATES = {
    "44100": 44100,  # CD quality
    "48000": 48000,  # Professional/broadcast standard
}

# Fixed sample rate - we always use 48KHz for best quality
SAMPLING_RATE = 48000


def get_sample_rate() -> int:
    """
    Get the current sample rate (always 48KHz).
    
    Returns:
        int: Sample rate in Hz (48000)
    """
    return SAMPLING_RATE


def set_sample_rate(rate: Literal["44100", "48000"]) -> None:
    """
    Legacy function for compatibility. 
    Note: System now always uses 48KHz.
    
    Args:
        rate: Sample rate - only 48000 is actually supported
    """
    if rate != "48000":
        print(f"Warning: Requested {rate}Hz but system always uses 48KHz")
    # Don't actually change anything - always use 48KHz


# Main configuration constants

# BLOCKSIZE optimized for ultra-low latency
# Target: ~2.7ms buffer (128 samples at 48KHz)
BLOCKSIZE = 128

# Audio processing settings
DEFAULT_FADE_DURATION = 0.4
DEFAULT_REVERB_DECAY = 0.5

# Network settings  
DEFAULT_HOST = "10.40.49.47"
DEFAULT_PORT = 9999

# Hardware settings
N_SPEAKERS = 13

# Real-time performance settings
MAX_QUEUE_SIZE = 4  # Maximum audio chunks in queue (prevents buildup)
QUEUE_WARNING_THRESHOLD = 2  # Warn when queue exceeds this size

print("Spatial Audio AI initialized (LOW LATENCY MODE):")
print(f"  Sample Rate: {SAMPLING_RATE} Hz (fixed)")
print(f"  Block Size: {BLOCKSIZE} samples " +
      f"(~{BLOCKSIZE/SAMPLING_RATE*1000:.1f}ms)") 
print(f"  Max Queue: {MAX_QUEUE_SIZE} chunks") 