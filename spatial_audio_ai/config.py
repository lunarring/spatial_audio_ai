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
        rate: Sample rate ("44100" or "48000") - only 48000 is
              actually supported
    """
    if rate != "48000":
        print(f"Warning: Requested {rate}Hz but system always uses 48KHz")
    # Don't actually change anything - always use 48KHz


# Main configuration constants

# BLOCKSIZE optimized for 48KHz
# Target: ~21.3ms buffer (1024 samples at 48KHz)
# Note: Server must be restarted if this changes
BLOCKSIZE = 1024
# Define CHUNKSIZE for buffer sizes throughout the library
# Reduced from BLOCKSIZE * 4 to BLOCKSIZE for lower latency
CHUNKSIZE = BLOCKSIZE

# Audio processing settings
DEFAULT_FADE_DURATION = 0.4
DEFAULT_REVERB_DECAY = 0.5

# Network settings  
DEFAULT_HOST = "10.40.49.47"
DEFAULT_PORT = 9999

# Hardware settings
N_SPEAKERS = 13

print("Spatial Audio AI initialized:")
print(f"  Sample Rate: {SAMPLING_RATE} Hz (fixed)")
print(f"  Block Size: {BLOCKSIZE} samples "
      f"(~{BLOCKSIZE/SAMPLING_RATE*1000:.1f}ms)") 