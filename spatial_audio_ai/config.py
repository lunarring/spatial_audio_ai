"""
Spatial Audio AI Configuration

Central configuration for audio settings including sample rates and profiles.
Easy switching between different audio configurations.
"""

from typing import Literal, Dict

# Audio Configuration Options
SUPPORTED_SAMPLE_RATES = {
    "44100": 44100,  # CD quality
    "48000": 48000,  # Professional/broadcast standard
}

# Fixed sample rate - we always use 48KHz for best quality
SAMPLING_RATE = 48000

# Main configuration constants

# BLOCKSIZE optimized for 48KHz
# Target: ~5.3ms buffer (256 samples at 48KHz)
# Note: Server must be restarted if this changes
BLOCKSIZE = 256
# Define CHUNKSIZE for buffer sizes throughout the library
# Reduced from BLOCKSIZE * 4 to BLOCKSIZE for lower latency
CHUNKSIZE = BLOCKSIZE

# Network settings  
DEFAULT_HOST = "10.40.49.47"
DEFAULT_PORT = 9999

# Audio latency settings
# Lower values = lower latency but higher chance of dropouts
# Higher values = more latency but more stable playback
MAX_AUDIO_QUEUE_DEPTH = 3  # Target: ~16ms latency (3 * 5.3ms blocks)
# For ultra-low latency: set to 2 (~11ms) - may cause dropouts
# For stable playback: set to 4 (~21ms) - more stable but higher latency

# Audio driver latency mode
# 'ultra' = Use 3ms WASAPI drivers + aggressive settings (~25ms total)  
# 'low' = Use 3ms WASAPI drivers + safe settings (~35ms total)
# 'stable' = Use any available drivers + conservative settings (~100ms+ total)
AUDIO_LATENCY_MODE = 'ultra'  # Change to 'ultra' for lowest latency

# Hardware settings
N_SPEAKERS = 13

# UDP streaming settings
USE_UDP = True  # Use UDP instead of TCP for audio streaming
UDP_PORT = DEFAULT_PORT  # UDP port for streaming
UDP_BUFFER_DEPTH = MAX_AUDIO_QUEUE_DEPTH  # Number of blocks to buffer for UDP

# ============================================================================
# PROFILE SYSTEM - Centralized Configuration
# ============================================================================

# Available audio profiles
ALLOWED_PROFILES = {
    'experimental_ultra_low',  # Experimental: 2-block queue, highest dropout risk
    'ultra_low_latency',      # Minimal buffering, reduced dropout risk  
    'low_latency',            # Low buffering, some dropout risk
    'balanced',               # Balanced latency/stability
    'high_buffer',            # Higher buffering, more stable
    'stable'                  # ZMQ-optimized profile with high buffering
}

# Profile definitions with queue depths
# Each profile maps to a multiple of UDP_BUFFER_DEPTH
PROFILE_DEFINITIONS = {
    'experimental_ultra_low': {
        'description': 'EXPERIMENTAL: Absolute minimal latency (~11ms), very high dropout risk',
        'udp_depth_multiplier': 2.0 / 3.0,  # 2 blocks - original ultra_low_latency setting
        'zmq_depth_multiplier': 4.0,        # ZMQ needs more buffering
        'recommended_protocol': 'udp',
        'min_buffer_blocks': 2,              # Start playback after 2 blocks (~11ms)
    },
    'ultra_low_latency': {
        'description': 'Low latency (~21ms), improved stability',
        'udp_depth_multiplier': 4.0 / 3.0,   # 4 blocks (was 1.0 = 3 blocks)  
        'zmq_depth_multiplier': 4.0,        # ZMQ needs more buffering
        'recommended_protocol': 'udp',
        'min_buffer_blocks': 3,              # Start playback after 3 blocks (~16ms)
    },
    'low_latency': {
        'description': 'Low latency (~21ms), moderate dropout risk', 
        'udp_depth_multiplier': 4.0 / 3.0,  # 4 blocks (was hardcoded as 4)
        'zmq_depth_multiplier': 6.0,
        'recommended_protocol': 'udp',
        'min_buffer_blocks': 4,              # Start playback after 4 blocks (~21ms)
    },
    'balanced': {
        'description': 'Balanced latency/stability (~32ms)',
        'udp_depth_multiplier': 2.0,        # UDP_BUFFER_DEPTH * 2
        'zmq_depth_multiplier': 8.0,
        'recommended_protocol': 'udp',
        'min_buffer_blocks': 6,              # Start playback after 6 blocks (~32ms)
    },
    'high_buffer': {
        'description': 'Higher stability (~64ms)',
        'udp_depth_multiplier': 4.0,        # UDP_BUFFER_DEPTH * 4
        'zmq_depth_multiplier': 10.0,
        'recommended_protocol': 'udp',
        'min_buffer_blocks': 12,             # Start playback after 12 blocks (~64ms)
    },
    'stable': {
        'description': 'ZMQ-optimized with high buffering (~191ms)',
        'udp_depth_multiplier': 8.0,        # Fallback for UDP
        'zmq_depth_multiplier': 24.0,       # UDP_BUFFER_DEPTH * 24 (reduce send burstiness)
        'recommended_protocol': 'zmq',
        'min_buffer_blocks': 32,             # Start playback after 32 blocks (~170ms) to avoid underruns
    }
}

def get_profile_queue_depth(profile: str, protocol: str = 'udp') -> int:
    """
    Get the queue depth for a given profile and protocol.
    
    Args:
        profile: Profile name (must be in ALLOWED_PROFILES)
        protocol: 'udp' or 'zmq'
    
    Returns:
        Queue depth in number of blocks
        
    Raises:
        ValueError: If profile is not recognized
    """
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"Invalid profile '{profile}'. Allowed: {sorted(ALLOWED_PROFILES)}")
    
    if profile not in PROFILE_DEFINITIONS:
        # Fallback for any missing profiles
        return UDP_BUFFER_DEPTH * 8
    
    config = PROFILE_DEFINITIONS[profile]
    
    if protocol.lower() == 'zmq':
        multiplier = config['zmq_depth_multiplier']
    else:  # udp or any other protocol
        multiplier = config['udp_depth_multiplier']
    
    return int(UDP_BUFFER_DEPTH * multiplier)

def get_recommended_protocol(profile: str) -> str:
    """
    Get the recommended protocol for a profile.
    
    Args:
        profile: Profile name
        
    Returns:
        'udp' or 'zmq'
    """
    if profile not in PROFILE_DEFINITIONS:
        return 'udp'  # Default fallback
    
    return PROFILE_DEFINITIONS[profile]['recommended_protocol']

def get_min_buffer_blocks(profile: str) -> int:
    """
    Get the minimum buffer blocks required before starting playback for a profile.
    
    Args:
        profile: Profile name (must be in ALLOWED_PROFILES)
        
    Returns:
        Minimum buffer blocks required before playback starts
        
    Raises:
        ValueError: If profile is not recognized
    """
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"Invalid profile '{profile}'. Allowed: {sorted(ALLOWED_PROFILES)}")
    
    if profile not in PROFILE_DEFINITIONS:
        # Fallback for any missing profiles - conservative default
        return 8
    
    return PROFILE_DEFINITIONS[profile]['min_buffer_blocks']

def list_profiles() -> Dict[str, str]:
    """
    Get a dictionary of all profiles with their descriptions.
    
    Returns:
        Dictionary mapping profile names to descriptions
    """
    return {
        profile: config['description'] 
        for profile, config in PROFILE_DEFINITIONS.items()
    }

# Legacy compatibility functions (deprecated)
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

print("Spatial Audio AI initialized:")
print(f"  Sample Rate: {SAMPLING_RATE} Hz (fixed)")
print(f"  Block Size: {BLOCKSIZE} samples "
      f"(~{BLOCKSIZE/SAMPLING_RATE*1000:.1f}ms)")
print(f"  Ultra-Low Latency: ~11ms total (professional grade)")
print(f"  Available Profiles: {sorted(ALLOWED_PROFILES)}") 