# ZMQ Audio Streaming Support

This document describes the new ZMQ-based audio streaming functionality added to the spatial audio system.

## Overview

The system now supports dual-protocol audio streaming:
- **UDP**: High-performance, low-latency streaming for real-time applications
- **ZMQ**: Reliable, high-latency streaming for unstable network connections

## Protocol Selection

The protocol is automatically selected based on the profile:

### UDP Profiles (Low Latency)
- `ultra_low_latency`: ~25ms total latency
- `low_latency`: ~35ms total latency  
- `balanced`: ~65ms total latency
- `high_buffer`: ~130ms total latency
- `super_buffer`: ~260ms total latency

### ZMQ Profiles (Stable/Reliable)
- `stable`: JSON-based transmission with high buffer depth
- `stable_zmq`: Optimized ZMQ profile with maximum reliability

## Usage

### Client Side

```python
from spatial_audio_ai import SoundNetworkStreamer

# Auto-select UDP for low latency
udp_streamer = SoundNetworkStreamer(profile="ultra_low_latency")

# Auto-select ZMQ for stable connections
zmq_streamer = SoundNetworkStreamer(profile="stable_zmq")

# Manual ZMQ usage
from spatial_audio_ai import SoundNetworkStreamerZMQ
zmq_streamer = SoundNetworkStreamerZMQ(host="10.40.49.47", zmq_port=5556)
```

### Server Side

```bash
# Start server with both UDP and ZMQ support (default)
python -m spatial_audio_ai.tools.server

# Start with custom ZMQ port
python -m spatial_audio_ai.tools.server --zmq-port 5557

# Disable ZMQ (UDP only)
python -m spatial_audio_ai.tools.server --no-zmq
```

## Technical Details

### ZMQ Message Format

Audio data is sent as JSON messages:

```json
{
  "client_id": "abc12345",
  "audio_data": {
    "seq": 123,
    "timestamp": 1234567890.123,
    "shape": [13, 1024],
    "dtype": "float32",
    "data": [[...], [...], ...]
  }
}
```

Control messages:

```json
{
  "client_id": "abc12345", 
  "control": {
    "profile": "stable_zmq"
  }
}
```

### Performance Characteristics

| Protocol | Latency | Throughput | Reliability | Best For |
|----------|---------|------------|-------------|----------|
| UDP      | 25-260ms | High       | Medium      | Real-time, stable networks |
| ZMQ      | 1000ms+ | Medium     | Very High   | Bad WiFi, mobile connections |

### Buffer Depths

- UDP profiles: 2-24 blocks (43ms - 520ms buffer)
- ZMQ profiles: 24-36 blocks (520ms - 780ms buffer)

## Dependencies

ZMQ support requires:
```bash
pip install git+https://github.com/lunarring/lunar_tools
```

If lunar_tools is not available, the system falls back to UDP-only mode.

## Testing

Run the ZMQ test script:

```bash
# Basic ZMQ streaming test
python examples/test_zmq_streaming.py

# Protocol comparison test  
python examples/test_zmq_streaming.py --comparison
```

## When to Use ZMQ

Use ZMQ streaming when:
- Network connection is unstable (WiFi dropouts, mobile networks)
- Latency tolerance is high (1+ seconds acceptable)
- Reliability is more important than real-time performance
- Client may disconnect/reconnect frequently

Use UDP streaming when:
- Low latency is critical (< 100ms)
- Network connection is stable
- Real-time performance is required
- Hardware constraints limit buffer sizes 