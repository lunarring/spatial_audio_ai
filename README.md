# Spatial Audio & AI

## Overview

This repository contains several Python modules that form a toolkit for audio processing, spatial sound control, server integration, and diffusion-based sound generation. It is designed to help developers work with spatial audio, process sound streams, and generate new audio using modern techniques. The key components include playback control, sound spatialization, server functionality, and stable audio diffusion.

## Features

- **Playback Stream**: Module for managing playback streams and audio processing.
- **Server Integration**: Server module that allows interaction with sound systems via APIs.
- **Sound System**: A framework for controlling sound output, with support for multiple speakers.
- **Spatializer**: Spatial sound control for multi-speaker systems.
- **Tools**: Helper functions for audio manipulation, effects like fade-in and fade-out.
- **Stable Audio Diffusion**: Generate creative audio clips using text prompts.
- **BlackHole Audio Client**: Command-line tool for streaming system audio through BlackHole with spatial mapping.

## Installation

### From PyPI (recommended)

```bash
pip install spatial-audio-ai
```

### From source

```bash
git clone https://github.com/lunarring/spatial_audio_ai.git
cd spatial_audio_ai
pip install -e .
```

## Command Line Tools

After installation, you can use the following command-line tools:

### BlackHole Audio Client (`bh`)

Stream system audio through BlackHole with spatial mapping:

```bash
# Start the BlackHole audio client with default settings
bh
```

This will:
- Start capturing audio from BlackHole 64ch device
- Launch a Gradio web interface at http://127.0.0.1:7860
- Begin streaming audio with spatial mapping to your configured server

### Audio File Playback (`playback`)

Stream audio files (.wav, .mp3) over the network:

```bash
# Stream audio file to all speakers with alternating mapping
playback song.wav

# Stream with stereo mapping
playback song.mp3 --mapping stereo

# Stream to a specific speaker only
playback song.wav --mapping single --speaker 5

# Stream at 50% volume
playback song.wav --volume 0.5

# Stream to a different host
playback song.wav --host 192.168.1.100
```

### Advanced Usage

```bash
# BlackHole client with specific mapping
bh blackhole --mapping stereo

# Send test audio to a specific speaker
bh test --speaker 5 --amplitude 0.2 --duration 2.0

# Connect to a different server
bh test --host 192.168.1.100 --port 8888
```

### Available Commands

#### `bh` - BlackHole Audio Client
- `bh`: Start the BlackHole audio relayer with Gradio interface
  - `--mapping`: Choose mapping scheme (`alternating`, `stereo`, `mono`)
- `bh test`: Send test audio to the spatial audio server
  - `--speaker`: Speaker number (1-13, default: 1)
  - `--amplitude`: Audio amplitude (default: 0.1)
  - `--duration`: Duration in seconds (default: 1.0)
  - `--host`: Server host address (default: 10.40.49.47)
  - `--port`: Server port number (default: 9999)

#### `playback` - Audio File Streaming
- `playback <file>`: Stream audio file to spatial audio system
  - `--mapping, -m`: Mapping scheme (`alternating`, `stereo`, `mono`, `single`)
  - `--speaker, -s`: Speaker number (1-13, required for single mapping)
  - `--volume, -v`: Volume level (0.0 to 2.0, default: 0.1)
  - `--host`: Server host address (default: 10.40.49.47)
  - `--port`: Server port (default: 9999)
  - `--no-resample`: Do not resample audio to system rate

#### Mapping Schemes
- **alternating**: Even channels get left audio, odd channels get right audio
- **stereo**: First 6 channels get left audio, next 6 get right audio
- **mono**: All channels get averaged left+right signal
- **single**: Route audio to one specific speaker only

## Usage

```python
import numpy as np
import soundfile as sf
from spatial_audio_ai import SO_Playback, Spatializer, Scene

# Create a simple sine wave
sample_rate = 44100
duration = 5  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
sine_wave = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
 
# Create playback object
so_playback = SO_Playback(sine_wave)

# Set up spatializer and scene
spatializer = Spatializer()
scene = Scene(spatializer)
scene.register(so_playback)
```

## Examples

Check out the `examples/` directory for practical usage examples:

- **`static_sine.py`**: Demonstrates how to create and play simple sine waves through the spatializer system
- **Audio file playback examples**: Various examples showing how to stream different audio formats
- **Spatial mapping demonstrations**: Examples of different channel mapping schemes in action
- **Integration examples**: How to integrate the spatial audio system with other applications

Run any example with:
```bash
python examples/<example_name>.py
```

## Development

To set up for development:

```bash
git clone https://github.com/lunarring/spatial_audio_ai.git
cd spatial_audio_ai
pip install -e .
```

Run tests:

```bash
python -m pytest
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any feature suggestions or bug reports.
