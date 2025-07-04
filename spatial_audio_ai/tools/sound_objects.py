from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import uuid
import time
from spatial_audio_ai.config import SAMPLING_RATE, CHUNKSIZE

@dataclass
class SoundMessage:
    sound: np.ndarray
    position: np.ndarray
    volume: float = 1.0

    def __post_init__(self):
        assert isinstance(self.sound, np.ndarray), "sound must be a numpy ndarray"
        assert self.sound.dtype == float, "sound must be of type float"
        assert self.sound.ndim == 1, "sound must be a 1-dimensional array"
        assert self.sound.size == CHUNKSIZE, f"sound must have size {CHUNKSIZE}"
        assert isinstance(self.position, np.ndarray), "position must be a numpy ndarray"
        assert self.position.dtype == float, "position must be of type float"
        assert self.position.ndim == 1, "position must be a 1-dimensional array"
        assert self.position.size == 2, "position must have size 2"

class SoundObjectTerminatedException(Exception):
    pass

class SoundObjectBase(ABC):
    @abstractmethod
    def query(self, tick: int) -> SoundMessage:
        pass

class SO_Playback(SoundObjectBase):
    def __init__(
        self, 
        sound: np.ndarray, 
        position: np.ndarray | None = np.zeros(2, dtype=float)
    ):
        self.sound = sound
        self.first_tick = None
        self.position = position
        self._id = uuid.uuid4()

    def query(self, tick: int) -> SoundMessage:
        self.update_position()
        if self.first_tick is None:
            self.first_tick = tick
        start = (tick - self.first_tick) * CHUNKSIZE
        sound = self.sound[start:start+CHUNKSIZE]
        if len(sound) == 0:
            raise SoundObjectTerminatedException
        sound = np.pad(sound, (0, CHUNKSIZE - len(sound)), 'constant')
        
        return SoundMessage(sound=sound, position=self.position)
    
    def update_position(self):
        pass

    def set_position(self, new_position: np.ndarray):
        self.position = new_position

    def get_position(self):
        return self.position


class SO_PlaybackCircularMove(SO_Playback):
    def __init__(
        self,
        sound: np.ndarray,
        position: np.ndarray | None = np.zeros(2, dtype=float),
        radius: float = 1.0,
        speed: float = 1.0,
        initial_angle: float = 0.0,
        direction: int = 1,
        center: np.ndarray = np.zeros(2, dtype=float),
        loop: bool = True
    ):
        """
        Circular moving sound object.
        Args:
            sound: The audio data.
            position: Initial position (ignored, calculated from angle).
            radius: Circle radius.
            speed: Angular speed (radians/sec).
            initial_angle: Starting angle (radians).
            direction: 1 for CCW, -1 for CW.
            center: Center of the circle (np.ndarray shape (2,)).
            loop: If True, loop sound when it ends, else terminate.
        """
        super().__init__(sound, position)
        self.radius = radius
        self.speed = speed
        self.initial_angle = initial_angle
        self.direction = direction
        self.center = center
        self.loop = loop
        self.start_time = time.time()

    def update_position(self):
        elapsed_time = time.time() - self.start_time
        angle = self.initial_angle + self.direction * self.speed * elapsed_time
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        self.set_position(np.array([x, y]))

    def query(self, tick: int) -> SoundMessage:
        self.update_position()
        if self.first_tick is None:
            self.first_tick = tick
        start = (tick - self.first_tick) * CHUNKSIZE
        if self.loop:
            start = start % self.sound.size
        sound = self.sound[start:start+CHUNKSIZE]
        if len(sound) == 0:
            raise SoundObjectTerminatedException
        sound = np.pad(sound, (0, CHUNKSIZE - len(sound)), 'constant')
        return SoundMessage(sound=sound, position=self.position)


class SO_PlaybackSine(SoundObjectBase):
    def __init__(
        self,
        frequency: float = 440.0,
        amplitude: float = 0.5,
        phase_offset: float = 0.0,
        position: np.ndarray = np.zeros(2, dtype=float),
        smoothing_factor: float = 0.95,
        mode: str = "smart"
    ):
        """
        Real-time sine wave generator with two operation modes.
        Args:
            frequency: Frequency in Hz.
            amplitude: Amplitude (0.0 to 1.0).
            phase_offset: Phase offset in radians.
            position: Position in 2D space.
            smoothing_factor: Smoothing factor for parameter changes (0.9-0.99).
            mode: Operation mode - "barebone" for immediate changes, "smart" for smooth transitions.
        """
        # Operation mode
        self.mode = mode  # "barebone" or "smart"
        
        # Target values (what we want to reach)
        self.target_frequency = frequency
        self.target_amplitude = amplitude
        self.target_phase_offset = phase_offset
        self.target_position = position.copy()
        
        # Current values (what we're actually using, smoothed in smart mode)
        self.current_frequency = frequency
        self.current_amplitude = amplitude
        self.current_phase_offset = phase_offset
        self.current_position = position.copy()
        
        # Phase state management
        self.current_phase = 0.0
        self.last_frequency = frequency  # Track for phase continuity
        self.last_chunk_end_value = 0.0  # Track last sample value for seamless transitions
        
        # Smoothing configuration (only used in smart mode)
        self.smoothing_factor = smoothing_factor
        self.amplitude_smoothing = smoothing_factor
        self.position_smoothing = smoothing_factor
        self.phase_offset_smoothing = smoothing_factor
        
        # State tracking
        self.last_chunk = np.zeros(CHUNKSIZE, dtype=float)
        self._id = uuid.uuid4()

    def query(self, tick: int) -> SoundMessage:
        """Generate next chunk based on the current mode."""
        if self.mode == "barebone":
            return self._query_barebone(tick)
        else:  # smart mode
            return self._query_smart(tick)

    def _query_barebone(self, tick: int) -> SoundMessage:
        """Barebone mode: immediate parameter changes, no smoothing."""
        chunk_duration = CHUNKSIZE / SAMPLING_RATE
        t = np.linspace(0, chunk_duration, CHUNKSIZE, endpoint=False)
        
        # Use target values directly, no smoothing
        self.current_frequency = self.target_frequency
        self.current_amplitude = self.target_amplitude
        self.current_phase_offset = self.target_phase_offset
        self.current_position = self.target_position.copy()
        
        # Generate sine wave chunk with current parameters
        instantaneous_phase = (
            2 * np.pi * self.current_frequency * t + 
            self.current_phase + 
            self.current_phase_offset
        )
        
        sound_chunk = self.current_amplitude * np.sin(instantaneous_phase)
        
        # Update phase for next chunk (maintain continuity)
        self.current_phase += 2 * np.pi * self.current_frequency * chunk_duration
        self.current_phase = self.current_phase % (2 * np.pi)
        
        # Store last chunk and last sample value
        self.last_chunk = sound_chunk.copy()
        self.last_chunk_end_value = sound_chunk[-1]
        
        return SoundMessage(sound=sound_chunk, position=self.current_position)

    def _query_smart(self, tick: int) -> SoundMessage:
        """Smart mode: smooth parameter transitions with phase continuity for frequency changes."""
        chunk_duration = CHUNKSIZE / SAMPLING_RATE
        t = np.linspace(0, chunk_duration, CHUNKSIZE, endpoint=False)
        
        # Update frequency, preserve phase continuity (no abrupt phase adjustments)
        self.current_frequency = self.target_frequency
        self.last_frequency = self.target_frequency
        
        # Smooth other parameters using exponential smoothing
        self.current_amplitude = (
            self.amplitude_smoothing * self.current_amplitude + 
            (1 - self.amplitude_smoothing) * self.target_amplitude
        )
        
        self.current_phase_offset = (
            self.phase_offset_smoothing * self.current_phase_offset + 
            (1 - self.phase_offset_smoothing) * self.target_phase_offset
        )
        
        self.current_position = (
            self.position_smoothing * self.current_position + 
            (1 - self.position_smoothing) * self.target_position
        )
        
        # Generate sine wave chunk with smoothed parameters
        instantaneous_phase = (
            2 * np.pi * self.current_frequency * t + 
            self.current_phase + 
            self.current_phase_offset
        )
        
        sound_chunk = self.current_amplitude * np.sin(instantaneous_phase)
        
        # Update phase for next chunk (maintain continuity)
        self.current_phase += 2 * np.pi * self.current_frequency * chunk_duration
        self.current_phase = self.current_phase % (2 * np.pi)
        
        # Store last chunk and last sample value
        self.last_chunk = sound_chunk.copy()
        self.last_chunk_end_value = sound_chunk[-1]
        
        return SoundMessage(sound=sound_chunk, position=self.current_position)

    def set_frequency(self, frequency: float):
        """Update target frequency with phase continuity."""
        self.target_frequency = frequency

    def set_amplitude(self, amplitude: float):
        """Update target amplitude (will be smoothed in smart mode)."""
        self.target_amplitude = amplitude

    def set_phase_offset(self, phase_offset: float):
        """Update target phase offset (will be smoothed in smart mode)."""
        self.target_phase_offset = phase_offset

    def set_position(self, position: np.ndarray):
        """Update target position (will be smoothed in smart mode)."""
        self.target_position = position.copy()

    def set_mode(self, mode: str):
        """Set operation mode: 'barebone' or 'smart'."""
        if mode in ["barebone", "smart"]:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'barebone' or 'smart'")

    def get_mode(self):
        """Get current operation mode."""
        return self.mode

    def get_position(self):
        """Get current smoothed position."""
        return self.current_position

    def get_last_chunk(self):
        """Get the last generated chunk."""
        return self.last_chunk
    
    def set_smoothing_factor(self, factor: float):
        """Adjust smoothing factor for all parameters (0.9-0.99). Only used in smart mode."""
        self.smoothing_factor = np.clip(factor, 0.0, 0.99)
        self.amplitude_smoothing = self.smoothing_factor
        self.position_smoothing = self.smoothing_factor
        self.phase_offset_smoothing = self.smoothing_factor
    
    def set_individual_smoothing(self, amplitude: float = None, 
                                position: float = None, 
                                phase_offset: float = None):
        """Set individual smoothing factors for different parameters. Only used in smart mode."""
        if amplitude is not None:
            self.amplitude_smoothing = np.clip(amplitude, 0.0, 0.99)
        if position is not None:
            self.position_smoothing = np.clip(position, 0.0, 0.99)
        if phase_offset is not None:
            self.phase_offset_smoothing = np.clip(phase_offset, 0.0, 0.99)


class SO_PlaybackMultiHarmonic(SoundObjectBase):
    def __init__(
        self,
        fundamental_frequency: float = 440.0,
        amplitude: float = 0.5,
        phase_offset: float = 0.0,
        position: np.ndarray = np.zeros(2, dtype=float),
        smoothing_factor: float = 0.95,
        num_harmonics: int = 8,
        harmonic_decay: float = 0.5
    ):
        """
        Real-time multi-harmonic sine wave generator with orientation-controlled harmonics.
        Args:
            fundamental_frequency: Base frequency in Hz.
            amplitude: Overall amplitude (0.0 to 1.0).
            phase_offset: Phase offset in radians.
            position: Position in 2D space.
            smoothing_factor: Smoothing factor for parameter changes (0.9-0.99).
            num_harmonics: Number of harmonics to generate (including fundamental).
            harmonic_decay: Base decay factor for higher harmonics.
        """
        # Target values (what we want to reach)
        self.target_frequency = fundamental_frequency
        self.target_amplitude = amplitude
        self.target_phase_offset = phase_offset
        self.target_position = position.copy()
        
        # Current values (what we're actually using, smoothed)
        self.current_frequency = fundamental_frequency
        self.current_amplitude = amplitude
        self.current_phase_offset = phase_offset
        self.current_position = position.copy()
        
        # Harmonic configuration
        self.num_harmonics = num_harmonics
        self.harmonic_decay = harmonic_decay
        
        # Harmonic amplitudes controlled by orientation
        self.target_harmonic_amplitudes = np.array([
            harmonic_decay ** i for i in range(num_harmonics)
        ])
        self.current_harmonic_amplitudes = self.target_harmonic_amplitudes.copy()
        
        # Phase state management for each harmonic
        self.current_phases = np.zeros(num_harmonics)
        self.last_frequency = fundamental_frequency
        
        # Smoothing configuration
        self.smoothing_factor = smoothing_factor
        self.amplitude_smoothing = smoothing_factor
        self.position_smoothing = smoothing_factor
        self.phase_offset_smoothing = smoothing_factor
        self.harmonic_smoothing = smoothing_factor
        
        # State tracking
        self.last_chunk = np.zeros(CHUNKSIZE, dtype=float)
        self._id = uuid.uuid4()
        
        # Position-based harmonic control
        self.x_harmonic_control_enabled = True
        self.x_min_position = -2.0  # Min X position
        self.x_max_position = 2.0   # Max X position
        self.x_min_harmonics = 1    # Min number of active harmonics
        self.x_max_harmonics = num_harmonics  # Max number of active harmonics
        
        # Position-based volume control
        self.z_volume_control_enabled = True
        self.z_min_position = -2.0  # Min Z position
        self.z_max_position = 2.0   # Max Z position
        self.z_min_volume = 0.0     # Min volume multiplier
        self.z_max_volume = 1.0     # Max volume multiplier
        self.current_volume_multiplier = 1.0
        
        # Frequency filtering parameters
        self.frequency_filter_enabled = True
        self.filter_type = "lowpass"  # Options: "lowpass", "highpass", "bandpass", "notch", "custom"
        self.filter_cutoff_low = 1000.0  # Hz - low cutoff for bandpass/highpass
        self.filter_cutoff_high = 4000.0  # Hz - high cutoff for bandpass/lowpass
        self.filter_rolloff = 12  # dB/octave (6, 12, 18, 24)
        self.custom_harmonic_mask = np.ones(num_harmonics)  # Custom per-harmonic multipliers

    def query(self, tick: int) -> SoundMessage:
        """Generate next chunk with smooth parameter transitions and multiple harmonics."""
        chunk_duration = CHUNKSIZE / SAMPLING_RATE
        t = np.linspace(0, chunk_duration, CHUNKSIZE, endpoint=False)
        
        # Handle frequency changes with phase continuity
        if abs(self.target_frequency - self.last_frequency) > 0.1:
            self.current_frequency = self.target_frequency
            self.last_frequency = self.target_frequency
        else:
            self.current_frequency = self.target_frequency
        
        # Smooth other parameters using exponential smoothing
        self.current_amplitude = (
            self.amplitude_smoothing * self.current_amplitude + 
            (1 - self.amplitude_smoothing) * self.target_amplitude
        )
        
        self.current_phase_offset = (
            self.phase_offset_smoothing * self.current_phase_offset + 
            (1 - self.phase_offset_smoothing) * self.target_phase_offset
        )
        
        self.current_position = (
            self.position_smoothing * self.current_position + 
            (1 - self.position_smoothing) * self.target_position
        )
        
        # Update harmonic amplitudes based on position
        self._update_harmonics_from_position()
        self._update_volume_from_position()
        
        # Smooth harmonic amplitudes
        self.current_harmonic_amplitudes = (
            self.harmonic_smoothing * self.current_harmonic_amplitudes + 
            (1 - self.harmonic_smoothing) * self.target_harmonic_amplitudes
        )
        
        # Apply frequency filtering if enabled
        if self.frequency_filter_enabled:
            filtered_amplitudes = self._apply_frequency_filter(self.current_harmonic_amplitudes)
        else:
            filtered_amplitudes = self.current_harmonic_amplitudes
        
        # Generate multi-harmonic sound chunk
        sound_chunk = np.zeros(CHUNKSIZE, dtype=float)
        
        for i in range(self.num_harmonics):
            harmonic_freq = self.current_frequency * (i + 1)  # Fundamental, 2nd, 3rd harmonics...
            harmonic_amplitude = self.current_amplitude * filtered_amplitudes[i] * self.current_volume_multiplier
            
            # Generate harmonic with phase continuity
            instantaneous_phase = (
                2 * np.pi * harmonic_freq * t + 
                self.current_phases[i] + 
                self.current_phase_offset
            )
            
            harmonic_chunk = harmonic_amplitude * np.sin(instantaneous_phase)
            sound_chunk += harmonic_chunk
            
            # Update phase for next chunk (maintain continuity)
            self.current_phases[i] += 2 * np.pi * harmonic_freq * chunk_duration
            self.current_phases[i] = self.current_phases[i] % (2 * np.pi)
        
        # Store last chunk
        self.last_chunk = sound_chunk.copy()
        
        return SoundMessage(sound=sound_chunk, position=self.current_position)

    def set_frequency(self, frequency: float):
        """Update target frequency with phase continuity."""
        self.target_frequency = frequency

    def set_amplitude(self, amplitude: float):
        """Update target amplitude (will be smoothed)."""
        self.target_amplitude = amplitude

    def set_phase_offset(self, phase_offset: float):
        """Update target phase offset (will be smoothed)."""
        self.target_phase_offset = phase_offset

    def set_position(self, position: np.ndarray):
        """Update target position (will be smoothed)."""
        self.target_position = position.copy()

    def _update_harmonics_from_position(self):
        """Update number of active harmonics based on X position."""
        if not self.x_harmonic_control_enabled:
            return
        
        x_pos = self.current_position[0]  # X coordinate
        
        # Clamp X position to valid range
        x_clamped = np.clip(x_pos, self.x_min_position, self.x_max_position)
        
        # Map X position to number of active harmonics
        x_ratio = (x_clamped - self.x_min_position) / (self.x_max_position - self.x_min_position)
        num_active_harmonics = int(
            self.x_min_harmonics + x_ratio * (self.x_max_harmonics - self.x_min_harmonics)
        )
        num_active_harmonics = np.clip(num_active_harmonics, self.x_min_harmonics, self.x_max_harmonics)
        
        # Update harmonic amplitudes - only first N harmonics are active
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        # Zero out harmonics beyond the active count
        for i in range(self.num_harmonics):
            if i >= num_active_harmonics:
                base_amplitudes[i] = 0.0
        
        self.target_harmonic_amplitudes = base_amplitudes
    
    def _update_volume_from_position(self):
        """Update volume multiplier based on Z position."""
        if not self.z_volume_control_enabled:
            self.current_volume_multiplier = 1.0
            return
        
        z_pos = self.current_position[1]  # Z coordinate (stored as Y in our 2D position)
        
        # Clamp Z position to valid range
        z_clamped = np.clip(z_pos, self.z_min_position, self.z_max_position)
        
        # Map Z position to volume multiplier
        z_ratio = (z_clamped - self.z_min_position) / (self.z_max_position - self.z_min_position)
        self.current_volume_multiplier = self.z_min_volume + z_ratio * (self.z_max_volume - self.z_min_volume)

    def set_position_harmonic_control(self, enabled: bool, x_min: float = None, x_max: float = None, 
                                     min_harmonics: int = None, max_harmonics: int = None):
        """Configure X position to harmonic count mapping."""
        self.x_harmonic_control_enabled = enabled
        
        if x_min is not None:
            self.x_min_position = x_min
        if x_max is not None:
            self.x_max_position = x_max
        if min_harmonics is not None:
            self.x_min_harmonics = max(1, min_harmonics)
        if max_harmonics is not None:
            self.x_max_harmonics = min(self.num_harmonics, max_harmonics)
    
    def set_position_volume_control(self, enabled: bool, z_min: float = None, z_max: float = None,
                                   min_volume: float = None, max_volume: float = None):
        """Configure Z position to volume mapping."""
        self.z_volume_control_enabled = enabled
        
        if z_min is not None:
            self.z_min_position = z_min
        if z_max is not None:
            self.z_max_position = z_max
        if min_volume is not None:
            self.z_min_volume = max(0.0, min_volume)
        if max_volume is not None:
            self.z_max_volume = max_volume
    
    def get_position_control_info(self):
        """Get current position control configuration."""
        return {
            "x_harmonic_enabled": self.x_harmonic_control_enabled,
            "x_range": (self.x_min_position, self.x_max_position),
            "harmonic_range": (self.x_min_harmonics, self.x_max_harmonics),
            "z_volume_enabled": self.z_volume_control_enabled,
            "z_range": (self.z_min_position, self.z_max_position),
            "volume_range": (self.z_min_volume, self.z_max_volume),
            "current_volume_multiplier": self.current_volume_multiplier
        }



    def _apply_frequency_filter(self, amplitudes):
        """Apply frequency filtering to harmonic amplitudes."""
        filtered_amps = amplitudes.copy()
        
        if self.filter_type == "custom":
            # Use custom harmonic mask
            filtered_amps *= self.custom_harmonic_mask
        else:
            # Calculate frequency for each harmonic
            harmonic_freqs = np.array([
                self.current_frequency * (i + 1) for i in range(self.num_harmonics)
            ])
            
            # Apply frequency-based filtering
            filter_mask = self._calculate_filter_mask(harmonic_freqs)
            filtered_amps *= filter_mask
        
        return filtered_amps
    
    def _calculate_filter_mask(self, frequencies):
        """Calculate filter mask based on frequencies and filter parameters."""
        mask = np.ones_like(frequencies)
        
        if self.filter_type == "lowpass":
            # Low-pass filter: attenuate frequencies above cutoff
            cutoff = self.filter_cutoff_high
            for i, freq in enumerate(frequencies):
                if freq > cutoff:
                    # Calculate attenuation based on rolloff
                    octaves_above = np.log2(freq / cutoff)
                    attenuation_db = -self.filter_rolloff * octaves_above
                    attenuation_linear = 10 ** (attenuation_db / 20)
                    mask[i] = max(0.0, attenuation_linear)
        
        elif self.filter_type == "highpass":
            # High-pass filter: attenuate frequencies below cutoff
            cutoff = self.filter_cutoff_low
            for i, freq in enumerate(frequencies):
                if freq < cutoff:
                    # Calculate attenuation based on rolloff
                    octaves_below = np.log2(cutoff / freq)
                    attenuation_db = -self.filter_rolloff * octaves_below
                    attenuation_linear = 10 ** (attenuation_db / 20)
                    mask[i] = max(0.0, attenuation_linear)
        
        elif self.filter_type == "bandpass":
            # Band-pass filter: keep frequencies between low and high cutoffs
            low_cutoff = self.filter_cutoff_low
            high_cutoff = self.filter_cutoff_high
            for i, freq in enumerate(frequencies):
                if freq < low_cutoff:
                    octaves_below = np.log2(low_cutoff / freq)
                    attenuation_db = -self.filter_rolloff * octaves_below
                    attenuation_linear = 10 ** (attenuation_db / 20)
                    mask[i] = max(0.0, attenuation_linear)
                elif freq > high_cutoff:
                    octaves_above = np.log2(freq / high_cutoff)
                    attenuation_db = -self.filter_rolloff * octaves_above
                    attenuation_linear = 10 ** (attenuation_db / 20)
                    mask[i] = max(0.0, attenuation_linear)
        
        elif self.filter_type == "notch":
            # Notch filter: attenuate frequencies around a center frequency
            center_freq = (self.filter_cutoff_low + self.filter_cutoff_high) / 2
            bandwidth = self.filter_cutoff_high - self.filter_cutoff_low
            for i, freq in enumerate(frequencies):
                # Distance from center frequency
                freq_distance = abs(freq - center_freq)
                if freq_distance < bandwidth / 2:
                    # Attenuate based on distance from center
                    attenuation_factor = 1 - (freq_distance / (bandwidth / 2))
                    attenuation_db = -self.filter_rolloff * attenuation_factor
                    attenuation_linear = 10 ** (attenuation_db / 20)
                    mask[i] = max(0.0, attenuation_linear)
        
        return mask
    
    def set_frequency_filter(self, enabled: bool, filter_type: str = None, 
                           cutoff_low: float = None, cutoff_high: float = None, 
                           rolloff: int = None):
        """Configure frequency filtering parameters."""
        self.frequency_filter_enabled = enabled
        
        if filter_type is not None:
            valid_types = ["lowpass", "highpass", "bandpass", "notch", "custom"]
            if filter_type in valid_types:
                self.filter_type = filter_type
            else:
                raise ValueError(f"Filter type must be one of: {valid_types}")
        
        if cutoff_low is not None:
            self.filter_cutoff_low = cutoff_low
        
        if cutoff_high is not None:
            self.filter_cutoff_high = cutoff_high
        
        if rolloff is not None:
            if rolloff in [6, 12, 18, 24]:
                self.filter_rolloff = rolloff
            else:
                raise ValueError("Rolloff must be 6, 12, 18, or 24 dB/octave")
    
    def set_custom_harmonic_mask(self, mask: np.ndarray):
        """Set custom per-harmonic amplitude multipliers (0.0 = mute, 1.0 = full)."""
        if len(mask) == self.num_harmonics:
            self.custom_harmonic_mask = np.clip(mask, 0.0, 1.0)
        else:
            raise ValueError(f"Mask must have {self.num_harmonics} elements")
    
    def get_frequency_filter_info(self):
        """Get current frequency filter configuration."""
        return {
            "enabled": self.frequency_filter_enabled,
            "type": self.filter_type,
            "cutoff_low": self.filter_cutoff_low,
            "cutoff_high": self.filter_cutoff_high,
            "rolloff": self.filter_rolloff,
            "custom_mask": self.custom_harmonic_mask.copy()
        }

    def get_position(self):
        """Get current smoothed position."""
        return self.current_position

    def get_last_chunk(self):
        """Get the last generated chunk."""
        return self.last_chunk
    
    def get_harmonic_amplitudes(self):
        """Get current harmonic amplitudes."""
        return self.current_harmonic_amplitudes.copy()
    
    def set_smoothing_factor(self, factor: float):
        """Adjust smoothing factor for all parameters (0.9-0.99)."""
        self.smoothing_factor = np.clip(factor, 0.0, 0.99)
        self.amplitude_smoothing = self.smoothing_factor
        self.position_smoothing = self.smoothing_factor
        self.phase_offset_smoothing = self.smoothing_factor
        self.harmonic_smoothing = self.smoothing_factor
    
    def set_individual_smoothing(self, amplitude: float = None, 
                                position: float = None, 
                                phase_offset: float = None,
                                harmonic: float = None):
        """Set individual smoothing factors for different parameters."""
        if amplitude is not None:
            self.amplitude_smoothing = np.clip(amplitude, 0.0, 0.99)
        if position is not None:
            self.position_smoothing = np.clip(position, 0.0, 0.99)
        if phase_offset is not None:
            self.phase_offset_smoothing = np.clip(phase_offset, 0.0, 0.99)
        if harmonic is not None:
            self.harmonic_smoothing = np.clip(harmonic, 0.0, 0.99)
    
    def set_harmonic_decay(self, decay: float):
        """Set the decay factor for harmonics."""
        self.harmonic_decay = decay
        # Update base amplitudes
        base_amplitudes = np.array([
            decay ** i for i in range(self.num_harmonics)
        ])
        # Apply current modulation pattern if orientation has been set
        # For now, just update the base
        self.target_harmonic_amplitudes = base_amplitudes
