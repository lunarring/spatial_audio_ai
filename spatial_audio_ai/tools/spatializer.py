import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
import soundfile as sf
import time
from spatial_audio_ai.tools.numpysocket import NumpySocket
import os
import random
from spatial_audio_ai.tools.client import SoundNetworkStreamer
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE
import uuid

sd.default.blocksize = BLOCKSIZE
CHUNKSIZE = BLOCKSIZE*4

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
        smoothing_factor: float = 0.95
    ):
        """
        Real-time sine wave generator with smooth parameter transitions.
        Args:
            frequency: Frequency in Hz.
            amplitude: Amplitude (0.0 to 1.0).
            phase_offset: Phase offset in radians.
            position: Position in 2D space.
            smoothing_factor: Smoothing factor for parameter changes (0.9-0.99).
        """
        # Target values (what we want to reach)
        self.target_frequency = frequency
        self.target_amplitude = amplitude
        self.target_phase_offset = phase_offset
        self.target_position = position.copy()
        
        # Current values (what we're actually using, smoothed)
        self.current_frequency = frequency
        self.current_amplitude = amplitude
        self.current_phase_offset = phase_offset
        self.current_position = position.copy()
        
        # Phase state management
        self.current_phase = 0.0
        self.last_frequency = frequency  # Track for phase continuity
        
        # Smoothing configuration
        self.smoothing_factor = smoothing_factor
        self.amplitude_smoothing = smoothing_factor
        self.position_smoothing = smoothing_factor
        self.phase_offset_smoothing = smoothing_factor
        
        # State tracking
        self.last_chunk = np.zeros(CHUNKSIZE, dtype=float)
        self._id = uuid.uuid4()

    def query(self, tick: int) -> SoundMessage:
        """Generate next chunk with smooth parameter transitions."""
        chunk_duration = CHUNKSIZE / SAMPLING_RATE
        t = np.linspace(0, chunk_duration, CHUNKSIZE, endpoint=False)
        
        # Handle frequency changes with phase continuity
        if abs(self.target_frequency - self.last_frequency) > 0.1:
            # Frequency changed significantly - maintain phase continuity
            # The phase should continue smoothly from where it was
            self.current_frequency = self.target_frequency
            self.last_frequency = self.target_frequency
        else:
            # Small or no frequency change
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
        
        # Generate sine wave chunk with smoothed parameters
        # Use instantaneous frequency to avoid artifacts
        instantaneous_phase = (
            2 * np.pi * self.current_frequency * t + 
            self.current_phase + 
            self.current_phase_offset
        )
        
        sound_chunk = self.current_amplitude * np.sin(instantaneous_phase)
        
        # Update phase for next chunk (maintain continuity)
        self.current_phase += 2 * np.pi * self.current_frequency * chunk_duration
        self.current_phase = self.current_phase % (2 * np.pi)
        
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

    def get_position(self):
        """Get current smoothed position."""
        return self.current_position

    def get_last_chunk(self):
        """Get the last generated chunk."""
        return self.last_chunk
    
    def set_smoothing_factor(self, factor: float):
        """Adjust smoothing factor for all parameters (0.9-0.99)."""
        self.smoothing_factor = np.clip(factor, 0.0, 0.99)
        self.amplitude_smoothing = self.smoothing_factor
        self.position_smoothing = self.smoothing_factor
        self.phase_offset_smoothing = self.smoothing_factor
    
    def set_individual_smoothing(self, amplitude: float = None, 
                                position: float = None, 
                                phase_offset: float = None):
        """Set individual smoothing factors for different parameters."""
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
        
        # Orientation mapping parameters
        self.orientation_mapping_mode = "basis_fourier"  # Available modes: quaternion_simple, quaternion_complex, basis_fourier, basis_chebyshev, basis_legendre, basis_wavelets, basis_radial

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
        
        # Smooth harmonic amplitudes
        self.current_harmonic_amplitudes = (
            self.harmonic_smoothing * self.current_harmonic_amplitudes + 
            (1 - self.harmonic_smoothing) * self.target_harmonic_amplitudes
        )
        
        # Generate multi-harmonic sound chunk
        sound_chunk = np.zeros(CHUNKSIZE, dtype=float)
        
        for i in range(self.num_harmonics):
            harmonic_freq = self.current_frequency * (i + 1)  # Fundamental, 2nd, 3rd harmonics...
            harmonic_amplitude = self.current_amplitude * self.current_harmonic_amplitudes[i]
            
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

    def set_orientation(self, orientation: np.ndarray):
        """
        Update harmonic amplitudes based on orientation quaternion.
        Args:
            orientation: Quaternion as [x, y, z, w] array
        """
        if self.orientation_mapping_mode == "quaternion_simple":
            self._map_orientation_simple(orientation)
        elif self.orientation_mapping_mode == "quaternion_complex":
            self._map_orientation_complex(orientation)
        elif self.orientation_mapping_mode == "basis_fourier":
            self._map_orientation_basis_fourier(orientation)
        elif self.orientation_mapping_mode == "basis_chebyshev":
            self._map_orientation_basis_chebyshev(orientation)
        elif self.orientation_mapping_mode == "basis_legendre":
            self._map_orientation_basis_legendre(orientation)
        elif self.orientation_mapping_mode == "basis_wavelets":
            self._map_orientation_basis_wavelets(orientation)
        elif self.orientation_mapping_mode == "basis_radial":
            self._map_orientation_basis_radial(orientation)

    def _map_orientation_simple(self, orientation: np.ndarray):
        """
        Simple orientation mapping: use quaternion components directly.
        """
        # Normalize quaternion if needed
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Map quaternion components to harmonic amplitudes
        # Each component influences different harmonics
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        # Create modulation based on quaternion components
        modulation = np.zeros(self.num_harmonics)
        
        # Use different quaternion components for different harmonics
        for i in range(self.num_harmonics):
            if i % 4 == 0:
                modulation[i] = abs(w) * 2.0  # Fundamental controlled by w
            elif i % 4 == 1:
                modulation[i] = abs(x) * 2.0  # 2nd harmonic by x
            elif i % 4 == 2:
                modulation[i] = abs(y) * 2.0  # 3rd harmonic by y
            else:
                modulation[i] = abs(z) * 2.0  # 4th harmonic by z
        
        # Apply modulation to base amplitudes
        self.target_harmonic_amplitudes = base_amplitudes * (0.1 + modulation)
        
        # Normalize to prevent clipping
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_complex(self, orientation: np.ndarray):
        """
        Complex orientation mapping: convert to Euler angles and use trigonometric functions.
        """
        # Normalize quaternion
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        # Create complex modulation patterns based on Euler angles
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            # Use different combinations of Euler angles for each harmonic
            angle_factor = (i + 1) * 0.5  # Scale factor for each harmonic
            
            # Combine roll, pitch, yaw with different weightings
            roll_contrib = np.cos(roll * angle_factor) ** 2
            pitch_contrib = np.sin(pitch * angle_factor) ** 2
            yaw_contrib = np.cos(yaw * angle_factor + np.pi/4) ** 2
            
            # Weight contributions differently for each harmonic
            if i % 3 == 0:
                modulation[i] = 0.7 * roll_contrib + 0.2 * pitch_contrib + 0.1 * yaw_contrib
            elif i % 3 == 1:
                modulation[i] = 0.2 * roll_contrib + 0.7 * pitch_contrib + 0.1 * yaw_contrib
            else:
                modulation[i] = 0.1 * roll_contrib + 0.2 * pitch_contrib + 0.7 * yaw_contrib
        
        # Apply modulation to base amplitudes
        self.target_harmonic_amplitudes = base_amplitudes * (0.2 + modulation * 1.5)
        
        # Normalize to prevent clipping
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_basis_fourier(self, orientation: np.ndarray):
        """
        Fourier basis functions: Use sine/cosine combinations for smooth, periodic variations.
        """
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Convert to Euler angles for more interpretable parameters
        roll, pitch, yaw = self._quat_to_euler(quat)
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        # Create Fourier basis modulation
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            # Use different frequency combinations for each harmonic
            freq_scale = (i + 1) * 0.5
            
            # Fourier series expansion with multiple frequencies
            fourier_term = (
                0.5 * np.cos(roll * freq_scale) +
                0.3 * np.sin(pitch * freq_scale * 2) +
                0.2 * np.cos(yaw * freq_scale * 3) +
                0.1 * np.sin(roll * pitch * freq_scale) +
                0.1 * np.cos(pitch * yaw * freq_scale)
            )
            
            # Add harmonics of the Fourier expansion
            fourier_term += (
                0.1 * np.cos(roll * freq_scale * 2) +
                0.05 * np.sin(pitch * freq_scale * 4)
            )
            
            modulation[i] = fourier_term
        
        # Normalize modulation to [0, 1] range
        modulation = (modulation - np.min(modulation)) / (np.max(modulation) - np.min(modulation))
        
        # Apply modulation with scaling
        self.target_harmonic_amplitudes = base_amplitudes * (0.1 + modulation * 1.5)
        
        # Final normalization
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_basis_chebyshev(self, orientation: np.ndarray):
        """
        Chebyshev polynomial basis: Creates sharp, dramatic changes in harmonic content.
        """
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Convert quaternion components to [-1, 1] range for Chebyshev polynomials
        params = np.array([x, y, z, w])
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            # Use different Chebyshev polynomials for each harmonic
            degree = (i % 6) + 1  # Use degrees 1-6
            param_idx = i % 4     # Cycle through quaternion components
            t = params[param_idx] # Parameter in [-1, 1]
            
            # Chebyshev polynomials T_n(t)
            if degree == 1:
                cheby_val = t
            elif degree == 2:
                cheby_val = 2*t**2 - 1
            elif degree == 3:
                cheby_val = 4*t**3 - 3*t
            elif degree == 4:
                cheby_val = 8*t**4 - 8*t**2 + 1
            elif degree == 5:
                cheby_val = 16*t**5 - 20*t**3 + 5*t
            else:  # degree == 6
                cheby_val = 32*t**6 - 48*t**4 + 18*t**2 - 1
            
            # Mix with other parameters for complexity
            mix_param = params[(param_idx + 1) % 4]
            mixed_val = 0.7 * cheby_val + 0.3 * mix_param
            
            modulation[i] = mixed_val
        
        # Normalize to [0, 1]
        modulation = (modulation + 1) / 2  # Shift from [-1,1] to [0,1]
        
        # Apply with strong contrast
        self.target_harmonic_amplitudes = base_amplitudes * (0.05 + modulation * 2.0)
        
        # Normalize
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_basis_legendre(self, orientation: np.ndarray):
        """
        Legendre polynomial basis: Smooth, orthogonal basis functions.
        """
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Convert to Euler for interpretable parameters
        roll, pitch, yaw = self._quat_to_euler(quat)
        
        # Normalize angles to [-1, 1] for Legendre polynomials
        angles = np.array([roll, pitch, yaw]) / np.pi
        angles = np.clip(angles, -1, 1)
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            degree = i % 6  # Use degrees 0-5
            angle_idx = i % 3  # Cycle through angles
            t = angles[angle_idx]
            
            # Legendre polynomials P_n(t)
            if degree == 0:
                legendre_val = 1
            elif degree == 1:
                legendre_val = t
            elif degree == 2:
                legendre_val = 0.5 * (3*t**2 - 1)
            elif degree == 3:
                legendre_val = 0.5 * (5*t**3 - 3*t)
            elif degree == 4:
                legendre_val = 0.125 * (35*t**4 - 30*t**2 + 3)
            else:  # degree == 5
                legendre_val = 0.125 * (63*t**5 - 70*t**3 + 15*t)
            
            # Add cross-terms for interaction
            cross_term = angles[(angle_idx + 1) % 3] * angles[(angle_idx + 2) % 3]
            combined_val = 0.8 * legendre_val + 0.2 * cross_term
            
            modulation[i] = combined_val
        
        # Normalize to [0, 1]
        modulation = (modulation - np.min(modulation)) / (np.max(modulation) - np.min(modulation))
        
        # Apply modulation
        self.target_harmonic_amplitudes = base_amplitudes * (0.2 + modulation * 1.3)
        
        # Normalize
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_basis_wavelets(self, orientation: np.ndarray):
        """
        Wavelet-like basis: Sharp transitions and localized features.
        """
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Use quaternion components directly
        params = np.array([x, y, z, w])
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            param_idx = i % 4
            t = params[param_idx] * 10  # Scale for wavelet frequency
            
            # Different wavelet-like functions
            wavelet_type = i % 4
            
            if wavelet_type == 0:
                # Morlet-like wavelet
                wavelet_val = np.cos(t) * np.exp(-t**2 / 8)
            elif wavelet_type == 1:
                # Mexican hat wavelet
                wavelet_val = (1 - t**2 / 2) * np.exp(-t**2 / 4)
            elif wavelet_type == 2:
                # Daubechies-like
                if abs(t) < 2:
                    wavelet_val = np.sin(t * np.pi) * (1 - abs(t) / 2)
                else:
                    wavelet_val = 0
            else:
                # Custom sharp wavelet
                wavelet_val = np.sin(t * 2) * np.exp(-abs(t) / 3) * np.cos(t / 2)
            
            # Add interaction with other parameters
            interaction = np.sum(params) * 0.1
            modulation[i] = wavelet_val + interaction
        
        # Normalize to [0, 1]
        modulation = (modulation - np.min(modulation)) / (np.max(modulation) - np.min(modulation))
        
        # Apply with emphasis on sharp features
        self.target_harmonic_amplitudes = base_amplitudes * (0.1 + modulation * 1.8)
        
        # Normalize
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _map_orientation_basis_radial(self, orientation: np.ndarray):
        """
        Radial basis functions: Distance-based smooth interpolation.
        """
        quat = orientation / np.linalg.norm(orientation)
        x, y, z, w = quat
        
        # Define some reference points in quaternion space
        reference_quats = np.array([
            [1, 0, 0, 0],    # Reference 1
            [0, 1, 0, 0],    # Reference 2
            [0, 0, 1, 0],    # Reference 3
            [0, 0, 0, 1],    # Reference 4
            [0.707, 0.707, 0, 0],  # Reference 5
            [0.5, 0.5, 0.5, 0.5],  # Reference 6
        ])
        
        # Base harmonic amplitudes
        base_amplitudes = np.array([
            self.harmonic_decay ** i for i in range(self.num_harmonics)
        ])
        
        modulation = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            # Use different reference points for each harmonic
            ref_idx = i % len(reference_quats)
            ref_quat = reference_quats[ref_idx]
            
            # Calculate distance to reference quaternion
            # Use quaternion distance: d = 1 - |q1 · q2|
            dot_product = np.abs(np.dot(quat, ref_quat))
            distance = 1 - dot_product
            
            # Radial basis function (Gaussian)
            sigma = 0.5  # Width parameter
            rbf_val = np.exp(-distance**2 / (2 * sigma**2))
            
            # Add some mixing with other reference points
            if i + 1 < len(reference_quats):
                ref2_quat = reference_quats[(ref_idx + 1) % len(reference_quats)]
                dot2 = np.abs(np.dot(quat, ref2_quat))
                distance2 = 1 - dot2
                rbf_val2 = np.exp(-distance2**2 / (2 * sigma**2))
                rbf_val = 0.7 * rbf_val + 0.3 * rbf_val2
            
            modulation[i] = rbf_val
        
        # Apply modulation
        self.target_harmonic_amplitudes = base_amplitudes * (0.3 + modulation * 1.2)
        
        # Normalize
        max_amp = np.max(self.target_harmonic_amplitudes)
        if max_amp > 1.0:
            self.target_harmonic_amplitudes /= max_amp

    def _quat_to_euler(self, quat):
        """Helper function to convert quaternion to Euler angles."""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

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
    
    def set_orientation_mapping_mode(self, mode: str):
        """Set the orientation mapping mode."""
        valid_modes = [
            "quaternion_simple", "quaternion_complex", 
            "basis_fourier", "basis_chebyshev", "basis_legendre", 
            "basis_wavelets", "basis_radial"
        ]
        if mode in valid_modes:
            self.orientation_mapping_mode = mode
        else:
            raise ValueError(f"Mode must be one of: {valid_modes}")


class Spatializer:
    def __init__(self,
                 subwoofer_last_channel_auto_mode=True,
                 process_function=None
    ):
        self.speaker_positions = np.asarray([
            (-4.80, 4.7), # 1
            (-3.0, 4.8),  # 2
            (-0.0, 4.8), # 3
            (3.0, 4.8), # 4
            (4.8, 4.7), # 5
            (4.8, 0.4), # 6
            (4.8, -4.6), # 7
            (2.7, -4.6), # 8
            (-0.0, -4.6), # 9
            (-2.7, -4.6), # 10
            (-4.8, -4.6), # 11
            (-4.8, 0.4), # 12
        ])
        self.subwoofer_last_channel_auto_mode = subwoofer_last_channel_auto_mode
        self.attenuation_scaler = 1.0
        self.process = process_function if process_function is not None else self.process_simple

    def process_simple(self, sm: SoundMessage):
        """
        Process a SoundMessage by calculating the attenuation for each speaker
        based on the distance from the sound's position to each speaker position.
        The sound is then attenuated accordingly and optionally processed for
        subwoofer output.
        """
        distances = np.linalg.norm((self.speaker_positions - sm.position), axis=1)
        attenuation = 1 / (1 + self.attenuation_scaler * distances)
        buffer = np.expand_dims(attenuation, 1) * sm.sound
        
        if self.subwoofer_last_channel_auto_mode:
            buffer = np.append(buffer, np.expand_dims(np.mean(buffer, axis=0), axis=0), axis=0)
        return buffer
    
    def process_only1(self, sm: SoundMessage):
        def process_only1(self, sm: SoundMessage):
            """
            Process a SoundMessage by selecting only the speaker with the highest
            attenuation to play the sound. 
            """
        distances = np.linalg.norm((self.speaker_positions - sm.position), axis=1)
        attenuation = 1 / (1 + self.attenuation_scaler * distances)
        idx_survive = np.argmax(attenuation)
        attenuation_flat = np.zeros_like(attenuation)
        attenuation_flat[idx_survive] = 1
        attenuation = attenuation_flat
        buffer = np.expand_dims(attenuation, 1) * sm.sound
        
        if self.subwoofer_last_channel_auto_mode:
            buffer = np.append(buffer, np.expand_dims(np.mean(buffer, axis=0), axis=0), axis=0)
        return buffer


class Scene:
    def __init__(
            self,
            spatializer: Spatializer, 
    ):
        self.sound_objects = []
        self.tick = 0
        self.spatializer = spatializer
        self.sleep_time = 0
        self.volume = 0.4

    def register(self, so: SoundObjectBase):
        self.sound_objects.append(so)

    def run(self):
        while True:
            sound_messages = []
            kill_idx = []
            for i, so in enumerate(self.sound_objects):
                try: 
                    sound_messages.append(so.query(self.tick))
                except SoundObjectTerminatedException:
                    kill_idx.append(i)
                except Exception as e:
                    print(f"Misbehaved soundobject! Going to die, because {e}")
                    kill_idx.append(i)
           
            self.sound_objects = [self.sound_objects[i] for i in range(len(self.sound_objects)) if not i in kill_idx]
            if len(self.sound_objects) == 0:
                return
                    
            speaker_sounds = np.array([self.spatializer.process(sm) * sm.volume for sm in sound_messages]) 
            buffer = np.sum(speaker_sounds, axis=0)
            buffer *= self.volume 
            
            self.tick += 1
            yield buffer
            
            time.sleep(self.sleep_time)
        
                
if __name__ == "__main__":

    # pool based sound scape playback
    name_space = "ambient"
    p_inject = 0.05
    box_size = 25
    # dir_scan = f'/home/lugo/git/spatial_audio_ai/soundpools/{name_space}/'
    dir_scan = '/home/lugo/audio/export/machine/'

    # Get a list of all .wav files in dir_scn
    wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
    sound = sf.read(f"{dir_scan}{wav_files[0]}")[0]
    # Set initial circular movement parameters
    init_radius = np.random.uniform(2, box_size/2)
    init_speed = np.random.uniform(0.2, 1.0)
    init_angle = np.random.uniform(0, 2*np.pi)
    init_direction = random.choice([-1, 1])
    so = SO_PlaybackCircularMove(
        sound,
        radius=init_radius,
        speed=init_speed,
        initial_angle=init_angle,
        direction=init_direction,
        center=np.zeros(2, dtype=float),
        loop=True
    )
    print(f"Initial circular moving sound: {wav_files[0]} at radius: {init_radius:.2f}, speed: {init_speed:.2f}, angle: {init_angle:.2f}, direction: {init_direction}")

    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.volume = 0.3
    scene.register(so)

    sound_streamer = SoundNetworkStreamer()
    
    # Implement precise real-time timing
    chunk_duration = CHUNKSIZE / SAMPLING_RATE
    start_time = time.perf_counter()
    
    for j, chunk in enumerate(scene.run()):

        if np.random.rand() < p_inject:
            wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
            random_file = random.choice(wav_files)
            sound = sf.read(f"{dir_scan}{random_file}")[0]
            # Set random radius, speed, angle, direction for each injected sound
            radius = np.random.uniform(2, box_size/2)
            speed = np.random.uniform(0.2, 1.0)
            angle = np.random.uniform(0, 2*np.pi)
            direction = random.choice([-1, 1])
            scene.register(SO_PlaybackCircularMove(
                sound,
                radius=radius,
                speed=speed,
                initial_angle=angle,
                direction=direction,
                center=np.zeros(2, dtype=float),
                loop=True
            ))
            print(f"Injected circular moving sound: {random_file} at radius: {radius:.2f}, speed: {speed:.2f}, angle: {angle:.2f}, direction: {direction}")
        
        chunk = np.clip(chunk, -1, 1)
        sound_streamer.send(chunk)
        
        # Schedule next chunk send time (precise timing)
        next_time = start_time + (j + 1) * chunk_duration
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
    
    #%%#
    # moving song
    if False:
        import lunar_tools as lt
        receiver = lt.OSCReceiver('10.40.50.9')
        sound_streamer = SoundNetworkStreamer()
        
        raw_sound1 = sf.read("/home/lugo/Downloads/song.wav")
        sound11 = np.sin(0.5*1e-2*np.linspace(0,44100*25*5,44100*25*5))
        sound12 = np.sin(0.7*1e-2*np.linspace(0,44100*25*5,44100*25*5))
        sound1 = (sound11 + sound12)*0.4
        sound1 = np.tile(np.expand_dims(sound1,axis=1), (1,2))
        raw_sound1 = (sound1, raw_sound1[1])
        
        raw_sound2 = sf.read("/home/lugo/Downloads/song.wav")
        sound21 = np.sin(4.8*1e-2*np.linspace(0,44100*25*5,44100*25*5))
        sound22 = np.sin(4.9*1e-2*np.linspace(0,44100*25*5,44100*25*5))
        sound2 = (sound21 + sound22)*0.2
        sound2 = np.tile(np.expand_dims(sound2,axis=1), (1,2))
        raw_sound2 = (sound2, raw_sound2[1])        
        
        so_a = SO_Playback(raw_sound1[0][:,0])
        so_b = SO_Playback(raw_sound2[0][:,0])
        
        spatializer = Spatializer()
        scene = Scene(spatializer)
        scene.register(so_a)
        scene.register(so_b)
        radius = 6
        for j, chunk in enumerate(scene.run()):
            angle = - receiver.get_last_value("/speaker_angle")
            angle -= 0.1
            x = radius * np.sin(angle)
            y = radius * np.cos(angle)
            position = np.array([x, y])
            so_a.set_position(position)

            angle_offet = np.pi * 180 / 180
            x = radius * np.sin(angle_offet)
            y = radius * np.cos(angle_offet)
            position2 = np.array([x, y])            
            so_b.set_position(position2)
            
            print(f"{x} {y} {angle}")
            sound_streamer.send(chunk)
            
            # Use proper real-time timing 
            chunk_duration = CHUNKSIZE / SAMPLING_RATE
            if j == 0:
                timing_start = time.perf_counter()
            next_time = timing_start + (j + 1) * chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # simple placement of two objects
    if False:

        
        # sound_a = generate_sine_tone(400, 6)
        # sound_b = generate_sine_tone(600, 6)
    
        # so_a = SO_Playback(sf.read("/home/lugo/Downloads/song.wav")[0][:,0])
        
        sound_a = sf.read("/home/lugo/audio/export/talking1.wav")[0]
        sound_b = sf.read("/home/lugo/audio/export/talking2.wav")[0]
    
        so_a = SO_Playback(sound_a)
        so_b = SO_Playback(sound_b)
    
        spatializer = Spatializer()
        scene = Scene(spatializer)
        scene.volume = 0.2
        scene.register(so_a)
        scene.register(so_b)
    
        sound_streamer = SoundNetworkStreamer()
        list_chunks = []
        for j, chunk in enumerate(scene.run()):
            
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            
            list_chunks.append(chunk)
            if j == 5:
                scene.register(SO_Playback(sound_a, position=np.array([-3., -3.1])))
            if j == 8:
                scene.register(SO_Playback(sound_b, position=np.array([3., 3.])))
            print(f"sent chunk {j}")
            
            # Use precise real-time timing - calculated per chunk
            chunk_duration = CHUNKSIZE / SAMPLING_RATE
            if j == 0:
                start_time = time.perf_counter()
            next_time = start_time + (j + 1) * chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)