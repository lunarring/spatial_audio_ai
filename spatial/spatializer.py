import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from tools import AudioHandler

sd.default.blocksize = 1024


@dataclass
class AudioSample:
    audio_buffer: np.ndarray
    onset: float  # Onset time in seconds
    position: Tuple[float, float]  # (x, y) position in meters
    duration: float = 0.0  # Duration in seconds, to be computed

    def __post_init__(self):
        if self.audio_buffer is not None:
            self.duration = len(self.audio_buffer) / 44100  # Assuming default sample rate
        else:
            self.duration = 0.0


class SoundSpatializer:
    def __init__(
        self,
        sample_rate: int = 44100,
        listener_position: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Initialize the SoundSpatializer.

        :param sample_rate: Sampling rate for audio processing and playback.
        :param speed_of_sound: Speed of sound in meters per second.
        :param listener_position: Tuple representing the (x, y) position of the listener in meters.
        """
        self.sample_rate = sample_rate
        self.speed_of_sound = 343.0
        self.listener_position = np.array(listener_position, dtype=float)
        self.speaker_positions: List[np.ndarray] = []  # List of np.array([x, y]) in meters
        self.audio_samples: List[AudioSample] = []  # List of AudioSample instances
        self.audio_output: np.ndarray = None  # Spatialized audio buffer
        self.verbose = False
        self.apply_delays = True
        self.attenuation_scaler = 1.0

    def define_speakers(self, speaker_positions: List[Tuple[float, float]]):
        """
        Define the positions of the speakers in 2D space and compute spatial parameters for all samples.

        :param speaker_positions: List of tuples representing (x, y) positions in meters.
                                  Example for stereo: [(-1, 0), (1, 0)]
        """
        self.speaker_positions = [np.array(pos, dtype=float) for pos in speaker_positions]
        if self.verbose:
            print(f"Defined {len(speaker_positions)} speaker(s) at positions: {speaker_positions} meters")

        # Compute spatial parameters for all existing audio samples
        self.compute_spatial_parameters()

    def add_audio_sample(self, audio_buffer: np.ndarray, position: Tuple[float, float], onset: float):
        """
        Add a new audio sample with its audio data, spatial position, and onset time.

        :param audio_buffer: NumPy array containing the audio data.
        :param position: Tuple representing the (x, y) position in meters.
        :param onset: Onset time in seconds.
        :return: Index of the added audio sample.
        """
        sample = AudioSample(audio_buffer=audio_buffer, position=position, onset=onset)
        self.audio_samples.append(sample)
        if self.verbose:
            print(f"Added new AudioSample at position: {sample.position} meters with onset {sample.onset} seconds")

        # Compute spatial parameters for the new sample
        if self.speaker_positions:
            self.compute_spatial_parameters_for_sample(len(self.audio_samples) - 1)
        else:
            if self.verbose:
                print("No speakers defined yet. Spatial parameters will be computed once speakers are defined.")

        return len(self.audio_samples) - 1  # Return the index of the new sample

    def remove_audio_sample(self, index: int):
        """
        Remove an audio sample by its index.

        :param index: Integer index of the audio sample to remove.
        """
        if 0 <= index < len(self.audio_samples):
            removed = self.audio_samples.pop(index)
            if self.verbose:
                print(f"Removed AudioSample at position: {removed.position} meters with onset {removed.onset} seconds")
        else:
            raise IndexError("Audio sample index out of range.")

    def compute_spatial_parameters(self):
        """
        Compute distances and angles from each audio sample to each speaker.
        """
        for sample_idx, sample in enumerate(self.audio_samples):
            sample.distances = []
            sample.angles = []
            for sp_idx, speaker_pos in enumerate(self.speaker_positions):
                vector = speaker_pos - sample.position  # Vector from sample to speaker
                distance = np.linalg.norm(vector)  # Euclidean distance
                angle_rad = np.arctan2(vector[1], vector[0])  # Angle in radians
                angle_deg = np.degrees(angle_rad)  # Convert to degrees

                sample.distances.append(distance)
                sample.angles.append(angle_deg)

                if self.verbose:
                    print(f"AudioSample {sample_idx + 1} to Speaker {sp_idx + 1}: "
                          f"Distance={distance:.2f}m, Angle={angle_deg:.2f}°")

    def compute_spatial_parameters_for_sample(self, sample_idx: int):
        """
        Compute distances and angles from a specific audio sample to each speaker.

        :param sample_idx: Index of the audio sample.
        """
        sample = self.audio_samples[sample_idx]
        sample.distances = []
        sample.angles = []
        for sp_idx, speaker_pos in enumerate(self.speaker_positions):
            vector = speaker_pos - sample.position  # Vector from sample to speaker
            distance = np.linalg.norm(vector)  # Euclidean distance
            angle_rad = np.arctan2(vector[1], vector[0])  # Angle in radians
            angle_deg = np.degrees(angle_rad)  # Convert to degrees

            sample.distances.append(distance)
            sample.angles.append(angle_deg)

            if self.verbose:
                print(f"AudioSample {sample_idx + 1} to Speaker {sp_idx + 1}: "
                      f"Distance={distance:.2f}m, Angle={angle_deg:.2f}°")

    def spatialize(self):
        """
        Spatialize all audio samples by applying attenuation and temporal delays based on their positions
        relative to each speaker, then mix them into speaker-specific buffers.
        """
        if not self.speaker_positions:
            raise ValueError("Speaker positions must be defined before spatializing audio.")

        if not self.audio_samples:
            raise ValueError("No audio samples to spatialize.")

        num_speakers = len(self.speaker_positions)
        num_samples = len(self.audio_samples)

        # Determine the total duration of the output (max onset + duration of samples + max delay)
        max_delay = 0.0
        for sample in self.audio_samples:
            # Compute maximum possible delay based on distance to any speaker
            if hasattr(sample, 'distances'):
                max_distance = max(sample.distances)
                delay = max_distance / self.speed_of_sound
                if delay > max_delay:
                    max_delay = delay
            else:
                # If spatial parameters haven't been computed yet
                vector = self.listener_position - sample.position
                distance = np.linalg.norm(vector)
                delay = distance / self.speed_of_sound
                if delay > max_delay:
                    max_delay = delay

        total_duration = max(sample.onset + sample.duration for sample in self.audio_samples) + max_delay
        total_length = int(total_duration * self.sample_rate)
        if self.verbose:
            print(f"Total duration of the output buffer: {total_duration:.2f} seconds ({total_length} samples)")

        # Initialize empty buffers for each speaker
        speaker_buffers = [np.zeros(total_length) for _ in range(num_speakers)]

        # Spatialization process: apply attenuation and delays, then mix into speaker buffers
        for sample_idx, sample in enumerate(self.audio_samples):
            for sp_idx, speaker_pos in enumerate(self.speaker_positions):
                distance = sample.distances[sp_idx]
                angle = sample.angles[sp_idx]


                # Calculate delay in seconds and then in samples
                if self.apply_delays:
                    delay_seconds = distance / self.speed_of_sound
                    delay_samples = int(delay_seconds * self.sample_rate)
                else:
                    delay_samples = 0

                # Calculate onset in samples
                onset_samples = int(sample.onset * self.sample_rate)

                # Total delay including onset
                total_delay_samples = onset_samples + delay_samples


                # Calculate attenuation based on the distance
                attenuation = 1 / (1 + self.attenuation_scaler * distance)
                
                # Apply attenuation to the audio buffer
                attenuated_buffer = sample.audio_buffer * attenuation

                # Insert the attenuated and delayed buffer into the speaker's buffer
                end_idx = total_delay_samples + len(attenuated_buffer)
                if end_idx > total_length:
                    # Truncate the buffer if it exceeds the total length
                    attenuated_buffer = attenuated_buffer[: total_length - total_delay_samples]
                    end_idx = total_length

                speaker_buffers[sp_idx][total_delay_samples:end_idx] += attenuated_buffer
                if self.verbose:
                    print(f"Spatialized AudioSample {sample_idx + 1} to Speaker {sp_idx + 1}: "
                            f"Attenuation={attenuation:.4f}, Delay={delay_samples} samples, Onset={onset_samples} samples")

        # Combine speaker buffers into a stereo (or multi-channel) audio output
        self.audio_output = np.stack(speaker_buffers, axis=-1)
        if self.verbose:
            print("Spatialization complete.")

    def play_audio(self):
        """
        Play the spatialized audio using sounddevice.
        """
        if self.audio_output is None:
            raise ValueError("Audio must be spatialized before playback.")

        if self.verbose:
            print("Playing back the spatialized audio...")
        sd.play(self.audio_output, self.sample_rate)
        sd.wait()
        if self.verbose:
            print("Playback finished.")


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a sine wave.

    :param frequency: Frequency of the sine wave in Hz.
    :param duration: Duration of the sine wave in seconds.
    :param sample_rate: Sampling rate in Hz.
    :param amplitude: Amplitude of the sine wave.
    :return: NumPy array containing the sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


if __name__ == "__main__":
    audio_handler = AudioHandler()
    # Initialize the spatializer with default parameters
    spatializer = SoundSpatializer(
        sample_rate=44100,
        listener_position=(0.0, 0.0)   # Listener at the origin in meters
    )

    spatializer.apply_delays = False
    # # # Define stereo speaker positions (left and right) in meters
    # stereo_speakers = [(-0.1, 0.0), (0.1, 0.0)]  # Left and Right speakers on the x-axis
    # spatializer.define_speakers(stereo_speakers)

    # Define 12 speaker positions.
    # x: 9.5m y: 7.7m ->x/2 4.75 y/2 3.85

    num_speakers = 12
    radius = 6.0  # meters
    cork_speakers = []
    cork_speakers.append([4.75, -3.85])
    cork_speakers.append([4.75, -1.9])
    cork_speakers.append([4.75, +1.9])
    cork_speakers.append([4.75, +3.85])
    cork_speakers.append([2.35, +3.85])
    cork_speakers.append([-2.35, +3.85])
    cork_speakers.append([-4.75, +3.85])
    cork_speakers.append([-4.75, +1.9])
    cork_speakers.append([-4.75, -1.9])
    cork_speakers.append([-4.75, -3.85])
    cork_speakers.append([-2.35, -3.85])
    cork_speakers.append([+2.35, -3.85])

    spatializer.define_speakers(cork_speakers)

    # Generate a 500 Hz sine wave for 0.5 seconds
    frequency = 500.0  # Hz
    duration = 1  # seconds
    sample_rate = 44100
    num_iterations = 12  # Number of iterations for shifting the sound

    sound = generate_sine_wave(frequency=frequency, duration=duration, sample_rate=sample_rate)
    list_onsets = np.arange(num_iterations) * (duration + 0.5)
    for i in range(num_iterations):
        angle = 2 * np.pi * i / num_iterations
        position = cork_speakers[i] # Position in a circle around the listener
        onset = list_onsets[i]
        spatializer.add_audio_sample(audio_buffer=sound, position=position, onset=onset)
        # print(f"Added AudioSample at position: {position:3f} with onset {onset:.2f} seconds")

    # Perform spatialization and play the sound
    spatializer.spatialize()
    
    audio_handler.save_sound_multichannel(spatializer.audio_output, "latest.npy")

    # spatializer.play_audio()