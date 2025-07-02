import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import soundfile as sf
import time
import os
import random
from spatial_audio_ai.tools.fast_network import QueueManagedStreamer
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE
import uuid

sd.default.blocksize = BLOCKSIZE
# Use BLOCKSIZE directly for minimal latency - no more 4x multiplication!
CHUNKSIZE = BLOCKSIZE

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

    sound_streamer = QueueManagedStreamer()
    if sound_streamer.connect():
        for j, chunk in enumerate(scene.run()):
            if j > 1000:  # Limit for example
                break

            if np.random.rand() < p_inject:
                wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
                random_file = random.choice(wav_files)
                sound = sf.read(f"{dir_scan}{random_file}")[0]
                # Set random radius, speed, angle, direction
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
                print(f"Injected moving sound: {random_file} at "
                      f"radius: {radius:.2f}, speed: {speed:.2f}")
            
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send_with_queue_management(chunk)
            
        sound_streamer.disconnect()
    else:
        print("Failed to connect to fast audio server")
        
    
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
            if j == 0:
                time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.05)
            else:
                time.sleep(CHUNKSIZE/SAMPLING_RATE)
    
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
            time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)