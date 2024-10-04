import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from tools import AudioHandler
sd.default.blocksize = 1024
from abc import ABC, abstractmethod
from dataclasses import dataclass
import soundfile as sf
import time
from numpysocket import NumpySocket
import os
import random

BLOCKSIZE = 1024
CHUNKSIZE = BLOCKSIZE*4
SAMPLING_RATE = 44100

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

    def query(self, tick: int) -> SoundMessage:
        if self.first_tick is None:
            self.first_tick = tick
        start = (tick - self.first_tick) * CHUNKSIZE
        sound = self.sound[start:start+CHUNKSIZE]
        if len(sound) == 0:
            raise SoundObjectTerminatedException
        sound = np.pad(sound, (0, CHUNKSIZE - len(sound)), 'constant')
        self.play_with_position()
        
        return SoundMessage(sound=sound, position=self.position)
    
    def play_with_position(self):
        period = 2 * np.pi * random.uniform(1, 10)  # Randomized period for one full rotation
        random_radius = random.uniform(1, 10)
        x = random_radius * np.sin(time.time() * (2 * np.pi / period))
        y = random_radius * np.cos(time.time() * (2 * np.pi / period))
        self.position += np.array([x, y])
    
    def set_position(self, new_position: np.ndarray):
        self.position = new_position

class Spatializer:
    def __init__(self,
                 subwoofer_last_channel_auto_mode = True,
    ):
        self.speaker_positions = np.asarray([
            (4.75, -3.85), (4.75, -1.9), (4.75, 1.9), (4.75, 3.85),
            (2.35, 3.85), (-2.35, 3.85), (-4.75, 3.85), (-4.75, 1.9),
            (-4.75, -1.9), (-4.75, -3.85), (-2.35, -3.85), (2.35, -3.85)
        ])
        # self.channel_base_factors = 0.5 * np.array([1.0, 0.85, 0.8, 1.0, 0.8, 0.75, 1.3, 1.0, 1.2, 1.5, 1.3, 0.7, 3.0], dtype=np.float32)
        self.subwoofer_last_channel_auto_mode = subwoofer_last_channel_auto_mode
        self.attenuation_scaler = 1.0
        self.process = self.process_simple
        # self.process = self.process_only1

    def process_simple(self, sm: SoundMessage):
        distances = np.linalg.norm((self.speaker_positions - sm.position), axis=1)
        attenuation = 1 / (1 + self.attenuation_scaler * distances)
        buffer = np.expand_dims(attenuation, 1) * sm.sound
        
        if self.subwoofer_last_channel_auto_mode:
            buffer = np.append(buffer, np.expand_dims(np.mean(buffer, axis=0), axis=0), axis=0)
        return buffer
    
    def process_only1(self, sm: SoundMessage):
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
        
class SoundNetworkStreamer:
    def __init__(self, host: str = "10.40.49.21", port: int = 9999):
        self.host = host
        self.port = port
        self.socket = NumpySocket()
        self.__enter__()

    def __enter__(self):
        self.socket.connect((self.host, self.port))
        return self

    def close(self):
        self.socket.close()

    def send(self, data: np.ndarray):
        if data.ndim == 2:
            if data.shape[1] % BLOCKSIZE == 0:
                # print(data.shape)
                self.socket.sendall(data)
            else:
                print(f"Data shape[1] must be divisible by blocksize. Current shape[1]: {data.shape[1]}, blocksize: {BLOCKSIZE}")
        else:
            print("Data must be a 2-dimensional numpy array")
                
if __name__ == "__main__":
    # simple placement of two objects
    if False:
        # def generate_sine_tone(frequency, duration, sampling_rate=44100):
        #     t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        #     sine_wave = np.sin(2 * np.pi * frequency * t)
        #     return sine_wave
        
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
        
#%%

    import numpy as np
    from scipy import signal
    from scipy.io.wavfile import write
    
    def apply_reverb(waveform, sr, reverb_decay=0.5):
        """
        Apply a simple reverb effect by convolving the waveform with a generated impulse response.
        
        Parameters:
        - waveform: numpy array, the input sound waveform.
        - sr: int, the sample rate of the waveform.
        - reverb_decay: float, how fast the reverb decays (between 0 and 1, where 0 is no reverb).
        
        Returns:
        - reverb_waveform: numpy array, the waveform with reverb applied.
        """
        # Create a simple impulse response to simulate reverb.
        # A more complex impulse response would provide a more realistic effect.
        ir_length = int(sr * 0.3)  # 0.3 second reverb tail
        impulse_response = np.zeros(ir_length)
        impulse_response[0] = 1.0  # Direct sound
        for i in range(1, ir_length):
            impulse_response[i] = reverb_decay ** i  # Simulated reverb decay
    
        # Normalize the impulse response to avoid clipping
        impulse_response /= np.max(np.abs(impulse_response))
    
        # Convolve the input waveform with the impulse response to create the reverb effect
        reverb_waveform = signal.convolve(waveform, impulse_response, mode='full')
    
        # Normalize the output to avoid clipping
        reverb_waveform /= np.max(np.abs(reverb_waveform))
    
        # Return the reverb-processed waveform
        return reverb_waveform

    # pool based sound scape playback
    if True:
        name_space = "ocean"
        p_inject = 0.4
        box_size = 25
        dir_scan = f'/home/lugo/audio/export/{name_space}/'


        # Get a list of all .wav files in dir_scn
        wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
        sound = sf.read(f"{dir_scan}{wav_files[0]}")[0]
        
        sound = apply_reverb(sound, SAMPLING_RATE, reverb_decay=0.3)
    
        so = SO_Playback(sound)
    
        spatializer = Spatializer()
        scene = Scene(spatializer)
        scene.volume = 0.2
        scene.register(so)
    
        sound_streamer = SoundNetworkStreamer()
        for j, chunk in enumerate(scene.run()):

            if np.random.rand() < p_inject:
                wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
                random_file = random.choice(wav_files)
                sound = sf.read(f"{dir_scan}{random_file}")[0]
                position = np.random.uniform(-box_size, box_size, size=2)
                scene.register(SO_Playback(sound, position=position))
                print(f"Injected sound: {random_file} at position: {position}")
                
            
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            
            # if j == 5:
            #     scene.register(SO_Playback(sound_a, position=np.array([-3., -3.1])))
            # if j == 8:
            #     scene.register(SO_Playback(sound_b, position=np.array([3., 3.])))
            # print(f"sent chunk {j}")
            time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
        
    
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
    
