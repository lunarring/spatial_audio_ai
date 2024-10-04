import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
sd.default.blocksize = 1024
from abc import ABC, abstractmethod
from dataclasses import dataclass
import soundfile as sf
import time
from numpysocket import NumpySocket
import os
import random
from playback_stream import SoundNetworkStreamer

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


class SO_PlaybackCircularMove(SO_Playback):
    def __init__(self, sound: np.ndarray, position: np.ndarray | None = np.zeros(2, dtype=float), radius: float = 1.0, speed: float = 1.0):
        super().__init__(sound, position)
        self.radius = radius
        self.speed = speed
        self.start_time = time.time()

    def update_position(self):
        """
        Update the position of the sound object to move in a circular path.
        The position is updated based on the elapsed time since the start.
        """
        elapsed_time = time.time() - self.start_time
        angle = self.speed * elapsed_time
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        self.set_position(np.array([x, y]))



class Spatializer:
    def __init__(self,
                 subwoofer_last_channel_auto_mode=True,
                 process_function=None
    ):
        self.speaker_positions = np.asarray([
            (4.75, -3.85), (4.75, -1.9), (4.75, 1.9), (4.75, 3.85),
            (2.35, 3.85), (-2.35, 3.85), (-4.75, 3.85), (-4.75, 1.9),
            (-4.75, -1.9), (-4.75, -3.85), (-2.35, -3.85), (2.35, -3.85)
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
        
    


    # pool based sound scape playback
    if True:
        name_space = "ocean"
        p_inject = 0.4
        box_size = 25
        dir_scan = f'/home/lugo/audio/export/{name_space}/'


        # Get a list of all .wav files in dir_scn
        wav_files = [f for f in os.listdir(dir_scan) if f.endswith('.wav')]
        sound = sf.read(f"{dir_scan}{wav_files[0]}")[0]
        
        # sound = apply_reverb(sound, SAMPLING_RATE, reverb_decay=0.3)
    
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
    
