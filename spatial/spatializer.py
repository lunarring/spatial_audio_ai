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

CHUNKSIZE = 1024*16
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

    def __init__(self, sound: np.ndarray, position: np.ndarray | None=np.zeros(2, dtype=float)):
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
        
        return SoundMessage(sound=sound, position=self.position)
    
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

    def process(self, sm: SoundMessage):
        # print(sm.position)
        distances = np.linalg.norm((self.speaker_positions - sm.position), axis=1)
        attenuation = 1 / (1 + self.attenuation_scaler * distances)
        
        # print(distances)
        # attenuation_flat = np.zeros_like(attenuation)
        # attenuation_flat[np.argmax(attenuation)] = 1
        # attenuation = attenuation_flat
        
        buffer = np.expand_dims(attenuation, 1) * sm.sound
        
        
        
        if self.subwoofer_last_channel_auto_mode:
            buffer = np.append(buffer, np.expand_dims(np.mean(buffer, axis=0), axis=0), axis=0)
        # buffer *= np.expand_dims(self.channel_base_factors, axis=1)
        return buffer

class Scene:
    def __init__(
            self,
            spatializer: Spatializer, 
    ):
        self.sound_objects = []
        self.tick = 0
        self.spatializer = spatializer

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
            
            self.tick += 1
            yield buffer
            
            time.sleep(1.0 * CHUNKSIZE / SAMPLING_RATE)
        

so_a = SO_Playback(sf.read("/home/lugo/Downloads/song.wav")[0][:,0])
# so_a = SO_Playback(sf.read("/home/lugo/audio/export/Whispering_rustle_of_fallen_leaves.wav")[0], np.array([0,0]))
so_b = SO_Playback(sf.read("/home/lugo/audio/export/Leaves_crunching_underfoot_softly.wav")[0])

spatializer = Spatializer()
scene = Scene(spatializer)
scene.register(so_a)
# scene.register(so_b)


import logging
import numpy as np
from numpysocket import NumpySocket


# logger = logging.getLogger("simple client")
# logger.setLevel(logging.INFO)

# x = np.tile(np.expand_dims(so_b.sound,0), [13,1])
# x[:,:] = 0
# x[6,:] = so_b.sound

# xxx
# x = np.random.rand(12,44100)*0.1

#%%#
import lunar_tools as lt
receiver = lt.OSCReceiver('10.40.50.9')

# while True:
#     angle = receiver.get_last_value("/speaker_angle")
#     x = radius * np.sin(angle)
#     y = radius * np.cos(angle)
#     print(f"{x} {y} {angle}")
    
#%%
radius = 6

with NumpySocket() as s:
    
    # logger.info("sending numpy array:")
    s.connect(("10.40.49.21", 9999))
    for chunk in scene.run():
        angle = - receiver.get_last_value("/speaker_angle")
        x = radius * np.sin(angle)
        y = radius * np.cos(angle)
        position = np.array([x, y])
        print(f"{x} {y} {angle}")
        
        # position = spatializer.speaker_positions[2]
        so_a.set_position(position)
        # l = chunk.copy()
        # chunk[:,:] = 0
        # chunk[11,:] = l[11,:]
        s.sendall(chunk)
        # s.detach()



