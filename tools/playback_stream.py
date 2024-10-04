import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
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

sd.default.blocksize = BLOCKSIZE
        
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
    
