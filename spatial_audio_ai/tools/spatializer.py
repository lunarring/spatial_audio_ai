import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
import soundfile as sf
import time
from spatial_audio_ai.tools.numpysocket import FastNumpySocket
import os
import random
from spatial_audio_ai.tools.client import SoundNetworkStreamer
from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE, CHUNKSIZE
import uuid
from spatial_audio_ai.tools.sound_objects import SoundMessage, SoundObjectTerminatedException, SoundObjectBase

sd.default.blocksize = BLOCKSIZE
    



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

    sound_streamer = SoundNetworkStreamer(profile='ultra_low_latency')
    
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
        sound_streamer = SoundNetworkStreamer(profile='ultra_low_latency')
        
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
    
        sound_streamer = SoundNetworkStreamer(profile='ultra_low_latency')
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