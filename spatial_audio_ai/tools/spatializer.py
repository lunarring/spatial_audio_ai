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


class EarsSpatializer:
    def __init__(self,
                 ears_position=np.array([0.0, 0.0]),
                 subwoofer_last_channel_auto_mode=True
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
        self.ears_position = ears_position
        self.subwoofer_last_channel_auto_mode = subwoofer_last_channel_auto_mode
        self.attenuation_scaler = 1.0
        self.last_active_speakers = []  # Track which speakers are playing
        self.last_speaker_volumes = np.zeros(len(self.speaker_positions))  # Track speaker volumes (normalized 0..1)
        self.last_speaker_gains_raw = np.zeros(len(self.speaker_positions))  # Track raw attenuation factors

    def set_ears_position(self, ears_position):
        """Set the position of the ears in space"""
        self.ears_position = ears_position
    
    def get_active_speakers(self):
        """Get list of currently active speaker indices"""
        return self.last_active_speakers
    
    def get_speaker_volumes(self):
        """Get current volume levels for all speakers (0-1 range)"""
        return self.last_speaker_volumes.copy()

    def get_speaker_gains(self):
        """Get current raw attenuation factors per speaker (not normalized)."""
        return self.last_speaker_gains_raw.copy()

    def process(self, sm: SoundMessage):
        """
        Process a SoundMessage using ear-based spatialization.
        
        1. Compute vector d between sound position and ears position
        2. Find speakers in the direction from ears toward sound
        3. Volume increases as sound approaches ears
        """
        # Compute vector d from ears to sound
        d_vector = sm.position - self.ears_position
        d_norm = np.linalg.norm(d_vector)
        
        # Calculate attenuation for all speakers (zero for non-selected ones)
        attenuation = np.zeros(len(self.speaker_positions))
        
        if d_norm < 1e-6:  # Handle case where sound is at ears position
            # Sound is at ears - very loud, use closest speakers
            distances = np.linalg.norm((self.speaker_positions - sm.position), axis=1)
            closest_indices = np.argsort(distances)[:2]
            self.last_active_speakers = closest_indices.tolist()
            
            for speaker_idx in self.last_active_speakers:
                attenuation[speaker_idx] = 1.0  # Maximum volume when at ears
        else:
            # Normalize the direction vector
            d_direction = d_vector / d_norm
            
            # Find speakers using perpendicular distance only (forward ray)
            target_t = float(np.dot((sm.position - self.ears_position), d_direction))
            closest_speakers = self._find_speakers_in_direction(
                self.ears_position, d_direction, self.speaker_positions, target_t=target_t
            )
            self.last_active_speakers = closest_speakers

            # Volume increases as sound approaches ears (stronger near, weaker far)
            # Purely ear-distance based (independent of speaker geometry)
            base_volume = np.clip(1.0 / (1e-3 + d_norm), 0.1, 1.0)

            # Compute perpendicular distances for the two speakers
            perp_distances = []
            for speaker_idx in closest_speakers:
                sp = self.speaker_positions[speaker_idx]
                v = sp - self.ears_position
                t = float(np.dot(v, d_direction))
                proj_point = self.ears_position + t * d_direction
                d_perp = float(np.linalg.norm(sp - proj_point))
                perp_distances.append(d_perp)

            # Convert perpendicular distances to mixing weights
            eps = 1e-6
            if perp_distances[0] <= 1e-3 and (len(perp_distances) == 1 or perp_distances[0] < perp_distances[1] - 1e-6):
                weights = [1.0, 0.0]
            elif len(perp_distances) > 1 and perp_distances[1] <= 1e-3 and perp_distances[1] < perp_distances[0] - 1e-6:
                weights = [0.0, 1.0]
            else:
                # Inverse-distance weighting on perpendiculars
                inv = [1.0 / max(d, eps) for d in perp_distances]
                s = inv[0] + (inv[1] if len(inv) > 1 else 0.0)
                if s <= 0:
                    weights = [0.5, 0.5]
                else:
                    w0 = inv[0] / s
                    w1 = (inv[1] / s) if len(inv) > 1 else 0.0
                    weights = [w0, w1]

            # Normalize weights to sum to 1 (energy distribution independent of geometry)
            wsum = max(sum(weights), eps)
            weights = [w / wsum for w in weights]

            # Apply only ear-distance scaling; do not attenuate by speaker distance
            for i, speaker_idx in enumerate(closest_speakers):
                attenuation[speaker_idx] = weights[i] * base_volume
        
        # Store raw gains and normalized volumes for visualization
        self.last_speaker_gains_raw = attenuation.copy()
        vis = self.last_speaker_gains_raw.copy()
        max_val = float(np.max(vis)) if vis.size > 0 else 0.0
        if max_val > 0:
            vis = vis / max_val
        self.last_speaker_volumes = vis
        
        buffer = np.expand_dims(attenuation, 1) * sm.sound
        
        if self.subwoofer_last_channel_auto_mode:
            buffer = np.append(buffer, np.expand_dims(np.mean(buffer, axis=0), axis=0), axis=0)
        
        return buffer
    
    def _find_speakers_in_direction(self, ears_pos, direction, speaker_positions, target_t=None):
        """
        Choose the two speakers with the smallest perpendicular distance to the
        ears→sound line (ray) in the forward direction. Projection length (t)
        is only used to discard speakers behind the ears; ranking is purely by
        perpendicular distance to the line.

        Args:
            ears_pos: (2,) ears position
            direction: (2,) unit vector from ears to sound
            speaker_positions: (N,2) array
            target_t: unused (kept for signature compatibility)

        Returns:
            List[int]: indices of two best speakers
        """
        candidates = []
        for i, sp in enumerate(speaker_positions):
            v = sp - ears_pos
            t = float(np.dot(v, direction))
            if t < 0:
                continue  # behind ears, ignore
            proj_point = ears_pos + t * direction
            d_perp = float(np.linalg.norm(sp - proj_point))
            candidates.append((i, d_perp))

        if not candidates:
            # Fallback: if everything is behind (rare), use global perpendicular distance w.r.t. infinite line
            # Compute to entire line without the t>=0 constraint
            all_perp = []
            for i, sp in enumerate(speaker_positions):
                v = sp - ears_pos
                t_any = float(np.dot(v, direction))
                proj_point = ears_pos + t_any * direction
                d_perp_any = float(np.linalg.norm(sp - proj_point))
                all_perp.append((i, d_perp_any))
            all_perp.sort(key=lambda x: x[1])
            return [all_perp[0][0], all_perp[1][0] if len(all_perp) > 1 else all_perp[0][0]]

        candidates.sort(key=lambda x: x[1])
        return [candidates[0][0], candidates[1][0] if len(candidates) > 1 else candidates[0][0]]


class Scene:
    def __init__(
            self,
            spatializer, 
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