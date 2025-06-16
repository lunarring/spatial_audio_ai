#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import abc

import numpy as np
import torch
from diffusers import StableAudioPipeline

try:
    from einops import rearrange
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    STABLE_AUDIO_TOOLS_AVAILABLE = True
except ImportError:
    STABLE_AUDIO_TOOLS_AVAILABLE = False

from spatial_audio_ai.tools.tools import (
    apply_fade_in_out, 
    save_sound, 
    clean_prompt_for_filename
)


class AudioDiffusion(abc.ABC):
    @abc.abstractmethod
    def set_seed(self, seed=420):
        pass

    @abc.abstractmethod
    def set_random_seed(self):
        pass

    @abc.abstractmethod
    def set_audio_end_in_s(self, audio_end_in_s):
        pass

    @property
    @abc.abstractmethod
    def sampling_rate(self):
        pass

    @abc.abstractmethod
    def generate_sound(self, *args, **kwargs):
        pass


class StableAudioOpen(AudioDiffusion):
    def __init__(
        self, 
        num_inference_steps=100, 
        force_mono=True,
        audio_end_in_s=10,
        fade_duration=0.2
    ):
        self.pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            torch_dtype=torch.float16
        ).to("cuda")
        from spatial_audio_ai.config import SAMPLING_RATE
        self._sampling_rate = SAMPLING_RATE
        self.seed = None
        self.generator = None
        self.device = self.pipe._execution_device
        self.do_classifier_free_guidance = True
        self.num_inference_steps = num_inference_steps
        self.force_mono = force_mono
        self.audio_end_in_s = audio_end_in_s
        self.fade_duration = fade_duration
        self.em = None
        self.set_seed()

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def set_seed(self, seed=420):
        self.seed = seed
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def set_random_seed(self):
        seed = np.random.randint(99999999999)
        self.set_seed(seed)
        return seed

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def get_embedding(self, prompt):
        return self.pipe.encode_prompt(
            prompt, 
            self.device, 
            self.do_classifier_free_guidance
        )

    def generate_sound(self, prompt_embeds, num_waveforms_per_prompt=1):
        assert not isinstance(prompt_embeds, str), (
            "prompt_embeds should not be a string. "
            "Use get_embedding method first."
        )
        # prompt_embeds = self.get_embedding(prompt)
        audio = self.pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=self.num_inference_steps,
            audio_end_in_s=self.audio_end_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=self.generator,
        ).audios

        output = audio[0].T.float().cpu().numpy()
        if self.force_mono:
            output = output[:, 0]
        return output

    def blend_two_embeds(self, embed1, embed2, weight):
        if self.em is None:
            sys.path.append('../../rtd_comfy/sdxl_turbo')
            from embeddings_mixer import EmbeddingsMixer
            self.em = EmbeddingsMixer(self.pipe)
        assert 0 <= weight <= 1, "Weight should be between 0 and 1"
        blended_embed = self.em.blend_two_embeds(
            [embed1], [embed2], weight
        )[0]
        
        return blended_embed


class StableAudioOpenSmall(AudioDiffusion):
    def __init__(
        self,
        steps=8,
        cfg_scale=1.0,
        sampler_type="pingpong",
        force_mono=True,
        audio_end_in_s=10,
        fade_duration=0.2
    ):
        if not STABLE_AUDIO_TOOLS_AVAILABLE:
            raise ImportError(
                "stable_audio_tools, torchaudio, and einops are required "
                "for StableAudioOpenSmall. Install with: "
                "pip install stable-audio-tools torchaudio einops"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-small"
        )
        self.model = self.model.to(self.device)
        
        self._sampling_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.sampler_type = sampler_type
        self.force_mono = force_mono
        self.audio_end_in_s = audio_end_in_s
        self.fade_duration = fade_duration
        self.seed = None

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def set_seed(self, seed=420):
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def set_random_seed(self):
        seed = np.random.randint(99999999999)
        self.set_seed(seed)
        return seed

    def set_steps(self, steps):
        self.steps = steps

    def set_cfg_scale(self, cfg_scale):
        self.cfg_scale = cfg_scale

    def set_audio_end_in_s(self, audio_end_in_s):
        self.audio_end_in_s = audio_end_in_s

    def set_sampler_type(self, sampler_type):
        self.sampler_type = sampler_type

    def generate_sound(self, prompt):
        conditioning = [{
            "prompt": prompt,
            "seconds_total": self.audio_end_in_s
        }]
        
        # Calculate sample_size based on desired duration and sample rate
        calculated_sample_size = int(self.audio_end_in_s * self.sampling_rate)
        print(f"Target duration: {self.audio_end_in_s}s, "
              f"Sample rate: {self.sampling_rate}Hz, "
              f"Calculated samples: {calculated_sample_size}")
        
        output = generate_diffusion_cond(
            self.model,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            conditioning=conditioning,
            sample_size=calculated_sample_size,
            sampler_type=self.sampler_type,
            device=self.device
        )
        
        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        
        # Peak normalize, clip, convert to float32
        output = output.to(torch.float32).div(
            torch.max(torch.abs(output))
        ).clamp(-1, 1)
        
        # Convert to numpy
        output = output.cpu().numpy()
        
        if self.force_mono and output.shape[0] > 1:
            output = output[0]  # Take first channel for mono
        elif not self.force_mono and output.shape[0] == 1:
            output = output[0]  # Remove channel dimension for mono output
        elif not self.force_mono:
            output = output.T  # Transpose for stereo (time, channels)
        
        # Print actual duration for verification
        actual_duration = len(output) / self.sampling_rate
        print(f"Generated audio length: {len(output)} samples, "
              f"Actual duration: {actual_duration:.2f}s")
            
        return output


class SoundPoolGenerator:
    def __init__(self, audio_diffusion, directory='/home/lugo/audio/export'):
        self.audio_diffusion = audio_diffusion
        self.directory = directory
        self.min_duration_sound = 4
        self.max_duration_sound = 8

    def set_directory(self, directory):
        self.directory = directory

    def set_min_duration_sound(self, min_duration):
        self.min_duration_sound = min_duration

    def set_max_duration_sound(self, max_duration):
        self.max_duration_sound = max_duration

    def generate(self, list_prompts, nmb_sounds):
        output_dir = self.directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(nmb_sounds):
            prompt = random.choice(list_prompts)
            duration = random.uniform(
                self.min_duration_sound, 
                self.max_duration_sound
            )
            seed = self.audio_diffusion.set_random_seed()

            self.audio_diffusion.set_seed(seed)
            self.audio_diffusion.set_audio_end_in_s(duration)

            # Handle different interfaces for the two classes
            if hasattr(self.audio_diffusion, 'get_embedding'):
                # StableAudioOpen interface
                prompt_embeds = self.audio_diffusion.get_embedding(prompt)
                sound = self.audio_diffusion.generate_sound(prompt_embeds)
            else:
                # StableAudioOpenSmall interface
                sound = self.audio_diffusion.generate_sound(prompt)
                
            sound = apply_fade_in_out(sound)

            filename = clean_prompt_for_filename(prompt)
            file_path = os.path.join(output_dir, f"{filename}_{seed}.wav")
            save_sound(sound, file_path, self.audio_diffusion.sampling_rate)
            print(f"Sound generation complete ({i+1}/{nmb_sounds}). "
                  f"File saved: '{filename}_{seed}.wav'")


class SpatialSoundPoolPlayer:
    """
    Play back generated sound pools with spatial audio positioning.
    """
    def __init__(
        self, 
        name_space, 
        base_dir='/home/lugo/audio/export',
        p_inject=0.3,
        box_size=15.0,
        volume=0.2,
        use_circular=False,
        circular_radius_range=(2.0, 10.0),
        circular_speed_range=(0.2, 1.0),
        circular_angle_range=(0.0, 2*np.pi),
        circular_direction_choices=(-1, 1),
        circular_center=np.zeros(2, dtype=float),
        circular_loop=True
    ):
        from spatial_audio_ai.tools.spatializer import (
            Spatializer, Scene
        )
        from spatial_audio_ai.tools.client import SoundNetworkStreamer
        self.name_space = name_space
        self.base_dir = base_dir
        self.p_inject = p_inject  # Probability of injecting new sound each frame
        self.box_size = box_size  # Size of spatial area
        self.volume = volume
        self.use_circular = use_circular
        self.circular_radius_range = circular_radius_range
        self.circular_speed_range = circular_speed_range
        self.circular_angle_range = circular_angle_range
        self.circular_direction_choices = circular_direction_choices
        self.circular_center = circular_center
        self.circular_loop = circular_loop
        # Directory containing the sound pool
        self.dir_scan = f'{base_dir}/{name_space}/'
        # Initialize spatial audio components
        self.spatializer = Spatializer()
        self.scene = Scene(self.spatializer)
        self.scene.volume = volume
        self.sound_streamer = SoundNetworkStreamer()
        # Check if sound pool exists
        if not os.path.exists(self.dir_scan):
            raise FileNotFoundError(
                f"Sound pool directory not found: {self.dir_scan}"
            )
        # Get initial list of wav files
        self.wav_files = [
            f for f in os.listdir(self.dir_scan) 
            if f.endswith('.wav')
        ]
        if not self.wav_files:
            raise FileNotFoundError(
                f"No .wav files found in {self.dir_scan}"
            )
        print(f"Found {len(self.wav_files)} sounds in pool '{name_space}'")

    def start_initial_sound(self):
        """Start playback with one initial sound"""
        import soundfile as sf
        from spatial_audio_ai.tools.spatializer import SO_Playback, SO_PlaybackCircularMove
        sound = sf.read(f"{self.dir_scan}{self.wav_files[0]}")[0]
        if self.use_circular:
            radius = np.random.uniform(*self.circular_radius_range)
            speed = np.random.uniform(*self.circular_speed_range)
            angle = np.random.uniform(*self.circular_angle_range)
            direction = random.choice(self.circular_direction_choices)
            center = self.circular_center
            loop = self.circular_loop
            so = SO_PlaybackCircularMove(
                sound,
                radius=radius,
                speed=speed,
                initial_angle=angle,
                direction=direction,
                center=center,
                loop=loop
            )
            print(
                f"Started with: {self.wav_files[0]} (circular, "
                f"radius={radius:.2f}, speed={speed:.2f}, "
                f"angle={angle:.2f}, "
                f"direction={direction})"
            )
        else:
            so = SO_Playback(sound, position=np.array([0.0, 0.0]))
            print(f"Started with: {self.wav_files[0]}")
        self.scene.register(so)

    def play(self, duration_minutes=5):
        """
        Play the spatial sound pool for specified duration.
        Args:
            duration_minutes: How long to play (default 5 minutes)
        """
        import soundfile as sf
        import time
        from spatial_audio_ai.tools.spatializer import (
            SO_Playback, SO_PlaybackCircularMove, CHUNKSIZE, SAMPLING_RATE
        )
        self.start_initial_sound()
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        print(f"Starting spatial playback for {duration_minutes} minutes...")
        print(f"Injection probability: {self.p_inject}")
        print(f"Spatial area: {self.box_size}x{self.box_size}")
        try:
            for j, chunk in enumerate(self.scene.run()):
                if np.random.rand() < self.p_inject:
                    self.wav_files = [
                        f for f in os.listdir(self.dir_scan) 
                        if f.endswith('.wav')
                    ]
                    random_file = random.choice(self.wav_files)
                    sound = sf.read(f"{self.dir_scan}{random_file}")[0]
                    if self.use_circular:
                        radius = np.random.uniform(*self.circular_radius_range)
                        speed = np.random.uniform(*self.circular_speed_range)
                        angle = np.random.uniform(*self.circular_angle_range)
                        direction = random.choice(self.circular_direction_choices)
                        center = self.circular_center
                        loop = self.circular_loop
                        so = SO_PlaybackCircularMove(
                            sound,
                            radius=radius,
                            speed=speed,
                            initial_angle=angle,
                            direction=direction,
                            center=center,
                            loop=loop
                        )
                        print(
                            f"Injected: {random_file} (circular, "
                            f"radius={radius:.2f}, speed={speed:.2f}, "
                            f"angle={angle:.2f}, "
                            f"direction={direction})"
                        )
                    else:
                        position = np.random.uniform(
                            -self.box_size, self.box_size, size=2
                        )
                        so = SO_Playback(
                            sound,
                            position=position
                        )
                        print(
                            f"Injected: {random_file} at position: "
                            f"({position[0]:.1f}, {position[1]:.1f})"
                        )
                    self.scene.register(so)
                chunk = np.clip(chunk, -1, 1)
                self.sound_streamer.send(chunk)
                time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
                elapsed = time.time() - start_time
                if elapsed > duration_seconds:
                    print(f"Playback completed after {elapsed:.1f} seconds")
                    break
                if j % 430 == 0:
                    active_sounds = len(self.scene.sound_objects)
                    print(f"Time: {elapsed:.1f}s | "
                          f"Active sounds: {active_sounds}")
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")
        except Exception as e:
            print(f"Playback error: {e}")

    def set_injection_probability(self, p_inject):
        """Set probability of injecting new sounds"""
        self.p_inject = p_inject

    def set_spatial_area(self, box_size):
        """Set the size of the spatial positioning area"""
        self.box_size = box_size

    def set_volume(self, volume):
        """Set overall volume"""
        self.volume = volume
        self.scene.volume = volume


# For backward compatibility
StableAudioDiffusion = AudioDiffusion


# Examples
# %% make me here an example of how to use the StableAudioOpenSmall class
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__XXX":
    # Initialize StableAudioOpenSmall with custom parameters
    audio_diffusion_small = StableAudioOpenSmall(
        steps=8,
        cfg_scale=1.0,
        sampler_type="pingpong",
        force_mono=True,  # Use mono for compatibility with fade function
        audio_end_in_s=20
    )
    
    # Set a random seed for reproducibility
    seed = audio_diffusion_small.set_random_seed()
    print(f"Using seed: {seed}")
    
    # Generate a tech house drum loop
    prompt = "128 BPM tech house drum loop"
    print(f"Generating: {prompt}")
    sound = audio_diffusion_small.generate_sound(prompt)
    sound = apply_fade_in_out(sound, audio_diffusion_small.sampling_rate)
    save_sound(
        sound, "tech_house_small.wav", audio_diffusion_small.sampling_rate
    )
    print("Tech house saved as tech_house_small.wav")


# %% just generate a short sound and save it. 
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__XXX":
    
    audio_diffusion = StableAudioOpen(num_inference_steps=100)
    audio_diffusion.set_random_seed()
    prompt = "person singing a song"
    prompt_embeds = audio_diffusion.get_embedding(prompt)
    sound = audio_diffusion.generate_sound(prompt_embeds)
    sound = apply_fade_in_out(sound, audio_diffusion.sampling_rate)
    save_sound(sound, "singing.wav", audio_diffusion.sampling_rate)


# %% blend two prompts. Change line below to if __name__ == "__main__" 
# to run it. This example requires you have the repo rtd_comfy
if __name__ == "__main__XXX":
    audio_diffusion = StableAudioOpen(num_inference_steps=100)
    prompt1 = "loud metal weird scratching"
    prompt2 = prompt1 + ", psychedelic, horrible, scary"
    weight = 0.5
    
    p1 = audio_diffusion.get_embedding(prompt1)
    p2 = audio_diffusion.get_embedding(prompt2)
    
    prompt_embeds = audio_diffusion.blend_two_embeds(p1, p2, weight)
    sound = audio_diffusion.generate_sound(prompt_embeds)
    sound = apply_fade_in_out(sound, audio_diffusion.sampling_rate)
    save_sound(sound, "blended.wav", audio_diffusion.sampling_rate)


# %% Example using StableAudioOpenSmall
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__XXX":
    audio_diffusion_small = StableAudioOpenSmall(steps=8, cfg_scale=1.0)
    audio_diffusion_small.set_random_seed()
    audio_diffusion_small.set_audio_end_in_s(11)
    prompt = "128 BPM tech house drum loop"
    sound = audio_diffusion_small.generate_sound(prompt)
    sound = apply_fade_in_out(sound, audio_diffusion_small.sampling_rate)
    save_sound(sound, "tech_house.wav", audio_diffusion_small.sampling_rate)


# %% Generate a sound pool given many prompts using StableAudioOpenSmall
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__":
    # Use the smaller, faster model for bulk generation
    audio_diffusion_small = StableAudioOpenSmall(
        steps=8, 
        cfg_scale=1.0,
        force_mono=True  # Ensure mono for fade compatibility
    )
    
    spg = SoundPoolGenerator(
        audio_diffusion_small,
        directory="/home/lugo/audio/export/machine"
    )
    spg.set_min_duration_sound(3)
    spg.set_max_duration_sound(8)
    
    # Test with ambient prompts for atmospheric sound generation
    list_prompts = [
        # Machine/Electronic sounds
        'malfunctioning industrial machinery',
        'glitchy circuit-bent electronics',
        'metallic scraping with digital artifacts',
        'corrupted data transmission noise',
        'robotic voice distortion breakdown',
        'quantum computer error sounds',
        'mechanical insect grinding gears',
        'broken electronic appliance feedback',
        'alien machinery powering up',
        'fractured digital signal processing',
        'rusted mechanical clockwork malfunction',
        'extraterrestrial communication static',
        
        # Natural environments
        'gentle forest stream with birdsong',
        'thunderstorm rolling across mountains',
        'tropical rainforest at dawn',
        'crashing ocean waves on rocky shore',
        'wind howling through ancient canyon',
        'crackling campfire under starry night',
        'bamboo forest rustling in light breeze',
        'distant whale songs underwater',
        'ice cracking on frozen lake',
        'heavy rainfall on tropical leaves',
        
        # Human/Organic
        'bustling open-air market conversations',
        'children playing in a playground',
        'heart beating with deep breathing',
        'choir singing harmonic overtones',
        'distant crowd cheering at stadium',
        'woman humming lullaby to baby',
        'creaking wooden floorboards in old house',
        'monks chanting in stone monastery',
        'slow footsteps on gravel path',
        'person whispering secrets',
        
        # Musical elements
        '128 BPM tribal drumming with shakers',
        'melancholic cello solo with reverb',
        'analog synthesizer ambient pad',
        'jazzy piano improvisation with brushed drums',
        'orchestral string section swells',
        'acoustic guitar fingerpicking pattern',
        'slow ethereal harp arpeggios',
        'distorted electric guitar feedback',
        'minimalist piano with felt dampeners',
        'tabla rhythms with tanpura drone',
        
        # Abstract/Experimental
        'crystalline structures dissolving in acid',
        'time-stretched glass breaking',
        'memories fading into cosmic void',
        'consciousness expanding beyond dimensions',
        'neural pathways forming connections',
        'quantum particles colliding in vacuum',
        'ancient knowledge transmitted telepathically',
        'dreams echoing through subconscious',
        'cellular division amplified',
        'thoughts materializing into sound waves',
        
        # Hybrid/Mixed
        'forest creatures playing metallic instruments',
        'digital rainstorm on cybernetic landscape',
        'organic heartbeat with electronic processing',
        'bird calls transformed through vocoder',
        'underwater cave with resonant frequencies',
        'human breath controlling synthesizer parameters',
        'insect swarm with granular synthesis',
        'stone temple with electromagnetic resonance',
        'crackling fire with ambient string quartet',
        'shamanic ritual with analog oscillators'
    ]
    
    print(f"Generating {len(list_prompts) * 2} sounds using "
          f"StableAudioOpenSmall...")
    spg.generate(list_prompts, 100)


# %% Play back the generated sound pool with spatial audio
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__XXX":
    # Create spatial sound pool player for the generated sounds
    player = SpatialSoundPoolPlayer(
        name_space='elements_small',
        base_dir='soundpools',
        p_inject=0.3,  # 30% chance to inject new sound each frame
        box_size=15.0,  # 15x15 unit spatial area
        volume=0.2
    )
    
    # Play for 3 minutes
    player.play(duration_minutes=3)


