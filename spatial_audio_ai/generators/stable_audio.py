#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys

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


class StableAudioOpen:
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
        self.sampling_rate = 44100
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

    def set_seed(self, seed=420):
        self.seed = seed
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def set_random_seed(self):
        seed = np.random.randint(99999999999)
        self.set_seed(seed)
        return seed

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def set_audio_end_in_s(self, audio_end_in_s):
        self.audio_end_in_s = audio_end_in_s

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


class StableAudioOpenSmall:
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
        
        self.sampling_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.sampler_type = sampler_type
        self.force_mono = force_mono
        self.audio_end_in_s = audio_end_in_s
        self.fade_duration = fade_duration
        self.seed = None

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
    def __init__(self, audio_diffusion, base_dir='/home/lugo/audio/export'):
        self.audio_diffusion = audio_diffusion
        self.base_dir = base_dir
        self.min_duration_sound = 4
        self.max_duration_sound = 8

    def set_base_dir(self, base_dir):
        self.base_dir = base_dir

    def set_min_duration_sound(self, min_duration):
        self.min_duration_sound = min_duration

    def set_max_duration_sound(self, max_duration):
        self.max_duration_sound = max_duration

    def generate(self, list_prompts, name_space, nmb_sounds):
        output_dir = f'{self.base_dir}/{name_space}'
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


# Examples
# %% make me here an example of how to use the StableAudioOpenSmall class
# Change line below to if __name__ == "__main__" to run it.
if __name__ == "__main__":
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


# %% Generate a sound pool given many prompts. Saves wavs to disk. 
# Change line below to if __name__ == "__main__" to run it. 
# This example requires you have the repo rtd_comfy
if __name__ == "__main__XXX":
    audio_diffusion = StableAudioOpen(num_inference_steps=100)
    spg = SoundPoolGenerator(audio_diffusion)
    spg.set_min_duration_sound(3)
    spg.set_max_duration_sound(8)
    spg.set_base_dir('soundpools')
    list_prompts = ['wind', 'water', 'fire', 'earth']
    spg.generate(list_prompts, name_space='elements', nmb_sounds=100)


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



