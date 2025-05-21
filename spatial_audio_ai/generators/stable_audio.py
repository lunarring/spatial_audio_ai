#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import numpy as np
import sys
import lunar_tools as lt
import torch
import os
from datetime import datetime
sys.path.append("../tools")
from tools import apply_fade_in_out, save_sound, clean_prompt_for_filename
import random


class AudioDiffusion:
    def __init__(
        self, 
        num_inference_steps=100, 
        force_mono=True,
        audio_end_in_s=10,
        fade_duration=0.2
    ):
        self.pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
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
        return self.pipe.encode_prompt(prompt, self.device, self.do_classifier_free_guidance)

    def generate_sound(self, prompt_embeds, num_waveforms_per_prompt=1):
        assert not isinstance(prompt_embeds, str), "prompt_embeds should not be a string. Use get_embedding method first."
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
        blended_embed = self.em.blend_two_embeds([embed1], [embed2], weight)[0]
        
        return blended_embed

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
            duration = random.uniform(self.min_duration_sound, self.max_duration_sound)
            seed = self.audio_diffusion.set_random_seed()

            self.audio_diffusion.set_seed(seed)
            self.audio_diffusion.set_audio_end_in_s(duration)

            prompt_embeds = self.audio_diffusion.get_embedding(prompt)
            sound = self.audio_diffusion.generate_sound(prompt_embeds)
            sound = apply_fade_in_out(sound)

            filename = clean_prompt_for_filename(prompt)
            file_path = os.path.join(output_dir, f"{filename}_{seed}.wav")
            save_sound(sound, file_path, self.audio_diffusion.sampling_rate)
            print(f"Sound generation complete ({i+1}/{nmb_sounds}). File saved: '{filename}_{seed}.wav'")



# Examples
# %% just generate a short sound and save it. Change line below to if __name__ == "__main__" to run it. 
if __name__ == "__main__":
    
    audio_diffusion = AudioDiffusion(num_inference_steps=100)
    audio_diffusion.set_random_seed()
    audio_diffusion.set_audio_end_in_s(6)
    audio_diffusion.set_num_inference_steps(100)
    prompt = "person singing a song"
    prompt_embeds = audio_diffusion.get_embedding(prompt)
    sound = audio_diffusion.generate_sound(prompt_embeds)
    sound = apply_fade_in_out(sound, audio_diffusion.sampling_rate)
    save_sound(sound, f"singing.wav", audio_diffusion.sampling_rate)

    
    
# %% blend two prompts. Change line below to if __name__ == "__main__" to run it. This example requires you have the repo rtd_comfy
if __name__ == "__main__XXX":
    audio_diffusion = AudioDiffusion(num_inference_steps=100)
    prompt1 = "loud metal weird scratching"
    prompt2 = prompt1 + ", psychedelic, horrible, scary"
    weight = 0.5
    
    p1 = audio_diffusion.get_embedding(prompt1)
    p2 = audio_diffusion.get_embedding(prompt2)
    
    prompt_embeds = audio_diffusion.blend_two_embeds(p1, p2, weight)
    sound = audio_diffusion.generate_sound(prompt_embeds)
    sound = apply_fade_in_out(sound, audio_diffusion.sampling_rate)
    save_sound(sound, f"blended.wav", audio_diffusion.sampling_rate)
        
    
# %% Generate a sound pool given many prompts. Saves wavs to disk. Change line below to if __name__ == "__main__" to run it. This example requires you have the repo rtd_comfy
if __name__ == "__main__XXX":
    audio_diffusion = AudioDiffusion(num_inference_steps=100)
    spg = SoundPoolGenerator(audio_diffusion)
    spg.set_min_duration_sound(3)
    spg.set_max_duration_sound(8)
    spg.set_base_dir('soundpools')
    list_prompts = ['wind', 'water', 'fire', 'earth']
    spg.generate(list_prompts, name_space='elements', nmb_sounds=100)



