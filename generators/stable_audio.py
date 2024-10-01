#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:54:49 2024

@author: lugo
"""

import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import numpy as np
import sys
sys.path.append('../../rtd_comfy/sdxl_turbo')
from embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
import torch
import os
from datetime import datetime
from tools import AudioHandler


class AudioDiffusion:
    def __init__(
        self, 
        pipe, 
        num_inference_steps=100, 
        force_mono=True,
        audio_end_in_s=10,
        auto_fade=True,
        fade_duration=0.2
    ):
        self.pipe = pipe
        self.em = EmbeddingsMixer(pipe)
        self.ah = AudioHandler(pipe.vae.sampling_rate)
        self.seed = None
        self.generator = None
        self.device = self.pipe._execution_device
        self.do_classifier_free_guidance = True
        self.num_inference_steps = num_inference_steps
        self.force_mono = force_mono
        self.audio_end_in_s = audio_end_in_s
        self.auto_fade = auto_fade
        self.fade_duration = fade_duration
        self.set_seed()

    def set_seed(self, seed=420):
        self.seed = seed
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def set_random_seed(self):
        seed = np.random.randint(99999999999)
        self.set_seed(seed)

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
        if self.auto_fade:
            output = self.ah.apply_fade_in_out(output, self.fade_duration)
        return output


    def get_latents(self):
        print("NOT IMPLEMENTED! LEGACY CODE")

        device = self._execution_device
        do_classifier_free_guidance = True

        p1 = self.encode_prompt(prompt1, device, do_classifier_free_guidance)


        num_channels_vae = self.transformer.config.in_channels
        waveform_length = int(self.transformer.config.sample_size)

        generator = torch.Generator("cuda").manual_seed(420)
        latents1 = self.prepare_latents(
            1 * 1,
            num_channels_vae,
            waveform_length,
            p1.dtype,
            device,
            generator,
            None,
            None,
            1,
            audio_channels=self.vae.config.audio_channels, #this is interesting
        )

    def blend_two_embeds(self, embed1, embed2, weight):
        assert 0 <= weight <= 1, "Weight should be between 0 and 1"
        blended_embed = self.em.blend_two_embeds([embed1], [embed2], weight)[0]
        
        return blended_embed




if __name__ == "__main__":
    
    # %%
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    audio_diffusion = AudioDiffusion(pipe, num_inference_steps=100)
    audio_handler = AudioHandler(pipe.vae.sampling_rate)
    
    xxx
    
    # %% generate one simple sound and save 
    audio_diffusion.set_random_seed()
    audio_diffusion.set_audio_end_in_s(5)
    prompt = "hammer knocking wood"
    prompt_embeds = audio_diffusion.get_embedding(prompt)
    output = audio_diffusion.generate_sound(prompt_embeds)
    output = audio_handler.apply_fade_in_out(output)
    audio_handler.save_sound(output, f"hammer.wav")
    
    
    # # multichannel = np.zeros((12, 12 * 44100))
    # list_sounds = []
    # for i in range(12):
    #     list_sounds.append(output)
    #     # multichannel[i, i * 44100:(i + 1) * 44100] = output
    # audio_diffusion.save_sound_multichannel(list_sounds, "multichannel_beats")
    
    
    # %% blend two prompts
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    audio_diffusion = AudioDiffusion(pipe, num_inference_steps=100)
    audio_diffusion.set_random_seed()
    prompt1 = "loud metal weird scratching"
    prompt2 = prompt1 + ", psychedelic, horrible, scary"
    nmb_sounds = 5
    
    p1 = audio_diffusion.get_embedding(prompt1)
    p2 = audio_diffusion.get_embedding(prompt2)
    
    list_sounds = []
    
    for i, weight in enumerate(np.linspace(0, 1, nmb_sounds)):
        prompt_embeds = audio_diffusion.blend_two_embeds(p1, p2, weight)[0]
        
        # run the generation using audio_diffusion class
        output = audio_diffusion.generate_sound(prompt_embeds)
        
        list_sounds.append(output)
    
        audio_handler.save_sound(output, f"sound_{i}.wav")
    
        
    #%% prompt ball
    
    
    prompt_base = "ASMR symphony"
    prompt_base = "pure rap"
    prompt_base = "tiger roar"
    
    prompt_base = "metal knocknig rusty"
    prompt_base = "a hammer hitting the nail, percursive"
    prompt_base = "percussive kick sound"
    prompt_base = "trance"
    prompt_base = "very powerful thunderstorm techno beat deep trippy"
    prompt_base = "DMT trip sound"
    prompt_base = "very powerful thunderstorm techno beat deep trippy"
    prompt_base = "palm leaves rattle rustling rhytmically loud fast, deep beat"
    
    list_augs = ["psychedelic", "beautiful", "mesmerizing", "ethereal", "new age", "tranquil", "serene", "dreamy", "hypnotic", "luminous", "celestial", "blissful", "harmonious", "radiant"]
    
    # list_augs = [
    #     "psychedelic", "trippy", "energetic", "vibrant", "intense", "hypnotic", 
    #     "pulsating", "deep", "groovy", "cosmic", "alien", "futuristic", 
    #     "mechanical", "robotic", "synthetic", "electronic", "spacey", "otherworldly", 
    #     "transcendent", "mind-bending", "euphoric", "dark", "mystical", "shamanic"
    # ]
    
    
    
    nmb_sounds = 12
    weight = 1.0 # 0.0 would be effective mono, 1.0 maximum difference between the different channels
    num_augs = 2  # more nmb_augs = more divergent sounds!
    duration = 32
    seed = 426
    num_inference_steps = 50
    
    basename = prompt_base.replace(" ", "_")
    audio_handler.set_output_dir(f"/home/lugo/audio/export/")
    prompt_embeds_base = audio_diffusion.get_embedding(prompt_base)
    audio_diffusion.set_audio_end_in_s(duration)
    audio_diffusion.set_seed(seed)
    audio_diffusion.set_num_inference_steps(num_inference_steps)
    np.random.seed(seed)  # Set the seed for numpy as well
    list_sounds = []
    new_prompts = []
    
    for i in range(nmb_sounds):
        # Randomly choose augmentations from list_augs based on num_augs
        augments = np.random.choice(list_augs, num_augs, replace=False)
        # Create the new prompt by combining the base prompt with the chosen augmentations
        new_prompt = f"{prompt_base}, {', '.join(augments)}"
        print(new_prompt)
        new_prompts.append(new_prompt)
        
        prompt_embeds_new = audio_diffusion.get_embedding(new_prompt)
        prompt_embeds_mixed = audio_diffusion.blend_two_embeds(prompt_embeds_base, prompt_embeds_new, weight)
        output = audio_diffusion.generate_sound(prompt_embeds_mixed)
        audio_handler.save_sound(output, f"sound_{i}.wav")
        list_sounds.append(output)
    
    # add last output as bass channel...
    list_sounds.append(np.mean(list_sounds, axis=0))
    # force mono
    # for i in range(nmb_sounds-1):
        # list_sounds.append(output)
    
    # Get the current date and time in the format YYMMDD_HHMM
    date_str = datetime.now().strftime("%y%m%d_%H%M")
    
    # Append the date string to the basename
    basename_with_date = f"{date_str}_{basename}"
    
    audio_handler.save_sound_multichannel(list_sounds, basename_with_date)
    audio_handler.save_sound_multichannel(list_sounds, "latest.npy")
    
    # Save prompts, settings, and seed to a text file
    settings = {
        "prompt_base": prompt_base,
        "list_augs": list_augs,
        "nmb_sounds": nmb_sounds,
        "weight": weight,
        "num_augs": num_augs,
        "duration": duration,
        "basename_with_date": basename_with_date,
        "seed": seed  
    }
    
    with open(f"/home/lugo/audio/export/{basename_with_date}.txt", "w") as f:
        f.write("Settings:\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
        f.write("\nPrompts:\n")
        for prompt in new_prompts:
            f.write(f"{prompt}\n")
    
    
    #%% resave the sounds
    audio_handler = AudioHandler()
    list_sounds_test = []
    for i in range(13):
        sound = output.copy()
        
        # # sound = np.roll(sound, np.random.randint(50)-255)
        
        # if i==13:
        #     sound*=5
        # else:
        #     sound*=2
        #     # pass
        #     # list_sounds_test.append(0*sound)
        list_sounds_test.append(sound)
            
    audio_handler.save_sound_multichannel(list_sounds_test, "latest.npy", apply_rebalancing=True)
    
