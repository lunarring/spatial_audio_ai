#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import soundfile as sf
import numpy as np
import sys
import lunar_tools as lt
import torch
import os
from datetime import datetime

class AudioHandler():
    def __init__(self, sampling_rate=44100):
        self.sampling_rate = sampling_rate
        self.set_output_dir("/home/lugo/audio/export")
        # self.channel_scalars_old = np.array([1.0, 0.85, 0.8, 1.0, 0.8, 0.75, 1.0, 0.9, 0.9, 1.0, 0.85, 0.65, 3.0], dtype=np.float32)
        self.channel_scalars = np.array([1.0, 0.85, 0.8, 1.0, 0.8, 0.75, 1.3, 1.0, 1.2, 1.5, 1.3, 0.7, 3.0], dtype=np.float32)
        self.channel_scalars *= 0.5
        self.subwoofer_last_channel_auto_mode = True
        self.nmb_channels = len(self.channel_scalars)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_sound(self, output, filename):
        sf.write(f"{self.output_dir}/{filename}", output, self.sampling_rate)

    def save_sound_multichannel(self, list_sounds, filename, apply_rebalancing=True):
        if isinstance(list_sounds, list):
            output = np.array(list_sounds)
        else:
            output = list_sounds
        if output.shape[0] == self.nmb_channels-1 and self.subwoofer_last_channel_auto_mode:
            # in this case subewoofer was not present in incoming signals and we automatically generate it!
            output = np.append(output, np.expand_dims(np.mean(output, axis=0), axis=0), axis=0)
        elif output.shape[0] == self.nmb_channels:
            pass
        else:
            raise ValueError(f"Unexpected number of channels in the output array: {output.shape[1]}")
            
        output = np.clip(output, -1, 1)
        if apply_rebalancing:
            output * np.expand_dims(self.channel_scalars, axis=1)
        
        np.save(f"{self.output_dir}/{filename}", output)

    def apply_fade_in_out(self, sound, ramp_duration=0.4):
        ramp_samples = int(ramp_duration * self.sampling_rate)
        
        # Create linear ramps
        fade_in = np.linspace(0, 1, ramp_samples)
        fade_out = np.linspace(1, 0, ramp_samples)
        
        # Apply fade-in
        sound[:ramp_samples] *= fade_in
        
        # Apply fade-out
        sound[-ramp_samples:] *= fade_out
        
        return sound

    def generate_calibration_sound(self, duration_single=2, delay_begin=3):
        nmb_samples_single = int(duration_single * self.sampling_rate)
        nmb_samples_delay = int(delay_begin * self.sampling_rate)
    
        # Generate white noise
        white_noise = np.random.normal(0, 1, nmb_samples_single).astype(np.float32)
    
        # Create an empty array for the multichannel sound
        multichannel_noise = np.zeros((self.nmb_channels, self.nmb_channels * nmb_samples_single + nmb_samples_delay), dtype=np.float32)
    
        # Stack the white noise diagonally and multiply by the channel scalars
        for i in range(self.nmb_channels):
            start_index = i * nmb_samples_single + nmb_samples_delay
            end_index = start_index + nmb_samples_single
            multichannel_noise[i, start_index:end_index] = white_noise * self.channel_scalars[i]
    
        # Save the white noise to a file for verification
        self.save_sound_multichannel(multichannel_noise, "latest.npy", apply_rebalancing=False)