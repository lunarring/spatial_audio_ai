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
import random
from scipy import signal


def apply_fade_in_out(sound, sampling_rate=44100, ramp_duration=0.4):
    """
    Apply a fade-in and fade-out effect to the given sound.

    Parameters:
    - sound (np.ndarray): The sound array to apply the fade effect.
    - sampling_rate (int): The sampling rate of the sound.
    - ramp_duration (float): The duration of the fade-in and fade-out in seconds.

    Returns:
    - np.ndarray: The sound array with fade-in and fade-out applied.
    """
    assert isinstance(sound, np.ndarray), "sound must be a numpy ndarray"
    assert sound.ndim == 1, "sound must be a 1-dimensional array"
    ramp_samples = int(ramp_duration * sampling_rate)
    
    # Create linear ramps
    fade_in = np.linspace(0, 1, ramp_samples)
    fade_out = np.linspace(1, 0, ramp_samples)
    
    # Apply fade-in
    sound[:ramp_samples] *= fade_in
    
    # Apply fade-out
    sound[-ramp_samples:] *= fade_out
    
    return sound

def generate_sine_tone(frequency, duration, sampling_rate=44100):
    """
    Generate a sine wave tone.

    Parameters:
    - frequency (float): The frequency of the sine wave in Hz.
    - duration (float): The duration of the sine wave in seconds.
    - sampling_rate (int): The sampling rate in samples per second.

    Returns:
    - np.ndarray: The generated sine wave.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave

def clean_prompt_for_filename(prompt):
    """
    Clean up the given prompt to create a valid filename.

    Parameters:
    - prompt (str): The prompt string to be cleaned.

    Returns:
    - str: A cleaned string suitable for use as a filename.
    """
    filename = prompt.replace(" ", "_")
    filename = filename.replace("'", "")
    filename = filename.replace('"', "")
    filename = filename.replace('.', "")
    return filename

def save_sound(sound: np.ndarray, file_path, sampling_rate=44100):
    """
    Save a sound to a file.

    Parameters:
    - sound (np.ndarray): The sound data to be saved.
    - file_path (str): The full file path where the sound will be saved.
    - sampling_rate (int): The sampling rate of the sound.
    """
    assert isinstance(sound, np.ndarray), "sound must be a numpy ndarray"
    assert sound.ndim in [1, 2], "sound must be a 1-dimensional or 2-dimensional array"
    assert file_path.endswith('.wav'), "File path must end with '.wav'"
    
    # Clip the sound to ensure values are within the valid range
    sound = np.clip(sound, -1, 1)
    
    # Save the sound using soundfile
    sf.write(file_path, sound, sampling_rate)



def save_sound_multichannel(list_sounds, file_path):
    """
    Save a multichannel sound to a file as a numpy array.

    Parameters:
    - list_sounds (list or np.ndarray): The list of sound data to be saved.
    - file_path (str): The full file path where the sound will be saved.
    """
    assert file_path.endswith('.npy'), "File path must end with '.npy'"
    if isinstance(list_sounds, list):
        output = np.array(list_sounds)
    else:
        output = list_sounds

    # Clip the output to ensure values are within the valid range
    output = np.clip(output, -1, 1)

    # Save the output as a numpy file
    np.save(file_path, output)



def apply_reverb(waveform, sampling_rate=44100, decay=0.5):
    """
    Apply a simple reverb effect by convolving the waveform with a generated impulse response.
    
    Parameters:
    - waveform: numpy array, the input sound waveform.
    - sampling_rate: int, the sample rate of the waveform. Defaults to 44100.
    - decay: float, how fast the reverb decays (between 0 and 1, where 0 is no reverb).
    
    Returns:
    - reverb_waveform: numpy array, the waveform with reverb applied.
    """
    # Create a simple impulse response to simulate reverb.
    # A more complex impulse response would provide a more realistic effect.
    impulse_response_length = int(sampling_rate * 0.3)  # 0.3 second reverb tail
    impulse_response = np.zeros(impulse_response_length)
    impulse_response[0] = 1.0  # Direct sound
    for i in range(1, impulse_response_length):
        impulse_response[i] = decay ** i  # Simulated reverb decay

    # Normalize the impulse response to avoid clipping
    impulse_response /= np.max(np.abs(impulse_response))

    # Convolve the input waveform with the impulse response to create the reverb effect
    reverb_waveform = signal.convolve(waveform, impulse_response, mode='full')

    # Normalize the output to avoid clipping
    reverb_waveform /= np.max(np.abs(reverb_waveform))

    # Return the reverb-processed waveform
    return reverb_waveform

