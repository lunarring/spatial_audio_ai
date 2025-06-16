#!/usr/bin/env python3
"""
Utility script to easily switch between sample rates for Spatial Audio AI.

Usage:
    python -m spatial_audio_ai.switch_sample_rate 48000
    python -m spatial_audio_ai.switch_sample_rate 44100
    python -m spatial_audio_ai.switch_sample_rate --status
"""

import argparse
import sys
from spatial_audio_ai.config import get_sample_rate, set_sample_rate, SUPPORTED_SAMPLE_RATES

def main():
    parser = argparse.ArgumentParser(description='Switch sample rate for Spatial Audio AI')
    parser.add_argument('rate', nargs='?', choices=['44100', '48000'], 
                        help='Sample rate to set (44100 or 48000)')
    parser.add_argument('--status', action='store_true', 
                        help='Show current sample rate')
    
    args = parser.parse_args()
    
    if args.status or args.rate is None:
        current_rate = get_sample_rate()
        print(f"Current sample rate: {current_rate} Hz")
        if not args.status:
            print("Available rates:", list(SUPPORTED_SAMPLE_RATES.keys()))
        return
    
    try:
        set_sample_rate(args.rate)
        print(f"Sample rate set to {args.rate} Hz for this session")
        print("Note: This only affects the current session.")
        print(f"To make it permanent, edit DEFAULT_SAMPLE_RATE in config.py to '{args.rate}'")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 