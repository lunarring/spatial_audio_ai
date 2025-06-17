#!/usr/bin/env python3
"""
Utility script to check sample rate for Spatial Audio AI.
Note: System now always uses 48KHz for best quality.

Usage:
    python -m spatial_audio_ai.switch_sample_rate --status
"""

import argparse
import sys
from spatial_audio_ai.config import get_sample_rate, set_sample_rate, SUPPORTED_SAMPLE_RATES

def main():
    parser = argparse.ArgumentParser(description='Check sample rate for Spatial Audio AI')
    parser.add_argument('rate', nargs='?', choices=['44100', '48000'], 
                        help='Legacy parameter (system always uses 48KHz)')
    parser.add_argument('--status', action='store_true', 
                        help='Show current sample rate')
    
    args = parser.parse_args()
    
    current_rate = get_sample_rate()
    print(f"System sample rate: {current_rate} Hz (fixed)")
    
    if args.rate and args.rate != "48000":
        print(f"Note: Requested {args.rate}Hz but system always uses 48KHz for best quality")
    elif args.rate == "48000":
        print("✓ System is already configured for 48KHz")
        
    if not args.status and not args.rate:
        print("System is now fixed at 48KHz - no switching needed!")

if __name__ == "__main__":
    main() 