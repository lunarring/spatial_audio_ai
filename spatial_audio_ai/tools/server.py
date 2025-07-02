#!/usr/bin/env python3
"""
Ultra-low latency audio server using FastAudioSocket protocol.

Replaces the heavy numpy serialization server with a lightweight
binary protocol for minimal latency audio reception and playback.
"""

import logging
import time
import sys
import numpy as np
from spatial_audio_ai.tools.fast_network import FastAudioSocket
from spatial_audio_ai.tools.sound_system import SoundSystem
from spatial_audio_ai.config import N_SPEAKERS, BLOCKSIZE

# Create logger
logger = logging.getLogger("fast_sound_server")
logger.setLevel(logging.INFO)


def handle_fast_client(conn, addr, sound_system, verbose=False):
    """Handle a single client connection using fast protocol"""
    try:
        logger.info(f"Connected: {addr}")
        if verbose:
            print(f"Fast client connected from {addr}")
        
        frames_received = 0
        
        while True:
            try:
                # Receive audio data using fast protocol
                audio_data = conn.receive_audio()
                
                # Check for disconnection or empty message
                if audio_data is None or not audio_data.size:
                    logger.info(f"Client {addr} disconnected or sent empty data.")
                    break
                
                # Reshape the flat array back to 2D (channels, samples)
                # Expected format: (N_SPEAKERS, BLOCKSIZE)
                expected_size = N_SPEAKERS * BLOCKSIZE
                if audio_data.size != expected_size:
                    logger.warning(
                        f"Unexpected audio data size {audio_data.size}, "
                        f"expected {expected_size} from {addr}"
                    )
                    continue
                
                # Reshape to (channels, samples)
                sound_array = audio_data.reshape(N_SPEAKERS, BLOCKSIZE)
                
                frames_received += 1
                
                if verbose and frames_received % 100 == 0:
                    logger.info(f"Received {frames_received} frames from {addr}")
                    logger.info(
                        f"Audio stats - Min: {np.min(sound_array):.3f}, "
                        f"Max: {np.max(sound_array):.3f}, "
                        f"Mean: {np.mean(sound_array):.3f}"
                    )
                
                # Add to playback queue
                sound_system.add_to_playback_queue(sound_array)
                
            except Exception as e:
                logger.error(
                    f"Unexpected error in handle_fast_client for {addr}: {e}", 
                    exc_info=True
                )
                print(f"Error handling fast client {addr}. Check logs.")
                break
                
        logger.info(f"Disconnected: {addr} (received {frames_received} frames)")
        if verbose:
            print(f"Fast client disconnected from {addr}")
            
    except Exception as e:
        logger.error(
            f"Unexpected error in handle_fast_client for {addr}: {e}", 
            exc_info=True
        )
        print(f"Error handling fast client {addr}. Check logs.")


class SoundServer:
    """Ultra-low latency sound server using fast audio protocol"""
    
    def __init__(
        self, 
        host="10.40.49.47", 
        port=9999, 
        log_level=logging.WARNING,
        mock_mode=False,
        verbose=False
    ):
        self.host = host
        self.port = port
        self.log_level = log_level
        self.mock_mode = mock_mode
        self.verbose = verbose
        self.server_socket = None
    
    def start(self):
        """Start the fast sound server"""
        # Initialize SoundSystem when server is started
        sound_system = SoundSystem(self.log_level, mock_mode=self.mock_mode)
        
        self.server_socket = FastAudioSocket()
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)  # Only allow one connection
            
            print(f"Fast Sound Server started on {self.host}:{self.port}")
            print("Press Ctrl+C to stop.")
            
            while True:
                if self.verbose:
                    print("Waiting for client connection...")
                
                try:
                    # Wait for a client to connect
                    conn, addr = self.server_socket.accept()
                    
                    # Handle this client (blocking until client disconnects)
                    with conn: 
                        handle_fast_client(conn, addr, sound_system, self.verbose)
                        
                except Exception as e:
                    logger.error(f"Error during accept or client handling: {e}")
                    print(f"An error occurred: {e}. Server continues listening.")
                    time.sleep(1)  # Avoid fast error loop
                        
        except KeyboardInterrupt:
            print("\nFast Sound Server shutting down gracefully...")
        except Exception as e:
            print(f"\nCritical server error: {e}")
            logger.critical(f"Critical server error: {e}", exc_info=True)
        finally:
            print("Cleaning up resources...")
            if self.server_socket:
                self.server_socket.close()
            print("Fast Sound Server stopped.")
            sys.exit(0)


def main():
    """Main function to run the fast server directly"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Start the ultra-low latency spatial audio server'
    )
    parser.add_argument(
        '--mock', action='store_true', 
        help='Run in mock mode (no hardware required)'
    )
    parser.add_argument('--host', default="10.40.49.47", help='Host address')
    parser.add_argument('--port', type=int, default=9999, help='Port number')
    parser.add_argument(
        '--verbose', action='store_true', 
        help='Enable verbose output for client connections'
    )
    args = parser.parse_args()
    
    print("Starting Ultra-Low Latency Audio Server")
    print(f"Block size: {BLOCKSIZE} samples (~{BLOCKSIZE/48000*1000:.1f}ms)")
    
    server = SoundServer(
        host=args.host, 
        port=args.port, 
        mock_mode=args.mock, 
        verbose=args.verbose
    )
    server.start()


if __name__ == "__main__":
    main()


