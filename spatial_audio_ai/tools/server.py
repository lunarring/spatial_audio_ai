#!/usr/bin/python3
import logging
import time
import socket
import sys
import numpy as np
from spatial_audio_ai.tools.numpysocket import NumpySocket
from spatial_audio_ai.tools.sound_system import SoundSystem
# import threading # No longer needed for single client

# Create logger
logger = logging.getLogger("sound server")
logger.setLevel(logging.INFO)

def handle_client(conn, addr, sound_system):
    """Handle a single client connection"""
    try:
        logger.info(f"Connected: {addr}")
        print(f"Client connected from {addr}")
        
        while True:
            try:
                sound_array = conn.recv()
                
                # Check for disconnection or empty message
                if sound_array is None or not sound_array.size:
                    logger.info(f"Client {addr} disconnected or sent empty data.")
                    break # Exit the loop for this client
                    
                if len(sound_array.shape) == 2:
                    logger.info(f"Received sound array with shape {sound_array.shape}")
                    logger.info(f"Sound array stats - Min: {np.min(sound_array)}, Max: {np.max(sound_array)}, Mean: {np.mean(sound_array)}, Std: {np.std(sound_array)}")
                    sound_system.add_to_playback_queue(sound_array)
                else:
                    logger.warning(f"Received array with unexpected shape {sound_array.shape} from {addr}. Disconnecting.")
                    break # Optional: disconnect on malformed data
            except socket.timeout: # If conn had a timeout
                logger.info(f"Socket timeout waiting for data from {addr}. Assuming disconnect.")
                break
            except Exception as e:
                logger.info(f"Connection error with {addr}: {e}")
                break # Exit loop on other errors
                
        logger.info(f"Disconnected: {addr}")
        print(f"Client disconnected from {addr}")
    except Exception as e: # Catch-all for unexpected errors in handle_client setup/teardown
        logger.error(f"Unexpected error in handle_client for {addr}: {e}", exc_info=True)
        print(f"Error handling client {addr}. Check logs.")
    # 'with conn:' in main ensures conn.close() is called


class SoundServer:
    """Sound server class that can be used to start a server instance"""
    
    def __init__(
        self, 
        host="10.40.49.47", 
        port=9999, 
        log_level=logging.WARNING,
        mock_mode=False
    ):
        self.host = host
        self.port = port
        self.log_level = log_level
        self.mock_mode = mock_mode
    
    def start(self):
        """Start the sound server"""
        # Only initialize SoundSystem when server is started
        sound_system = SoundSystem(self.log_level, mock_mode=self.mock_mode)
        
        with NumpySocket() as s:
            s.bind((self.host, self.port))
            s.listen(1)  # Only allow one connection in the backlog
            s.settimeout(1.0)  # Timeout for s.accept() to allow KeyboardInterrupt
            
            print(f"Server started on {self.host}:{self.port}. Press Ctrl+C to stop.")
            
            try:
                while True:
                    print("Waiting for client connection...")
                    
                    try:
                        # Wait for a client to connect
                        conn, addr = s.accept()
                        
                        # Handle this client (blocking until client disconnects)
                        with conn: 
                            handle_client(conn, addr, sound_system)
                            
                    except socket.timeout:
                        # This is for s.accept() timeout, just continue the loop
                        continue
                    except Exception as e:
                        logger.error(f"Error during accept or client handling setup: {e}")
                        # Decide if server should continue or stop on such errors
                        # For now, let's print and continue listening, but could also break
                        print(f"An error occurred: {e}. Server continues listening.")
                        time.sleep(1) # Avoid fast error loop
                        
            except KeyboardInterrupt:
                print("\nServer shutting down gracefully...")
            except Exception as e:
                # Catch other unexpected errors in the main server loop
                print(f"\nCritical server error: {e}")
                logger.critical(f"Critical server error: {e}", exc_info=True)
            finally:
                print("Cleaning up resources...")
                # s.close() is handled by 'with NumpySocket() as s:'
                print("Server stopped.")
                sys.exit(0)


def main():
    """Main function to run the server directly"""
    import argparse
    parser = argparse.ArgumentParser(description='Start the spatial audio server')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no hardware required)')
    parser.add_argument('--host', default="10.40.49.47", help='Host address')
    parser.add_argument('--port', type=int, default=9999, help='Port number')
    args = parser.parse_args()
    
    server = SoundServer(host=args.host, port=args.port, mock_mode=args.mock)
    server.start()


if __name__ == "__main__":
    main()


