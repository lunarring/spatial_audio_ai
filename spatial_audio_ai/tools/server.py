#!/usr/bin/python3
import logging
import numpy as np
import time
import socket
import sys
from numpysocket import NumpySocket

from sound_system import SoundSystem
# import threading # No longer needed for single client

logger = logging.getLogger("sound server")
logger.setLevel(logging.INFO)
sound_system = SoundSystem(logging.WARNING)

def handle_client(conn, addr):
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
                    # logger.info(f"Sound array stats - Min: {np.min(sound_array)}, Max: {np.max(sound_array)}, Mean: {np.mean(sound_array)}, Std: {np.std(sound_array)}")
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

def main():
    with NumpySocket() as s:
        s.bind(("10.40.49.47", 9999))
        s.listen(1)  # Only allow one connection in the backlog
        s.settimeout(1.0)  # Timeout for s.accept() to allow KeyboardInterrupt
        
        print("Server started. Press Ctrl+C to stop.")
        
        try:
            while True:
                print("Waiting for client connection...")
                
                try:
                    # Wait for a client to connect
                    conn, addr = s.accept()
                    
                    # Handle this client (blocking until client disconnects)
                    # conn is also a NumpySocket instance and can be used as a context manager
                    with conn: 
                        handle_client(conn, addr)
                        
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

if __name__ == "__main__":
    main()


