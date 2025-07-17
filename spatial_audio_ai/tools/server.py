#!/usr/bin/python3
import logging
import time as time_module
import socket
import sys
import numpy as np
import struct  # For parsing custom UDP headers
from spatial_audio_ai.tools.numpysocket import FastNumpySocket
from spatial_audio_ai.tools.sound_system import SoundSystem
# import threading # No longer needed for single client

# Create logger
logger = logging.getLogger("sound server")
logger.setLevel(logging.INFO)

def handle_client(conn, addr, sound_system, verbose=False):
    """Handle a single client connection"""
    chunk_counter = 0
    try:
        logger.info(f"Connected: {addr}")
        if verbose:
            print(f"Client connected from {addr}")
        
        while True:
            try:
                t0 = time_module.perf_counter()
                sound_array = conn.recv()
                t1 = time_module.perf_counter()
                
                # Only log timing occasionally (every 50 chunks ≈ 1 second)
                chunk_counter += 1
                if chunk_counter % 50 == 0:
                    print(f"[SERVER] recv+unpack: {t1-t0:.3f}s (chunk {chunk_counter})")
                
                # Check for disconnection or empty message
                if sound_array is None or not sound_array.size:
                    logger.info(f"Client {addr} disconnected or sent empty data.")
                    break # Exit the loop for this client
                    
                if len(sound_array.shape) == 2:
                    if verbose:
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
        if verbose:
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
        mock_mode=False,
        verbose=False
    ):
        self.host = host
        self.port = port
        self.log_level = log_level
        self.mock_mode = mock_mode
        self.verbose = verbose
    
    def start(self):
        """Start the sound server"""
        # Only initialize SoundSystem when server is started
        sound_system = SoundSystem(self.log_level, mock_mode=self.mock_mode)

        # Use UDP socket for streaming with sequence numbers and simple jitter handling
        with FastNumpySocket(type=socket.SOCK_DGRAM) as s:
            s.bind((self.host, self.port))
            s.settimeout(1.0)  # Timeout to allow graceful shutdown

            print(f"UDP server started on {self.host}:{self.port}. Press Ctrl+C to stop.")
            clients = set()
            last_seq_map = {}

            try:
                while True:
                    try:
                        data, addr = s.recvfrom(65536)  # Receive UDP datagram
                        # New client registration
                        if addr not in clients:
                            clients.add(addr)
                            last_seq_map[addr] = None
                            logger.info(f"[SERVER][JOIN] client={addr}")
                            if self.verbose:
                                print(f"[SERVER][JOIN] client={addr}")
                        # Extract and parse custom header
                        raw_header = data[:s.HEADER_SIZE]
                        try:
                            magic, seq, timestamp, shape0, shape1, dtype_code = struct.unpack('<4sQdIII', raw_header)
                        except Exception as e:
                            logger.warning(f"[SERVER][MALFORMED] bad header from {addr}: {e}")
                            continue
                        # Validate magic
                        if magic != FastNumpySocket.MAGIC:
                            logger.warning(f"[SERVER][MALFORMED] invalid magic from {addr}: {magic}")
                            continue
                        # Determine shape and dtype
                        shape = (shape0,) if shape1 == 1 else (shape0, shape1)
                        dtype = s._code_to_dtype(dtype_code)
                        # Sequence order checks
                        last_seq = last_seq_map[addr]
                        if last_seq is not None:
                            if seq > last_seq + 1:
                                lost = seq - last_seq - 1
                                logger.warning(f"[SERVER][LOSS] client={addr} lost={lost} frames (seq {last_seq+1}-{seq-1})")
                            elif seq <= last_seq:
                                logger.warning(f"[SERVER][REORDER] client={addr} seq={seq} <= last_seq={last_seq}")
                        last_seq_map[addr] = seq
                        # Validate payload length
                        payload = data[s.HEADER_SIZE:]
                        expected = np.prod(shape) * dtype.itemsize
                        if len(payload) < expected:
                            logger.warning(f"[SERVER][INCOMPLETE] expected={expected} bytes but got={len(payload)} from {addr}")
                            continue
                        # Reconstruct and enqueue
                        frame = np.frombuffer(payload[:expected], dtype=dtype).reshape(shape)
                        sound_system.add_to_playback_queue(frame, seq)
                    except socket.timeout:
                        continue
            except KeyboardInterrupt:
                print("\nServer shutting down gracefully...")
            except Exception as e:
                print(f"\nCritical server error: {e}")
                logger.critical(f"Critical server error: {e}", exc_info=True)
            finally:
                print("Cleaning up resources...")
                print("Server stopped.")
                sys.exit(0)


def main():
    """Main function to run the server directly"""
    import argparse
    parser = argparse.ArgumentParser(description='Start the spatial audio server')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no hardware required)')
    parser.add_argument('--host', default="10.40.49.47", help='Host address')
    parser.add_argument('--port', type=int, default=9999, help='Port number')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for client connections')
    args = parser.parse_args()
    
    server = SoundServer(host=args.host, port=args.port, mock_mode=args.mock, verbose=args.verbose)
    server.start()


if __name__ == "__main__":
    main()


