#!/usr/bin/python3
import logging
import time as time_module
import socket
import sys
import numpy as np
import struct  # For parsing custom UDP headers
import threading
import json
from spatial_audio_ai.config import UDP_BUFFER_DEPTH, get_profile_queue_depth, ALLOWED_PROFILES, get_min_buffer_blocks
from spatial_audio_ai.tools.numpysocket import FastNumpySocket
from spatial_audio_ai.tools.sound_system import SoundSystem

try:
    import lunar_tools as lt
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("Warning: lunar_tools not available. ZMQ support disabled.")

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
                    error_msg = f"[SERVER][MALFORMED] Received array with unexpected shape {sound_array.shape} from {addr}. Disconnecting."
                    logger.warning(error_msg)
                    print(error_msg)
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


# Control message magic for profile selection
CONTROL_MAGIC = b'NPCC'

def handle_zmq_client(zmq_server, sound_system, verbose=False):
    """Handle ZMQ client messages in a separate thread"""
    logger = logging.getLogger("zmq_handler")
    zmq_seq_map = {}
    message_count = 0
    
    logger.info("ZMQ handler started")
    print("[ZMQ][HANDLER] 🚀 ZMQ handler thread started and running")
    print(f"[ZMQ][HANDLER] ZMQ server object: {zmq_server}")
    print(f"[ZMQ][HANDLER] Verbose mode: {verbose}")
    
    loop_count = 0
    last_log_time = time_module.perf_counter()
    
    while True:
        try:
            loop_count += 1
            
            # Check for messages (non-blocking)
            messages = zmq_server.get_messages()
            
            # Log every 5 seconds to show the handler is alive
            current_time = time_module.perf_counter()
            if verbose and current_time - last_log_time > 5.0:
                print(f"[ZMQ][HANDLER] ❤️ Handler alive - loop_count: {loop_count}, message_count: {message_count}")
                last_log_time = current_time
            
            if not messages:
                time_module.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            if verbose:
                print(f"[ZMQ][HANDLER] 📨 Received {len(messages)} messages")
                
            for msg in messages:
                try:
                    message_count += 1
                    if verbose:
                        print(f"[ZMQ][HANDLER] 🔍 Processing message #{message_count}: {type(msg)}")
                        print(f"[ZMQ][HANDLER] 🔍 Message keys: {list(msg.keys()) if isinstance(msg, dict) else 'Not a dict'}")
                    
                    # Handle control messages
                    if 'control' in msg:
                        profile = msg['control'].get('profile', 'stable')
                        client_id = msg.get('client_id', 'unknown')
                        
                        # Validate profile and get depth from centralized config
                        if profile not in ALLOWED_PROFILES:
                            profile = 'stable'  # Fallback to stable for invalid profiles
                        
                        depth = get_profile_queue_depth(profile, protocol='zmq')
                        min_buffer = get_min_buffer_blocks(profile)
                        sound_system.set_max_queue_depth(depth)
                        sound_system.set_min_buffer_blocks(min_buffer)
                        
                        log_msg = f"[ZMQ][PROFILE] ✅ client={client_id} profile={profile} -> queue_depth={depth}, min_buffer={min_buffer}"
                        logger.info(log_msg)
                        print(log_msg)
                        
                        # Send acknowledgment back to client
                        ack_msg = {
                            "client_id": client_id,
                            "control_ack": {
                                "profile": profile,
                                "queue_depth": depth,
                                "min_buffer_blocks": min_buffer,
                                "status": "connected",
                                "server_timestamp": time_module.perf_counter()
                            }
                        }
                        try:
                            zmq_server.send_json(ack_msg)
                            print(f"[ZMQ][ACK] ✅ Sent acknowledgment to client {client_id}")
                        except Exception as e:
                            print(f"[ZMQ][ACK] ❌ Failed to send acknowledgment: {e}")
                        
                        continue
                    
                    # Handle audio data messages
                    if 'audio_data' in msg:
                        audio_info = msg['audio_data']
                        client_id = msg.get('client_id', 'unknown')
                        seq = audio_info.get('seq', 0)
                        timestamp = audio_info.get('timestamp', time_module.perf_counter())
                        
                        # Reconstruct numpy array from JSON
                        array_data = np.array(audio_info['data'], dtype=audio_info['dtype'])
                        frame = array_data.reshape(audio_info['shape'])
                        
                        # Sequence checking (similar to UDP)
                        last_seq = zmq_seq_map.get(client_id)
                        if last_seq is not None:
                            if seq > last_seq + 1:
                                lost = seq - last_seq - 1
                                error_msg = f"[ZMQ][LOSS] client={client_id} lost={lost} frames (seq {last_seq+1}-{seq-1})"
                                logger.warning(error_msg)
                                print(error_msg)
                            elif seq <= last_seq:
                                error_msg = f"[ZMQ][REORDER] client={client_id} seq={seq} <= last_seq={last_seq}"
                                logger.warning(error_msg)
                                print(error_msg)
                        zmq_seq_map[client_id] = seq
                        
                        # Add to playback queue
                        sound_system.add_to_playback_queue(frame, seq)
                        if verbose:
                            print(f"[ZMQ][AUDIO] ✅ Processed audio seq={seq} from client={client_id}, shape={frame.shape}")
                        
                        if verbose and seq % 50 == 0:  # Log occasionally
                            print(f"[ZMQ][AUDIO] 📊 Audio stats - seq={seq} from client={client_id}")
                            
                except Exception as e:
                    logger.error(f"[ZMQ][ERROR] ❌ Error processing message: {e}")
                    print(f"[ZMQ][ERROR] ❌ Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                        
        except Exception as e:
            logger.error(f"[ZMQ][ERROR] ❌ Handler error: {e}")
            print(f"[ZMQ][ERROR] ❌ Handler error: {e}")
            import traceback
            traceback.print_exc()
            time_module.sleep(0.1)  # Longer sleep on error

class SoundServer:
    """Sound server class that can be used to start a server instance"""
    
    def __init__(
        self, 
        host="10.40.49.47", 
        port=9999, 
        zmq_port=5556,
        log_level=logging.WARNING,
        mock_mode=False,
        verbose=False,
        enable_zmq=True
    ):
        self.host = host
        self.port = port
        self.zmq_port = zmq_port
        self.log_level = log_level
        self.mock_mode = mock_mode
        self.verbose = verbose
        self.enable_zmq = enable_zmq and ZMQ_AVAILABLE
        self.zmq_server = None
        self.zmq_clients = set()
        self.zmq_seq_map = {}
        self._stop_event = threading.Event()
    
    def start(self):
        """Start the dual-protocol sound server (UDP + ZMQ)"""
        # Only initialize SoundSystem when server is started
        sound_system = SoundSystem(self.log_level, mock_mode=self.mock_mode, verbose=self.verbose)

        # Initialize ZMQ server if enabled
        if self.enable_zmq:
            try:
                print(f"[ZMQ][INIT] Attempting to start ZMQ server...")
                print(f"[ZMQ][INIT] lunar_tools available: {ZMQ_AVAILABLE}")
                self.zmq_server = lt.ZMQPairEndpoint(is_server=True, ip=self.host, port=str(self.zmq_port))
                print(f"[ZMQ][INIT] ✅ ZMQ server created successfully on {self.host}:{self.zmq_port}")
                
                # Start ZMQ handler in separate thread
                zmq_thread = threading.Thread(
                    target=handle_zmq_client, 
                    args=(self.zmq_server, sound_system, self.verbose),
                    daemon=True,
                    name="ZMQ-Handler"
                )
                zmq_thread.start()
                print(f"[ZMQ][INIT] ✅ ZMQ handler thread started (thread: {zmq_thread.name})")
                
                # Give the thread a moment to start
                time_module.sleep(0.1)
                if zmq_thread.is_alive():
                    print(f"[ZMQ][INIT] ✅ ZMQ handler thread is running")
                else:
                    print(f"[ZMQ][INIT] ❌ ZMQ handler thread failed to start")
                    
            except Exception as e:
                print(f"[ZMQ][INIT] ❌ Failed to start ZMQ server: {e}")
                import traceback
                traceback.print_exc()
                self.enable_zmq = False

        # Use UDP socket for streaming with sequence numbers and simple jitter handling
        with FastNumpySocket(type=socket.SOCK_DGRAM) as s:
            s.bind((self.host, self.port))
            s.settimeout(1.0)  # Timeout to allow graceful shutdown

            protocols = ["UDP"]
            if self.enable_zmq:
                protocols.append("ZMQ")
            print(f"Server started with {'/'.join(protocols)} on {self.host}:{self.port}" + 
                  (f" (ZMQ: {self.zmq_port})" if self.enable_zmq else "") + ". Press Ctrl+C to stop.")
            clients = set()
            last_seq_map = {}

            try:
                while True:
                    try:
                        data, addr = s.recvfrom(65536)  # Receive UDP datagram
                        # Handle control packets for profile configuration
                        if data.startswith(CONTROL_MAGIC):
                            profile = data[len(CONTROL_MAGIC):].decode().strip()
                            if addr not in clients:
                                clients.add(addr)
                                last_seq_map[addr] = None
                            
                            # Validate profile and get depth from centralized config
                            if profile not in ALLOWED_PROFILES:
                                profile = 'balanced'  # Fallback to balanced for invalid UDP profiles
                            
                            depth = get_profile_queue_depth(profile, protocol='udp')
                            min_buffer = get_min_buffer_blocks(profile)
                            sound_system.set_max_queue_depth(depth)
                            sound_system.set_min_buffer_blocks(min_buffer)
                            msg = f"[SERVER][PROFILE] client={addr} profile={profile} -> queue_depth={depth}, min_buffer={min_buffer}"
                            logger.info(msg)
                            print(msg)
                            continue
                        # New audio client registration when no control packet
                        if addr not in clients:
                            clients.add(addr)
                            last_seq_map[addr] = None
                            msg = f"[SERVER][JOIN] client={addr}"  # client connected
                            logger.info(msg)
                            print(msg)
                        # Extract and parse custom header
                        raw_header = data[:s.HEADER_SIZE]
                        try:
                            magic, seq, timestamp, shape0, shape1, dtype_code = struct.unpack('<4sQdIII', raw_header)
                        except Exception as e:
                            error_msg = f"[SERVER][MALFORMED] bad header from {addr}: {e}"
                            logger.warning(error_msg)
                            print(error_msg)
                            continue
                        # Validate magic
                        if magic != FastNumpySocket.MAGIC:
                            error_msg = f"[SERVER][MALFORMED] invalid magic from {addr}: {magic}"
                            logger.warning(error_msg)
                            print(error_msg)
                            continue
                        # Determine shape and dtype
                        shape = (shape0,) if shape1 == 1 else (shape0, shape1)
                        dtype = s._code_to_dtype(dtype_code)
                        # Sequence order checks
                        last_seq = last_seq_map[addr]
                        if last_seq is not None:
                            if seq > last_seq + 1:
                                lost = seq - last_seq - 1
                                error_msg = f"[SERVER][LOSS] client={addr} lost={lost} frames (seq {last_seq+1}-{seq-1})"
                                logger.warning(error_msg)
                                print(error_msg)
                            elif seq <= last_seq:
                                error_msg = f"[SERVER][REORDER] client={addr} seq={seq} <= last_seq={last_seq}"
                                logger.warning(error_msg)
                                print(error_msg)
                        last_seq_map[addr] = seq
                        # Validate payload length
                        payload = data[s.HEADER_SIZE:]
                        expected = np.prod(shape) * dtype.itemsize
                        if len(payload) < expected:
                            error_msg = f"[SERVER][INCOMPLETE] expected={expected} bytes but got={len(payload)} from {addr}"
                            logger.warning(error_msg)
                            print(error_msg)
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
    parser.add_argument('--port', type=int, default=9999, help='UDP port number')
    parser.add_argument('--zmq-port', type=int, default=5556, help='ZMQ port number (default: 5556)')
    parser.add_argument('--no-zmq', action='store_true', help='Disable ZMQ support (UDP only)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for client connections')
    args = parser.parse_args()
    
    server = SoundServer(
        host=args.host, 
        port=args.port, 
        zmq_port=args.zmq_port,
        mock_mode=args.mock, 
        verbose=args.verbose,
        enable_zmq=not args.no_zmq
    )
    server.start()


if __name__ == "__main__":
    main()


