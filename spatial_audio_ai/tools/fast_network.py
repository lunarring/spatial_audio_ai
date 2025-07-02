#!/usr/bin/env python3
"""
Ultra-fast network protocol for real-time audio streaming.

Replaces heavy numpy serialization with lightweight binary protocol
for minimal latency audio transmission.
"""

import socket
import struct
import time
import threading
from typing import Optional, Tuple
import numpy as np
from spatial_audio_ai.config import MAX_QUEUE_SIZE, QUEUE_WARNING_THRESHOLD


class FastAudioSocket:
    """
    Ultra-lightweight socket for real-time audio transmission.
    
    Protocol format:
    - Header: 8 bytes
      - Magic number: 4 bytes (0x46415354 = "FAST")
      - Data length: 4 bytes (uint32, little-endian)
    - Data: Variable length (float32 array, little-endian)
    """
    
    MAGIC_NUMBER = 0x46415354  # "FAST" in hex
    HEADER_SIZE = 8
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set socket options for low latency
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.connected = False
        
    def connect(self, address: Tuple[str, int]):
        """Connect to server."""
        try:
            self.sock.connect(address)
            self.connected = True
            print(f"FastAudioSocket connected to {address}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise
            
    def bind(self, address: Tuple[str, int]):
        """Bind socket for server."""
        self.sock.bind(address)
        
    def listen(self, backlog: int = 1):
        """Listen for connections."""
        self.sock.listen(backlog)
        
    def accept(self) -> Tuple['FastAudioSocket', Tuple[str, int]]:
        """Accept incoming connection."""
        conn_sock, addr = self.sock.accept()
        fast_sock = FastAudioSocket()
        fast_sock.sock = conn_sock
        fast_sock.connected = True
        # Set low latency options on accepted socket
        fast_sock.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return fast_sock, addr
        
    def send_audio(self, data: np.ndarray) -> bool:
        """
        Send audio data with minimal overhead.
        
        Args:
            data: 2D numpy array (channels, samples) of float32
            
        Returns:
            bool: True if sent successfully
        """
        if not self.connected:
            return False
            
        try:
            # Ensure data is float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # Serialize to bytes
            data_bytes = data.tobytes()
            data_length = len(data_bytes)
            
            # Create header
            header = struct.pack('<II', self.MAGIC_NUMBER, data_length)
            
            # Send header + data in one call for efficiency
            message = header + data_bytes
            self.sock.sendall(message)
            return True
            
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False
            
    def receive_audio(self) -> Optional[np.ndarray]:
        """
        Receive audio data.
        
        Returns:
            numpy array or None if error
        """
        if not self.connected:
            return None
            
        try:
            # Receive header
            header_data = self._recv_exactly(self.HEADER_SIZE)
            if not header_data:
                return None
                
            # Parse header
            magic, data_length = struct.unpack('<II', header_data)
            if magic != self.MAGIC_NUMBER:
                print(f"Invalid magic number: {magic:08x}")
                return None
                
            # Receive data
            data_bytes = self._recv_exactly(data_length)
            if not data_bytes:
                return None
                
            # Convert back to numpy array
            # Note: receiver needs to know the shape, we'll handle this
            # in the higher-level protocol
            data = np.frombuffer(data_bytes, dtype=np.float32)
            return data
            
        except Exception as e:
            print(f"Receive error: {e}")
            self.connected = False
            return None
            
    def _recv_exactly(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes."""
        buffer = bytearray()
        while len(buffer) < size:
            try:
                chunk = self.sock.recv(size - len(buffer))
                if not chunk:
                    return None
                buffer.extend(chunk)
            except Exception:
                return None
        return bytes(buffer)
        
    def close(self):
        """Close the socket."""
        if self.connected:
            self.connected = False
            try:
                self.sock.close()
            except Exception:
                pass
                
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FastAudioStreamer:
    """
    High-level audio streamer using FastAudioSocket.
    Handles proper audio formatting and timing.
    """
    
    def __init__(self, host: str = "10.40.49.47", port: int = 9999, 
                 simulate: bool = False):
        self.host = host
        self.port = port
        self.simulate = simulate
        self.socket = None
        self.connected = False
        
        if simulate:
            print("[Simulation] FastAudioStreamer in simulation mode")
        
    def connect(self):
        """Connect to the audio server."""
        if self.simulate:
            self.connected = True
            print(f"[Simulation] Connected to {self.host}:{self.port}")
            return True
            
        try:
            self.socket = FastAudioSocket()
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
            
    def send(self, data: np.ndarray) -> bool:
        """
        Send audio data to server.
        
        Args:
            data: 2D numpy array (channels, samples)
            
        Returns:
            bool: Success status
        """
        if not self.connected:
            print("Not connected - cannot send")
            return False
            
        if self.simulate:
            # Simulate network delay
            time.sleep(0.0001)  # 0.1ms simulated latency
            return True
            
        # Validate data format
        if data.ndim != 2:
            print(f"Invalid data dimensions: {data.shape}")
            return False
            
        # Send using fast protocol
        return self.socket.send_audio(data)
        
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class QueueManagedStreamer(FastAudioStreamer):
    """
    Audio streamer with intelligent queue management to prevent latency buildup.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.stats = {
            'sent': 0,
            'dropped': 0,
            'queue_overruns': 0
        }
        
    def send_with_queue_management(self, data: np.ndarray) -> bool:
        """
        Send audio with intelligent queue management.
        Drops frames if queue builds up to maintain low latency.
        """
        with self.queue_lock:
            queue_size = len(self.send_queue)
            
            # Drop frames if queue is too full
            if queue_size >= MAX_QUEUE_SIZE:
                # Drop oldest frame(s)
                self.send_queue.pop(0)
                self.stats['dropped'] += 1
                self.stats['queue_overruns'] += 1
                print(f"Warning: Dropped frame due to queue overrun "
                      f"(queue: {queue_size})")
                      
            # Add new frame
            self.send_queue.append(data.copy())
            
            # Process queue
            if queue_size >= QUEUE_WARNING_THRESHOLD:
                print(f"Queue warning: {queue_size} frames pending")
                
        # Send immediately (non-blocking approach)
        return self._process_send_queue()
        
    def _process_send_queue(self) -> bool:
        """Process the send queue."""
        with self.queue_lock:
            if not self.send_queue:
                return True
                
            # Send oldest frame
            data = self.send_queue.pop(0)
            
        success = self.send(data)
        if success:
            self.stats['sent'] += 1
        else:
            self.stats['dropped'] += 1
            
        return success
        
    def get_stats(self) -> dict:
        """Get transmission statistics."""
        return self.stats.copy()
        
    def reset_stats(self):
        """Reset statistics."""
        with self.queue_lock:
            self.stats = {'sent': 0, 'dropped': 0, 'queue_overruns': 0} 