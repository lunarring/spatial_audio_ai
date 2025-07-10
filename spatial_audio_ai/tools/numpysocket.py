#!/usr/bin/env python3

from io import BytesIO
import logging
import socket
import struct
from typing import Any

import numpy as np


class FastNumpySocket(socket.socket):
    """Optimized socket for real-time audio streaming with raw binary transmission."""
    
    # Protocol constants
    MAGIC = b'NPSF'  # NumpySocket Fast
    HEADER_SIZE = 16  # Magic(4) + Shape(8) + DType(4)
    
    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        super().__init__(family, type, proto, fileno)
        
        # Optimize for real-time audio streaming
        if type == socket.SOCK_STREAM:  # TCP optimizations
            # Disable Nagle's algorithm for low latency
            self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Reasonable buffer sizes for fewer system calls
            self.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # 64KB receive buffer  
            self.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # 64KB send buffer
            
            # Enable keep-alive for connection stability  
            self.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # Conservative low latency settings (where supported)
            try:
                # Set higher priority (Linux specific) 
                self.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 4)  # Medium-high priority
            except (OSError, AttributeError):
                pass  # Not supported on this platform

    def sendall(self, frame: np.ndarray) -> None:  # type: ignore[override]
        """Send numpy array using raw binary format for minimal latency."""
        # Ensure contiguous array for efficient transmission
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
            
        # Pack header: magic + shape + dtype
        header = self._pack_header(frame)
        
        # Send header first
        super().sendall(header)
        
        # Send raw array data
        super().sendall(frame.tobytes())
        
        logging.debug(f"Fast frame sent: shape={frame.shape}, dtype={frame.dtype}")

    def recv(self, bufsize: int = 8192) -> np.ndarray:  # type: ignore[override]
        """Receive numpy array using raw binary format with stable processing."""
        # Receive fixed-size header
        header_data = self._recv_exact(self.HEADER_SIZE)
        if len(header_data) == 0:
            return np.array([])
            
        # Unpack header (optimized struct unpack)
        magic, shape0, shape1, dtype_code = struct.unpack('<4sIII', header_data)
        
        # Fast validation
        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic number: {magic}")
        
        # Convert back to 1D if second dimension is 1
        shape = (shape0,) if shape1 == 1 else (shape0, shape1)
        dtype = self._code_to_dtype(dtype_code)
            
        # Calculate data size
        data_size = shape0 * shape1 * dtype.itemsize
        
        # Receive raw array data using reliable chunked method
        array_data = self._recv_exact(data_size)
        if len(array_data) != data_size:
            raise ValueError(f"Incomplete data received: {len(array_data)}/{data_size}")
            
        # Fast reconstruction using frombuffer (most efficient)
        frame = np.frombuffer(array_data, dtype=dtype)
        if len(shape) > 1:
            frame = frame.reshape(shape)
        
        return frame

    def _recv_exact(self, size: int) -> bytes:
        """Receive exactly 'size' bytes from socket with robust error handling."""
        # Pre-allocate buffer for better performance
        data = bytearray(size)
        view = memoryview(data)
        pos = 0
        
        while pos < size:
            # Try to receive remaining bytes in larger chunks
            remaining = size - pos
            chunk_size = min(remaining, 65536)  # Match socket buffer size
            
            try:
                bytes_received = super().recv_into(view[pos:pos + chunk_size])
                if not bytes_received:
                    # Connection closed by peer
                    raise ConnectionResetError("Connection closed by peer during receive")
                pos += bytes_received
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                # Re-raise connection errors for upper layers to handle
                raise
            except socket.timeout:
                # Handle timeout gracefully
                raise socket.timeout("Socket timeout during receive")
            except Exception as e:
                # Handle other socket errors
                raise ConnectionError(f"Socket error during receive: {e}")
            
        return bytes(data[:pos])

    def _pack_header(self, frame: np.ndarray) -> bytes:
        """Pack array metadata into binary header with optimized encoding."""
        shape = frame.shape
        dtype_code = self._dtype_to_code(frame.dtype.type)
        
        # Convert to standard 2D shape representation
        if len(shape) == 1:
            shape0, shape1 = shape[0], 1
        else:
            shape0, shape1 = shape[0], shape[1]
            
        return struct.pack('<4sIII', self.MAGIC, shape0, shape1, dtype_code)

    def _unpack_header(self, header_data: bytes) -> tuple[bytes, tuple[int, int], np.dtype]:
        """Unpack binary header to get array metadata."""
        magic, shape0, shape1, dtype_code = struct.unpack('<4sIII', header_data)
        dtype = self._code_to_dtype(dtype_code)
        
        # Convert back to 1D if second dimension is 1
        shape = (shape0,) if shape1 == 1 else (shape0, shape1)
        
        return magic, shape, dtype

    # Optimized dtype mapping for faster lookups
    _DTYPE_CODES = {
        np.float32: 1,
        np.float64: 2, 
        np.int32: 3,
        np.int64: 4
    }
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODES.items()}
    
    def _dtype_to_code(self, dtype) -> int:
        """Convert numpy dtype to integer code for fast transmission."""
        return self._DTYPE_CODES.get(dtype, 1)  # Default to float32
    
    def _code_to_dtype(self, code: int):
        """Convert integer code back to numpy dtype."""
        return self._CODE_TO_DTYPE.get(code, np.float32)  # Default to float32

    def accept(self) -> tuple["FastNumpySocket", tuple[str, int] | tuple[Any, ...]]:
        fd, addr = super()._accept()  # type: ignore
        sock = FastNumpySocket(super().family, super().type, super().proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr


# Backward compatibility alias
NumpySocket = FastNumpySocket
