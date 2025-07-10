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
        """Receive numpy array using raw binary format."""
        # Receive fixed-size header
        header_data = self._recv_exact(self.HEADER_SIZE)
        if len(header_data) == 0:
            return np.array([])
            
        # Unpack header
        magic, shape, dtype = self._unpack_header(header_data)
        
        # Validate magic number
        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic number: {magic}")
            
        # Calculate data size
        data_size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Receive raw array data
        array_data = self._recv_exact(data_size)
        if len(array_data) != data_size:
            raise ValueError(f"Incomplete data received: {len(array_data)}/{data_size}")
            
        # Reconstruct array
        frame = np.frombuffer(array_data, dtype=dtype).reshape(shape)
        
        logging.debug(f"Fast frame received: shape={shape}, dtype={dtype}")
        return frame

    def _recv_exact(self, size: int) -> bytes:
        """Receive exactly 'size' bytes from socket."""
        data = bytearray()
        while len(data) < size:
            chunk = super().recv(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def _pack_header(self, frame: np.ndarray) -> bytes:
        """Pack array metadata into binary header."""
        # Support up to 2D arrays (typical for audio)
        if frame.ndim == 1:
            shape = (frame.shape[0], 1)
        elif frame.ndim == 2:
            shape = frame.shape
        else:
            raise ValueError(f"Unsupported array dimensions: {frame.ndim}")
            
        # Convert dtype to 4-byte code
        dtype_code = self._dtype_to_code(frame.dtype)
        
        # Pack: magic(4) + shape0(4) + shape1(4) + dtype(4)
        return struct.pack('<4sIII', self.MAGIC, shape[0], shape[1], dtype_code)

    def _unpack_header(self, header_data: bytes) -> tuple[bytes, tuple[int, int], np.dtype]:
        """Unpack binary header to get array metadata."""
        magic, shape0, shape1, dtype_code = struct.unpack('<4sIII', header_data)
        dtype = self._code_to_dtype(dtype_code)
        
        # Convert back to 1D if second dimension is 1
        shape = (shape0,) if shape1 == 1 else (shape0, shape1)
        
        return magic, shape, dtype

    def _dtype_to_code(self, dtype: np.dtype) -> int:
        """Convert numpy dtype to 4-byte code."""
        dtype_map = {
            np.float32: 1,
            np.float64: 2,
            np.int16: 3,
            np.int32: 4,
            np.uint8: 5
        }
        return dtype_map.get(dtype.type, 1)  # Default to float32

    def _code_to_dtype(self, code: int) -> np.dtype:
        """Convert 4-byte code to numpy dtype."""
        code_map = {
            1: np.float32,
            2: np.float64, 
            3: np.int16,
            4: np.int32,
            5: np.uint8
        }
        return np.dtype(code_map.get(code, np.float32))  # Default to float32

    def accept(self) -> tuple["FastNumpySocket", tuple[str, int] | tuple[Any, ...]]:
        fd, addr = super()._accept()  # type: ignore
        sock = FastNumpySocket(super().family, super().type, super().proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr


# Backward compatibility alias
NumpySocket = FastNumpySocket
