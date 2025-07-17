#!/usr/bin/env python3

from io import BytesIO
import logging
import socket
import struct
from typing import Any

import numpy as np
import time  # Add timestamp support
import errno  # For EMSGSIZE handling


class FastNumpySocket(socket.socket):
    """Optimized socket for real-time audio streaming with raw binary transmission."""
    
    # Protocol constants
    MAGIC = b'NPSF'  # NumpySocket Fast
    HEADER_SIZE = 32  # Magic(4) + Seq(8) + Timestamp(8) + Shape0(4) + Shape1(4) + DType(4)
    
    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        super().__init__(family, type, proto, fileno)
        self._seq = 0  # Initialize sequence counter
        
        # Optimize for real-time audio streaming
        if type == socket.SOCK_STREAM:  # TCP optimizations
            # Disable Nagle's algorithm for low latency
            self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Set optimal buffer sizes for audio chunks (~13KB for 13 speakers * 1024 samples * 4 bytes)
            self.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # 64KB receive buffer
            self.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # 64KB send buffer
            
            # Enable keep-alive for connection stability
            self.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def sendall(self, frame: np.ndarray) -> None:  # type: ignore[override]
        """Send numpy array using raw binary transmission with MTU-based fragmentation for UDP."""
        # Downcast to float32 for UDP to reduce payload size
        if self.type == socket.SOCK_DGRAM and frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        # Ensure contiguous array for efficient transmission
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        if self.type == socket.SOCK_DGRAM:
            # UDP: fragment into MTU-sized datagrams
            MTU = 1500  # Typical Ethernet MTU; adjust if needed
            header_size = self.HEADER_SIZE
            max_payload = MTU - header_size
            elem_size = frame.dtype.itemsize
            # Determine channel count and time-length
            if frame.ndim == 1:
                n_channels = 1
                time_len = frame.shape[0]
            elif frame.ndim == 2:
                n_channels, time_len = frame.shape
            else:
                raise ValueError(f"Unsupported array dimensions: {frame.ndim}")
            # Samples per packet based on payload limit
            spp = max(1, max_payload // (n_channels * elem_size))
            # Send each fragment
            for start in range(0, time_len, spp):
                if frame.ndim == 1:
                    sub = frame[start:start + spp]
                else:
                    sub = frame[:, start:start + spp]
                # Pad if last fragment is smaller
                if frame.ndim == 1:
                    if sub.shape[0] < spp:
                        sub = np.pad(sub, (0, spp - sub.shape[0]), 'constant')
                else:
                    if sub.shape[1] < spp:
                        pad = ((0, 0), (0, spp - sub.shape[1]))
                        sub = np.pad(sub, pad, 'constant')
                # Pack header and send fragment
                header = self._pack_header(sub)
                packet = header + sub.tobytes()
                super().send(packet)
                logging.debug(f"Fast UDP fragment sent: seq={self._seq-1}, channels={n_channels}, samples={spp}, size={len(packet)} bytes")
        else:
            # TCP: send header then payload
            header = self._pack_header(frame)
            super().sendall(header)
            super().sendall(frame.tobytes())
            logging.debug(f"Fast TCP frame sent: seq={self._seq-1}, shape={frame.shape}, dtype={frame.dtype}")

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
        """Receive exactly 'size' bytes from socket with optimized buffering."""
        # Pre-allocate buffer for better performance
        data = bytearray(size)
        view = memoryview(data)
        pos = 0
        
        while pos < size:
            # Try to receive remaining bytes in larger chunks
            remaining = size - pos
            chunk_size = min(remaining, 65536)  # Match socket buffer size
            
            bytes_received = super().recv_into(view[pos:pos + chunk_size])
            if not bytes_received:
                break
            pos += bytes_received
            
        return bytes(data[:pos])

    def _pack_header(self, frame: np.ndarray) -> bytes:
        """Pack array metadata into binary header with sequence number and timestamp."""
        seq = self._seq
        timestamp = time.perf_counter()
        self._seq += 1
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
        return struct.pack('<4sQdIII', self.MAGIC, seq, timestamp, shape[0], shape[1], dtype_code)

    def _unpack_header(self, header_data: bytes) -> tuple[bytes, tuple[int, int], np.dtype]:
        """Unpack binary header to get array metadata, discarding sequence and timestamp."""
        magic, seq, timestamp, shape0, shape1, dtype_code = struct.unpack('<4sQdIII', header_data)
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
