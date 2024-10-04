#!/usr/bin/python3
import logging
import numpy as np
import time
from numpysocket import NumpySocket

from sound_system import SoundSystem
import threading

logger = logging.getLogger("sound server")
logger.setLevel(logging.WARNING)
sound_system = SoundSystem(logging.WARNING)

def client_handler(addr, conn):    
    with conn:
        logger.info(f"connected: {addr}")
        while True:
            sound_array = conn.recv()
            if len(sound_array.shape) == 2:
                logger.info(f"array received, array shape is {sound_array.shape}")
                sound_system.add_to_playback_queue(sound_array)

    logger.info(f"disconnected: {addr}")


with NumpySocket() as s:
    s.bind(("10.40.49.21", 9999))
    s.listen()
    while True:
        print("Connecting to new client...")
        conn, addr = s.accept()
        thread = threading.Thread(target=client_handler, args=(addr,conn,))
        thread.start()
