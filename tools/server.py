#!/usr/bin/python3
import logging
import numpy as np
import time
from numpysocket import NumpySocket

from sound_system import SoundSystem
import threading

logger = logging.getLogger("sound server")
logger.setLevel(logging.INFO)
sound_system = SoundSystem(logging.WARNING)

def client_handler(addr, conn):    
    with conn:
        logger.info(f"connected: {addr}")
        while True:
            sound_array = conn.recv()
            if len(sound_array.shape) == 2:
                # logger.info(f"Sound array stats - Min: {np.min(sound_array)}, Max: {np.max(sound_array)}, Mean: {np.mean(sound_array)}, Std: {np.std(sound_array)}")
                sound_system.add_to_playback_queue(sound_array)
            # buffer_time = sound_system.get_current_buffer_time()
            # random_number = np.random.rand(500)
            # answer_array = np.array([random_number], dtype=np.float32)
            # conn.send(answer_array)

            # buffer_time = np.ones(400) * buffer_time
            # answer_array = np.array([buffer_time], dtype=np.float32)
            # conn.send(answer_array)

            # logger.info(f"sent random number: {random_number}")

    logger.info(f"disconnected: {addr}")


with NumpySocket() as s:
    s.bind(("10.40.49.47", 9999))
    s.listen()
    try:
        while True:
            print("Connecting to new client...")
            conn, addr = s.accept()
            thread = threading.Thread(target=client_handler, args=(addr, conn), daemon=True)
            thread.start()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")


