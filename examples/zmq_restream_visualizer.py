import argparse
import json
import time
import zmq
import pygame
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Subscribe to mono restream and visualize with pygame")
    parser.add_argument("--host", default="127.0.0.1", help="Server host to connect")
    parser.add_argument("--port", type=int, default=6000, help="ZMQ PUB port (restream)")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=300, help="Window height")
    parser.add_argument("--fps", type=int, default=60, help="Max redraw rate")
    args = parser.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "restream")

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Mono Restream Visualizer")
    clock = pygame.time.Clock()

    running = True
    last_block = np.zeros(256, dtype=np.float32)

    while running:
        # Poll socket non-blocking to keep UI responsive
        try:
            while True:
                topic, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
                msg = json.loads(payload.decode("utf-8"))
                data = np.array(msg["data"], dtype=np.float32)
                last_block = data
        except zmq.Again:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw background
        screen.fill((15, 15, 18))

        # Draw waveform
        if last_block.size > 0:
            w, h = args.width, args.height
            mid_y = h // 2
            # Normalize to [-1,1] and scale
            y = np.clip(last_block, -1.0, 1.0)
            xs = np.linspace(0, w - 1, num=last_block.size, dtype=np.int32)
            ys = (mid_y - (y * (h * 0.45))).astype(np.int32)
            points = list(zip(xs.tolist(), ys.tolist()))
            if len(points) >= 2:
                pygame.draw.lines(screen, (100, 220, 255), False, points, 2)

            # Overlay RMS bar
            rms = float(np.sqrt(np.mean(np.square(y))))
            bar_w = int(w * rms)
            pygame.draw.rect(screen, (80, 200, 120), pygame.Rect(0, 0, bar_w, 6))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sub.close(0)
    ctx.term()

if __name__ == "__main__":
    main() 