import argparse
import json
import time
import zmq
import pygame
import numpy as np

try:
    # Use project config if available
    from spatial_audio_ai.config import SAMPLING_RATE, BLOCKSIZE
except Exception:
    SAMPLING_RATE = 48000
    BLOCKSIZE = 256


def compute_linear_bars(block: np.ndarray, n_bars: int) -> np.ndarray:
    """Compute simple linear, non-overlapping spectrum bars from a mono block.
    - Hann window
    - rFFT magnitude -> power -> dB
    - Split bins evenly (excluding DC), take mean per group
    """
    x = block.astype(np.float32)
    n = len(x)
    if n < 8 or n_bars <= 0:
        return np.zeros(max(1, n_bars), dtype=np.float32)
    window = np.hanning(n).astype(np.float32)
    spec = np.fft.rfft(x * window)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    eps = 1e-12
    mag_db = 10.0 * np.log10(np.maximum(power, eps))
    # Drop DC bin to reduce correlation/offset
    bins = np.arange(1, len(mag_db), dtype=np.int32)
    groups = np.array_split(bins, n_bars)
    bars = np.empty(n_bars, dtype=np.float32)
    for i, g in enumerate(groups):
        if g.size == 0:
            bars[i] = -120.0
        else:
            bars[i] = float(np.mean(mag_db[g]))
    return bars


def normalize_bars_per_frame(bars: np.ndarray) -> np.ndarray:
    """Per-frame percentile normalization to 0..1 for clean scaling.
    Maps 10th percentile -> 0 and 95th percentile -> 1.
    """
    if bars.size == 0:
        return bars
    low = float(np.percentile(bars, 10))
    high = float(np.percentile(bars, 95))
    denom = max(high - low, 1e-6)
    norm = (bars - low) / denom
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Subscribe to mono restream and visualize with pygame (Simple FFT)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host to connect")
    parser.add_argument("--port", type=int, default=6000, help="ZMQ PUB port (restream)")
    parser.add_argument("--width", type=int, default=900, help="Window width")
    parser.add_argument("--height", type=int, default=400, help="Window height")
    parser.add_argument("--fps", type=int, default=60, help="Max redraw rate")
    parser.add_argument("--bars", type=int, default=32, help="Number of spectrum bars (linear)")
    parser.add_argument("--smooth", type=float, default=0.3, help="Temporal smoothing [0..1], 0 = off")
    args = parser.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "restream")

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Mono Restream FFT Visualizer (Simple)")
    clock = pygame.time.Clock()

    running = True
    last_block = np.zeros(BLOCKSIZE, dtype=np.float32)

    # Temporal smoothing state
    prev_bars = np.zeros(args.bars, dtype=np.float32)

    bg_color = (15, 15, 18)
    bar_color = (100, 220, 255)

    while running:
        # Poll socket non-blocking to keep UI responsive
        try:
            while True:
                topic, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
                msg = json.loads(payload.decode("utf-8"))
                data = np.array(msg["data"], dtype=np.float32)
                if data.size != last_block.size:
                    last_block = np.zeros_like(data)
                last_block = data
        except zmq.Again:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Compute and normalize bars per frame
        bars_db = compute_linear_bars(last_block, args.bars)
        bars_norm = normalize_bars_per_frame(bars_db)
        # Optional light smoothing
        if prev_bars.shape[0] != bars_norm.shape[0]:
            prev_bars = np.zeros_like(bars_norm)
        bars_smoothed = args.smooth * prev_bars + (1.0 - args.smooth) * bars_norm
        prev_bars = bars_smoothed

        # Draw background
        screen.fill(bg_color)

        # Draw bars
        w, h = args.width, args.height
        margin_x = 20
        margin_y = 20
        plot_w = w - 2 * margin_x
        plot_h = h - 2 * margin_y
        x0, y0 = margin_x, h - margin_y

        n = len(bars_smoothed)
        if n > 0:
            bar_w = max(1, int(plot_w / (n * 1.2)))
            gap = max(1, int(bar_w * 0.2))
            for i, v in enumerate(bars_smoothed):
                bh = int(v * plot_h)
                x = x0 + i * (bar_w + gap)
                y = y0 - bh
                pygame.draw.rect(screen, bar_color, pygame.Rect(x, y, bar_w, bh))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sub.close(0)
    ctx.term()

if __name__ == "__main__":
    main() 