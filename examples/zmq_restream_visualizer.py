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


def rfft_power(block: np.ndarray) -> np.ndarray:
    """Return rFFT power spectrum (linear power) with Hann window."""
    x = block.astype(np.float32)
    n = len(x)
    if n < 8:
        return np.zeros(1, dtype=np.float32)
    window = np.hanning(n).astype(np.float32)
    spec = np.fft.rfft(x * window)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return power


def bins_for_band(fmin: float, fmax: float | None, nfft: int, sr: int) -> tuple[int, int]:
    """Map a frequency band to rFFT bin range [i0, i1). Excludes DC if fmin<=0."""
    nyq = sr / 2.0
    fmax_eff = nyq if (fmax is None or fmax > nyq) else fmax
    i0 = int(np.floor((fmin / sr) * nfft))
    i1 = int(np.ceil((fmax_eff / sr) * nfft))
    i0 = max(i0, 1)  # drop DC
    i1 = max(i1, i0 + 1)
    return i0, i1


def band_db_from_power(power: np.ndarray, i0: int, i1: int) -> float:
    """Compute 10*log10(mean(power[i0:i1])) with epsilon guard."""
    seg = power[i0:i1]
    if seg.size == 0:
        return -120.0
    p = float(np.mean(seg))
    return 10.0 * np.log10(max(p, 1e-12))


def compute_linear_bars_from_power(power: np.ndarray, n_bars: int) -> np.ndarray:
    """Compute simple linear, non-overlapping spectrum bars from rFFT power.
    - Drop DC bin
    - Split bins evenly, mean power per group, convert to dB
    """
    if power.size <= 1 or n_bars <= 0:
        return np.zeros(max(1, n_bars), dtype=np.float32)
    eps = 1e-12
    # Convert to dB
    mag_db = 10.0 * np.log10(np.maximum(power, eps))
    # Drop DC
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
    parser = argparse.ArgumentParser(description="Subscribe to mono restream and visualize with pygame (Events + Simple FFT)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host to connect")
    parser.add_argument("--port", type=int, default=6000, help="ZMQ PUB port (restream)")
    parser.add_argument("--width", type=int, default=900, help="Window width")
    parser.add_argument("--height", type=int, default=420, help="Window height")
    parser.add_argument("--fps", type=int, default=60, help="Max redraw rate")
    parser.add_argument("--bars", type=int, default=0, help="Number of spectrum bars (linear); 0 to hide")
    parser.add_argument("--smooth", type=float, default=0.25, help="Bar smoothing [0..1], 0 = off")
    # Event detection params
    parser.add_argument("--bass-max", type=float, default=200.0, help="Bass band upper Hz (start at >0 Hz)")
    parser.add_argument("--treble-min", type=float, default=4000.0, help="Treble band lower Hz")
    parser.add_argument("--treble-max", type=float, default=12000.0, help="Treble band upper Hz (<= Nyquist)")
    parser.add_argument("--fast-ema", type=float, default=0.4, help="Fast EMA (higher = slower)")
    parser.add_argument("--slow-ema", type=float, default=0.95, help="Slow EMA (higher = slower)")
    parser.add_argument("--bass-thresh", type=float, default=6.0, help="Bass event threshold (dB)")
    parser.add_argument("--treble-thresh", type=float, default=8.0, help="Treble event threshold (dB)")
    args = parser.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "restream")

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Mono Restream Events + FFT (Simple)")
    clock = pygame.time.Clock()

    running = True
    last_block = np.zeros(BLOCKSIZE, dtype=np.float32)

    # Bars smoothing state
    prev_bars = np.zeros(max(1, args.bars), dtype=np.float32)

    # Event detection state (fast/slow EMAs for bands)
    fast_bass = -120.0
    slow_bass = -120.0
    fast_treb = -120.0
    slow_treb = -120.0

    # Layout colors
    bg_color = (15, 15, 18)
    bar_color = (90, 200, 255)
    bass_color = (80, 220, 120)
    treb_color = (240, 160, 80)
    alert_color = (240, 80, 80)
    text_color = (230, 230, 230)

    # Optional font
    font = None
    try:
        font = pygame.font.SysFont(None, 18)
    except Exception:
        font = None

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

        # Compute rFFT power once
        power = rfft_power(last_block)
        nfft = max(len(last_block), 1)

        # Compute band powers (in dB)
        i0_b, i1_b = bins_for_band(1.0, args.bass_max, nfft, SAMPLING_RATE)
        i0_t, i1_t = bins_for_band(args.treble_min, args.treble_max, nfft, SAMPLING_RATE)
        bass_db = band_db_from_power(power, i0_b, i1_b)
        treb_db = band_db_from_power(power, i0_t, i1_t)

        # Update EMAs (fast and slow)
        a_fast = np.clip(args.fast_ema, 0.0, 0.9999)
        a_slow = np.clip(args.slow_ema, 0.0, 0.9999)
        fast_bass = a_fast * fast_bass + (1.0 - a_fast) * bass_db
        slow_bass = a_slow * slow_bass + (1.0 - a_slow) * bass_db
        fast_treb = a_fast * fast_treb + (1.0 - a_fast) * treb_db
        slow_treb = a_slow * slow_treb + (1.0 - a_slow) * treb_db

        # Event deltas and normalized meters
        bass_delta = max(0.0, fast_bass - slow_bass)
        treb_delta = max(0.0, fast_treb - slow_treb)
        bass_norm = np.clip(bass_delta / max(args.bass_thresh, 1e-6), 0.0, 1.0)
        treb_norm = np.clip(treb_delta / max(args.treble_thresh, 1e-6), 0.0, 1.0)

        # Bars (optional)
        show_bars = args.bars > 0
        if show_bars:
            bars_db = compute_linear_bars_from_power(power, args.bars)
            bars_norm = normalize_bars_per_frame(bars_db)
            if prev_bars.shape[0] != bars_norm.shape[0]:
                prev_bars = np.zeros_like(bars_norm)
            bars_smoothed = args.smooth * prev_bars + (1.0 - args.smooth) * bars_norm
            prev_bars = bars_smoothed
        else:
            bars_smoothed = None

        # Draw
        screen.fill(bg_color)

        w, h = args.width, args.height
        margin_x = 20
        margin_y = 16
        plot_w = w - 2 * margin_x
        x0 = margin_x
        y_cursor = margin_y

        # Draw meters first (clean and simple)
        meter_h = 22
        gap = 10
        # Bass meter
        bass_col = alert_color if bass_delta >= args.bass_thresh else bass_color
        pygame.draw.rect(screen, (40, 60, 45), pygame.Rect(x0, y_cursor, plot_w, meter_h))
        pygame.draw.rect(screen, bass_col, pygame.Rect(x0, y_cursor, int(plot_w * bass_norm), meter_h))
        if font:
            txt = font.render(f"BASS  {bass_delta:.1f} dB", True, text_color)
            screen.blit(txt, (x0 + 6, y_cursor + 3))
        y_cursor += meter_h + gap

        # Treble meter
        treb_col = alert_color if treb_delta >= args.treble_thresh else treb_color
        pygame.draw.rect(screen, (60, 55, 45), pygame.Rect(x0, y_cursor, plot_w, meter_h))
        pygame.draw.rect(screen, treb_col, pygame.Rect(x0, y_cursor, int(plot_w * treb_norm), meter_h))
        if font:
            txt = font.render(f"TREBLE  {treb_delta:.1f} dB", True, text_color)
            screen.blit(txt, (x0 + 6, y_cursor + 3))
        y_cursor += meter_h + gap

        # Optional bars below meters
        if show_bars and bars_smoothed is not None:
            bars_area_h = h - y_cursor - margin_y
            n = len(bars_smoothed)
            if n > 0 and bars_area_h > 0:
                bar_w = max(1, int(plot_w / (n * 1.15)))
                bar_gap = max(1, int(bar_w * 0.15))
                y_base = y_cursor + bars_area_h
                for i, v in enumerate(bars_smoothed):
                    bh = int(v * bars_area_h)
                    x = x0 + i * (bar_w + bar_gap)
                    y = y_base - bh
                    pygame.draw.rect(screen, bar_color, pygame.Rect(x, y, bar_w, bh))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sub.close(0)
    ctx.term()

if __name__ == "__main__":
    main() 