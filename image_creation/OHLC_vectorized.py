from __future__ import annotations

# Fast, vectorized re-write of the original OHLC image generator.
# Goals:
#  - Remove per-pixel Python loops (use NumPy/CuPy vector ops)
#  - Keep the public API identical so notebooks don’t need changes
#  - Optional GPU acceleration via CuPy if available (set use_gpu=True)
#  - Save as single-channel (grayscale) PNG for compact size
#
# Public methods kept:
#   - step1_choose_window_to_plot(start, end, horizon_start, horizon_end, past_start)
#   - calculate_moving_avg()
#   - step2_scale_window()
#   - step3_create_image_object()
#   - step4_draw_price_on_image()
#   - step5_draw_volume()
#   - draw_moving_average()
#   - step6_annotate_image()
#   - step7_save_img(directory_not_complete)
#   - step8_clear_window()
#   - plot_window(start, end, horizon_start, horizon_end, past_start, directory_not_complete)
#
# Notes:
#  - "window_size" defines image width in pixels (one column per bar)
#  - chart_px + vol_px should be <= height_px; the chart is drawn at the top, volume at the bottom
#  - Moving average is trailing over Close with the given window_size; it’s drawn over the chart area

import os
from typing import Optional, Tuple

from PIL import Image
import numpy as _np

# Optional GPU via CuPy (transparent via xp alias)
try:
    import cupy as _cp  # type: ignore
    _has_cupy = True
except Exception:
    _cp = None  # type: ignore
    _has_cupy = False


def _get_xp(use_gpu: bool):
    if use_gpu and _has_cupy:
        return _cp
    return _np


class OHLC:
    def __init__(self, price_info, window_size: int, height_px: int, chart_px: int, vol_px: int, *, use_gpu: bool = False):
        """
        price_info: pandas.DataFrame with columns [Open, High, Low, Close, Vol] and Date index
        window_size: width in data points (and pixels)
        height_px: total image height in pixels
        chart_px: height of the candles/MA chart area (top)
        vol_px: height of the volume bars area (bottom)
        use_gpu: if True and CuPy is available, compute scaling on GPU
        """
        self.price_info = price_info
        self.window_size = int(window_size)
        self.height_px = int(height_px)
        self.chart_px = int(chart_px)
        self.vol_px = int(vol_px)

        assert self.chart_px + self.vol_px <= self.height_px, (
            "chart_px + vol_px must be <= height_px"
        )

        # dynamic state (set per window)
        self.current_window = None
        self.volume = None
        self.m_avg_data = None
        self.moving_average = None
        self.kill = False  # used as in original code for early aborts

        # scaled integer pixel rows for OHLC and MA
        self.current_window_scaled = None  # shape [W, 4] of ints (y positions)
        self.moving_avg_scaled = None     # shape [M] of ints (y positions)

        # canvas
        self.canvas = None  # NumPy uint8 (CPU) always; Pillow requires CPU array to save

        # book-keeping of indices
        self.__start_end = None
        self.__horizon = None
        self.__past_start = None

        # compute backend
        self.use_gpu = bool(use_gpu and _has_cupy)
        self.xp = _get_xp(self.use_gpu)

    # -----------------------------
    # 1) choose window
    # -----------------------------
    def step1_choose_window_to_plot(self, start: int, end: int, horizon_start: int, horizon_end: int, past_start: int):
        self.__start_end = (int(start), int(end))
        self.__horizon = (int(horizon_start), int(horizon_end))
        self.__past_start = int(past_start)

        # Select the core window for drawing candles
        self.current_window = self.price_info.iloc[start:end][["Open", "High", "Low", "Close"]].copy()
        # Extract volume separately
        self.volume = self.price_info.iloc[start:end]["Vol"].to_numpy()
        # For moving average, use the Close series up to `end` so the MA aligns with trailing window
        self.m_avg_data = self.price_info.iloc[max(0, end - max(self.window_size * 2, self.window_size*1)) : end][
            "Close"
        ].copy()

        # NEW: materialize horizon rows for later annotation
        # (treat horizon_end as exclusive, so end row is horizon_end-1)
        try:
            self.horizon_start = self.price_info.iloc[horizon_start]
        except Exception:
            self.horizon_start = None
        try:
            self.horizon_end = self.price_info.iloc[horizon_end - 1]
        except Exception:
            self.horizon_end = None

        if len(self.current_window) != self.window_size:
            # keep parity with original behavior: if mis-sized, mark kill and return
            self.kill = True

    # -----------------------------
    # 2) moving average
    # -----------------------------
    def calculate_moving_avg(self):
        if self.kill:
            return
        if self.m_avg_data is None or len(self.m_avg_data) < self.window_size:
            # Not enough data to compute trailing MA of window_size
            self.moving_average = _np.array([], dtype=_np.float32)
            return
        # trailing moving average over Close with window_size
        # prefer NumPy/CuPy conv for speed
        xp = self.xp
        close = xp.asarray(self.m_avg_data.to_numpy(), dtype=xp.float32)
        kernel = xp.ones(int(self.window_size), dtype=xp.float32) / float(self.window_size)
        ma = xp.convolve(close, kernel, mode="valid")  # shape: len(close)-W+1
        # Keep only the MA values aligned to the current visible window
        # If we computed MA over more history, the *last* window_size entries correspond best
        needed = min(len(ma), self.window_size)
        self.moving_average = ma[-needed:].get() if self.use_gpu else ma[-needed:]
        # ensure CPU ndarray for later scaling consistency
        self.moving_average = _np.asarray(self.moving_average, dtype=_np.float32)

    # -----------------------------
    # 3) scale to pixel coordinates
    # -----------------------------
    def step2_scale_window(self):
        if self.kill:
            return
        W = len(self.current_window)
        if W == 0:
            self.kill = True
            return
        # Data to scale
        price_np = self.current_window.to_numpy(dtype=_np.float32)  # shape [W, 4]
        ma_np = self.moving_average if self.moving_average is not None else _np.array([], dtype=_np.float32)

        # Min/max over price + MA to keep a common scale
        if ma_np.size > 0:
            glb_max = float(max(price_np.max(), ma_np.max()))
            glb_min = float(min(price_np.min(), ma_np.min()))
        else:
            glb_max = float(price_np.max())
            glb_min = float(price_np.min())

        if glb_max == glb_min:
            self.kill = True
            return

        # Scale to [0, chart_px-1] and invert y (0 at top)
        scale = (price_np - glb_min) / (glb_max - glb_min)
        y_price = _np.rint((self.chart_px - 1) * (1.0 - scale)).astype(_np.int32)
        self.current_window_scaled = y_price  # shape [W,4] ints

        if ma_np.size > 0:
            ma_scale = (ma_np - glb_min) / (glb_max - glb_min)
            y_ma = _np.rint((self.chart_px - 1) * (1.0 - ma_scale)).astype(_np.int32)
            # If MA shorter than window, right-align it
            if len(y_ma) < W:
                pad = W - len(y_ma)
                y_ma = _np.pad(y_ma, (pad, 0), constant_values=-1)  # -1 means "not drawn"
            self.moving_avg_scaled = y_ma
        else:
            self.moving_avg_scaled = _np.full(W, -1, dtype=_np.int32)

        # Volume scaling to [0, vol_px]
        self.__scale_volume()

    def __scale_volume(self):
        vol = _np.asarray(self.volume, dtype=_np.float32)
        vmax = float(vol.max()) if vol.size else 0.0
        if vmax <= 0:
            self.vol_scaled = _np.zeros_like(vol, dtype=_np.int32)
            return
        h = max(0, self.vol_px - 1)
        self.vol_scaled = _np.rint(h * (vol / vmax)).astype(_np.int32)

    # -----------------------------
    # 4) create canvas
    # -----------------------------
    def step3_create_image_object(self):
        if self.kill:
            return
        # Canvas is always on CPU for Pillow. Use grayscale L (uint8)
        W = len(self.current_window)
        H = self.height_px
        self.canvas = _np.zeros((H, W), dtype=_np.uint8)

    # -----------------------------
    # 5) draw candles (vectorized)
    # -----------------------------
    def step4_draw_price_on_image(self):
        if self.kill or self.canvas is None:
            return
        y = self.current_window_scaled  # [W,4] => (Open, High, Low, Close) in pixel rows
        W = y.shape[0]
        X = _np.arange(W, dtype=_np.int32)

        o = y[:, 0]
        h = y[:, 1]
        l = y[:, 2]
        c = y[:, 3]

        # High–low vertical bodies (bright)
        lo = _np.minimum(h, l)
        hi = _np.maximum(h, l)
        lengths = hi - lo + 1
        # Build concat row/col indices
        rows = _np.concatenate([_np.arange(lo[i], hi[i] + 1, dtype=_np.int32) for i in range(W)])
        cols = _np.repeat(X, lengths)
        self.canvas[rows, cols] = 255

        # Open/Close ticks (single pixels)
        self.canvas[o, X] = 255
        self.canvas[c, X] = 255

    # -----------------------------
    # 6) draw volume (vectorized)
    # -----------------------------
    def step5_draw_volume(self):
        if self.kill or self.canvas is None:
            return
        W = len(self.vol_scaled)
        if W == 0:
            return

        # Ensure integer heights within [0, vol_px]
        h = _np.asarray(self.vol_scaled, dtype=_np.int32)
        if self.vol_px > 0:
            h = _np.clip(h, 0, self.vol_px)

        X = _np.arange(W, dtype=_np.int32)

        # Fill exactly 'h[i]' pixels from the bottom up: rows [H - h[i], H)
        rows = _np.concatenate([
            _np.arange(self.height_px - h[i], self.height_px, dtype=_np.int32)
            for i in range(W) if h[i] > 0
        ])
        cols = _np.concatenate([
            _np.full(h[i], X[i], dtype=_np.int32)
            for i in range(W) if h[i] > 0
        ])

        if rows.size:
            self.canvas[rows, cols] = _np.maximum(self.canvas[rows, cols], 180)
    # -----------------------------
    # 7) draw moving average (vectorized polyline)
    # -----------------------------
    def draw_moving_average(self):
        if self.kill or self.canvas is None:
            return
        y = self.moving_avg_scaled
        if y is None or y.size == 0:
            return
        W = len(y)
        # vertical spans between consecutive points
        x0 = _np.arange(W - 1, dtype=_np.int32)
        y0 = y[:-1]
        y1 = y[1:]
        mask = (y0 >= 0) & (y1 >= 0)
        x0 = x0[mask]
        y0 = y0[mask]
        y1 = y1[mask]
        if x0.size == 0:
            return
        lo = _np.minimum(y0, y1)
        hi = _np.maximum(y0, y1)
        lengths = hi - lo + 1
        rows = _np.concatenate([_np.arange(lo[i], hi[i] + 1, dtype=_np.int32) for i in range(len(x0))])
        cols = _np.repeat(x0, lengths)
        # Use mid-gray so it’s visible over volume fills too
        self.canvas[rows, cols] = _np.maximum(self.canvas[rows, cols], 150)

    # -----------------------------
    # 8) annotate (no-op placeholder to keep API)
    # -----------------------------
    def step6_annotate_image(self):
        start_price = self.horizon_start["Open"]
        end_price = self.horizon_end["Close"]
        self.label = True if end_price > start_price else False

    # -----------------------------
    # 9) save
    # -----------------------------
    def step7_save_img(self, directory_not_complete: str):
        if self.kill or self.canvas is None:
            return
        out_path = f"{directory_not_complete}_{self.label}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img = Image.fromarray(self.canvas, mode="L")
        img.save(out_path)

    # -----------------------------
    # 10) clear state for next window
    # -----------------------------
    def step8_clear_window(self):
        # Reset only the per-window state
        self.current_window = None
        self.volume = None
        self.m_avg_data = None
        self.moving_average = None
        self.current_window_scaled = None
        self.moving_avg_scaled = None
        self.canvas = None
        self.kill = False

    # -----------------------------
    # Convenience orchestrator (unchanged signature)
    # -----------------------------
    def plot_window(self, start, end, horizon_start, horizon_end, past_start, directory_not_complete):
        self.step1_choose_window_to_plot(start, end, horizon_start, horizon_end, past_start)
        if self.kill:
            # bad window sizing; skip cleanly
            self.kill = False
            return
        self.calculate_moving_avg()
        self.step2_scale_window()
        if self.kill:
            # degenerate window or scaling issue
            self.kill = False
            return
        self.step3_create_image_object()
        self.step4_draw_price_on_image()
        self.step5_draw_volume()
        if self.kill:
            # volume or other issue
            self.kill = False
            return
        self.draw_moving_average()
        self.step6_annotate_image()
        self.step7_save_img(directory_not_complete)
        self.step8_clear_window()
