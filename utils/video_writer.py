"""Browser-playable video output without relying on OpenCV's optional encoders."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np


class BrowserVideoWriter:
    """Stream OpenCV BGR frames to bundled FFmpeg's H.264 encoder."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int]) -> None:
        from imageio_ffmpeg import get_ffmpeg_exe

        self.path = Path(path)
        self.width, self.height = map(int, size)
        self._closed = False
        self._frames_written = 0
        self._error_log = tempfile.TemporaryFile(mode="w+b", dir=self.path.parent)
        command = [
            get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{fps:.12g}", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.path),
        ]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._error_log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._error_log.close()
            raise

    def _error_message(self) -> str:
        self._error_log.seek(0)
        return self._error_log.read().decode("utf-8", errors="replace").strip()

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Video writer is closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("Video frame must match the configured BGR uint8 dimensions")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._frames_written += 1
        except OSError as exc:
            self._process.wait(timeout=30)
            message = self._error_message()
            try:
                self._process.stdin.close()
            except OSError:
                pass
            self._closed = True
            self._error_log.close()
            raise RuntimeError(f"H.264 encoding failed: {message}") from exc

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            try:
                code = self._process.wait(timeout=60)
            except subprocess.TimeoutExpired as exc:
                self._process.kill()
                self._process.wait()
                raise RuntimeError("H.264 encoder did not finish") from exc
            if code != 0 or self._frames_written == 0:
                raise RuntimeError(f"H.264 encoding failed: {self._error_message()}")
        finally:
            self._error_log.close()
