#!/usr/bin/env python3
"""
Valen Evaluator Client
======================
High-performance ZMQ client for communicating with remote Valen (Jev) decision server.
Handles image preprocessing, dual-view horizontal stitching [Global | Right Wrist],
JPEG compression, non-blocking requests with timeouts, and automatic reconnection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)


@dataclass
class ValenResult:
    status: str
    phase: int = 0
    choice: str = ""
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    cost_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status == "success" and bool(self.choice)


class ValenClient:
    """Non-blocking, fault-tolerant ZMQ client for remote Valen decision server."""

    def __init__(
        self,
        host: str = "10.8.8.98",
        port: int = 5559,
        endpoint: str | None = None,
        timeout_ms: int = 600,
        target_height: int = 448,
        jpeg_quality: int = 85,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality

        if endpoint is not None:
            self.endpoint = endpoint
        elif host in ("127.0.0.1", "localhost", "local") and Path("/dev/shm/valen.ipc").exists():
            self.endpoint = "ipc:///dev/shm/valen.ipc"
        else:
            self.endpoint = f"tcp://{self.host}:{self.port}"

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._init_socket()

    def _init_socket(self) -> None:
        """Create or recreate the ZMQ REQ socket with strict timeouts."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

        if self._context is None:
            self._context = zmq.Context()

        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        logger.info("ValenClient connected to %s (timeout=%dms)", self.endpoint, self.timeout_ms)

    def ping(self) -> bool:
        """Send a lightweight ping to verify connection with Valen server."""
        try:
            payload = json.dumps({"ping": True}).encode("utf-8")
            self._socket.send(payload)
            resp = self._socket.recv_json()
            return resp.get("status") == "ok" and resp.get("pong", False)
        except Exception as e:
            logger.debug("Valen ping failed (%s), resetting socket...", e)
            self._init_socket()
            return False

    def stitch_images(
        self,
        global_img: Any,
        wrist_img: Any,
    ) -> np.ndarray | None:
        """
        Horizontally concatenate Global view (Left) and Wrist view (Right).
        Normalizes inputs from Torch tensors (CHW or HWC) or Numpy arrays.
        """
        try:
            # Helper to convert input to numpy HWC uint8
            def to_numpy_hwc(img):
                if img is None:
                    return None
                if hasattr(img, "detach"):
                    img = img.detach().cpu().numpy()
                img = np.asarray(img)
                # Check for (C, H, W)
                if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                    img = np.transpose(img, (1, 2, 0))
                # Normalize float [0, 1] to uint8 [0, 255]
                if img.dtype in (np.float32, np.float64) and img.max() <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                # If grayscale, convert to BGR
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                return img

            g_np = to_numpy_hwc(global_img)
            w_np = to_numpy_hwc(wrist_img)

            if g_np is None and w_np is None:
                return None

            th = self.target_height
            if g_np is not None:
                gh, gw = g_np.shape[:2]
                target_gw = int(gw * (th / gh))
                g_resized = cv2.resize(g_np, (target_gw, th), interpolation=cv2.INTER_AREA)
            else:
                g_resized = np.zeros((th, th, 3), dtype=np.uint8)

            if w_np is not None:
                wh, ww = w_np.shape[:2]
                target_ww = int(ww * (th / wh))
                w_resized = cv2.resize(w_np, (target_ww, th), interpolation=cv2.INTER_AREA)
            else:
                w_resized = np.zeros((th, th, 3), dtype=np.uint8)

            # Horizontal stack: Left = Global, Right = Wrist
            stitched = np.hstack([g_resized, w_resized])
            return stitched
        except Exception as e:
            logger.error("Failed to stitch images: %s", e)
            return None

    def evaluate(
        self,
        global_img: Any,
        wrist_img: Any,
        phase: int,
        custom_questions: dict | None = None,
        custom_text: str | None = None,
    ) -> ValenResult:
        """
        Send stitched frame and phase query to Valen server.
        Non-blocking with respect to timeout; never throws unhandled exceptions.
        """
        stitched = self.stitch_images(global_img, wrist_img)
        if stitched is None:
            return ValenResult(status="error", error="Image stitching returned None")

        # Encode image to JPEG in memory
        ret, jpeg_buf = cv2.imencode(
            ".jpg",
            stitched,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ret:
            return ValenResult(status="error", error="Failed to encode stitched image to JPEG")

        jpeg_bytes = jpeg_buf.tobytes()

        # Build request metadata
        meta = {
            "phase": int(phase),
            "timestamp": time.time(),
        }
        if custom_questions:
            meta["questions"] = custom_questions
        if custom_text:
            meta["context_text"] = custom_text

        meta_bytes = json.dumps(meta).encode("utf-8")

        # Send multipart request [JSON metadata, JPEG bytes]
        t0 = time.perf_counter()
        try:
            self._socket.send_multipart([meta_bytes, jpeg_bytes])
            resp = self._socket.recv_json()
            r_cost = (time.perf_counter() - t0) * 1000.0

            if resp.get("status") == "success":
                return ValenResult(
                    status="success",
                    phase=resp.get("phase", phase),
                    choice=resp.get("choice", ""),
                    confidence=float(resp.get("confidence", 0.0)),
                    probabilities=resp.get("probabilities", {}),
                    cost_ms=r_cost,
                )
            else:
                return ValenResult(
                    status="error",
                    phase=phase,
                    error=resp.get("error", "Unknown server error"),
                    cost_ms=r_cost,
                )

        except zmq.Again:
            logger.warning("Valen request timed out (> %dms) to %s, resetting socket...", self.timeout_ms, self.endpoint)
            self._init_socket()
            return ValenResult(status="timeout", phase=phase, error=f"Timeout after {self.timeout_ms}ms")
        except Exception as e:
            logger.warning("Valen request failed: %s, resetting socket...", e)
            self._init_socket()
            return ValenResult(status="error", phase=phase, error=str(e))

    def close(self) -> None:
        """Close ZMQ socket and terminate context."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None
