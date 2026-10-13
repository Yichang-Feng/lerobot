#!/usr/bin/env python3
"""
Valen (Jev) Remote Decision Server
==================================
Runs on the evaluator host (remote machine or local dedicated process).
Listens for ZMQ multipart requests containing:
  Frame 0: JSON metadata / request spec (e.g. {"phase": 1})
  Frame 1: Binary JPEG bytes of horizontally stitched [Global | Right Wrist] image.

Returns:
  JSON with structured choice, confidence, candidate probabilities, and inference latency.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq

# Ensure valen package is importable
VALEN_ROOT = Path(__file__).resolve().parent.parent
if str(VALEN_ROOT) not in sys.path:
    sys.path.insert(0, str(VALEN_ROOT))

from valen.evaluation.inference import predict
from valen.modeling.factory import (
    backend_for_model,
    build_compiler,
    build_model,
    normalize_model_config,
)
from valen.training.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ValenService] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ValenService")


def get_default_questions_for_phase(phase: int) -> tuple[str, dict]:
    """Return prompt text and questions spec for Phase 0 (Table->Box), Phase 1 (Box->Table), or Phase 2 (Table B Target)."""
    if phase == 0:
        # Phase 0 / Subtask 1: Table to Box
        context_text = (
            "The input image is horizontally stitched from two camera views: "
            "the left half is the third-person global overview showing the tabletop, the blue box, and the robot arm; "
            "the right half is the right-wrist camera close-up showing the right gripper and its interaction with the water bottle. "
            "The current task is: pick up the water bottle from the table and place it into the blue box."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Determine the location and state of the water bottle in the workspace:"
                ),
                "criteria": {
                    "phase1_in_progress": "The water bottle is on the table or being transported in the air.",
                    "bottle_in_box_completed": "The water bottle is resting inside the blue box.",
                    "anomaly_dropped": "The water bottle has dropped or fallen outside the box.",
                },
            }
        }
    elif phase == 1:
        # Phase 1 / Subtask 2: Box to Table
        context_text = (
            "The input image is horizontally stitched from two camera views: "
            "the left half is the third-person global overview showing the tabletop, the blue box, and the robot arm; "
            "the right half is the right-wrist camera close-up showing the right gripper and its interaction with the water bottle. "
            "The current task is: take the water bottle out of the blue box and place it back on the table."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Determine the location and state of the water bottle in the workspace:"
                ),
                "criteria": {
                    "phase2_in_progress": "The water bottle is inside the blue box or being held in the air.",
                    "bottle_on_table_completed": "The water bottle is resting on the table, and the gripper has released it.",
                    "anomaly_dropped": "The water bottle has fallen off the table or dropped onto the floor.",
                },
            }
        }
    elif phase == 2:
        # Phase 2: Cross-table target decision at table B (Container detection: box vs table)
        context_text = (
            "The input image is horizontally stitched from two camera views: "
            "the left half is the third-person global overview showing Table B; "
            "the right half is the right-wrist camera close-up showing the robot gripper holding the water bottle. "
            "The robot is standing in front of Table B to place the water bottle into a container."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Look at Table B in the camera views. What is present on the tabletop?"
                ),
                "criteria": {
                    "place_into_box": "A box, storage container, cardboard box, or bin is present on Table B.",
                    "place_on_table": "Table B is flat and empty, with no box or container present.",
                },
            }
        }
    elif phase == 3:
        # Phase 3: Grasp & Lift Verification (桌上水瓶夹取与离地抬起验证)
        context_text = (
            "Observe the robot workspace and the water bottle. Left: global view. Right: right wrist view."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Where is the water bottle located?"
                ),
                "criteria": {
                    "held_in_air": "The water bottle is grasped by the right gripper and lifted in the air above the table.",
                    "on_table_or_unlifted": "The water bottle is resting on the table surface, not lifted in the air.",
                },
            }
        }
    elif phase == 4:
        # Phase 4: Table 1 - Water Bottle Presence Detection (第一张桌子: 水瓶存在性检测)
        context_text = (
            "The input image is horizontally stitched from two camera views: "
            "the left half is the third-person global overview showing Table 1; "
            "the right half is the right-wrist camera close-up showing the gripper in front of Table 1. "
            "The robot is standing in front of Table 1."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Look at the tabletop in front of the robot. What object is present on the table?"
                ),
                "criteria": {
                    "water_bottle_present": "There is a water bottle standing or placed on the table surface.",
                    "no_water_bottle": "There is no water bottle on the table; only the bare table surface is present.",
                },
            }
        }
    elif phase == 5:
        # Phase 5: Table 2 - Box Presence Detection (第二张桌子: 目标盒子存在性检测)
        context_text = (
            "The input image is horizontally stitched from two camera views: "
            "the left half is the third-person global overview showing Table 2; "
            "the right half is the right-wrist camera close-up showing the robot gripper holding the water bottle. "
            "The robot is standing in front of Table 2 to place the water bottle into a container."
        )
        questions = {
            "task_phase": {
                "type": "choice",
                "instructions": (
                    "Look at Table 2 in the camera views. What is present on the tabletop?"
                ),
                "criteria": {
                    "place_into_box": "A box, storage container, cardboard box, or bin is present on Table 2.",
                    "place_on_table": "Table 2 is flat and empty, with no box or container present.",
                },
            }
        }
    else:
        # Fallback to Phase 0
        return get_default_questions_for_phase(0)
    return context_text, questions





class ValenServer:
    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str,
        port: int = 5559,
        device: str = "cuda:0",
        shm_dir: str = "/dev/shm",
    ):
        self.port = port
        self.device = device if torch.cuda.is_available() else "cpu"
        self.shm_dir = Path(shm_dir)
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_img_path = self.shm_dir / f"valen_live_frame_{os.getpid()}.jpg"

        logger.info("Initializing Valen Jev Decision Server on device: %s", self.device)
        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Base Model: %s", base_model_path)

        ckpt_dir = Path(checkpoint_path)
        cfg_file = ckpt_dir / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found in checkpoint: {cfg_file}")

        config = json.loads(cfg_file.read_text(encoding="utf-8"))
        config = normalize_model_config(config)
        config["model_path"] = str(Path(base_model_path).resolve())
        config["device"] = self.device

        t0 = time.time()
        logger.info("Building model architecture...")
        self.model = build_model(config)

        logger.info("Loading checkpoint weights...")
        load_checkpoint(str(ckpt_dir), self.model)
        self.model.eval()

        logger.info("Building compiler...")
        self.compiler = build_compiler(config, media_root=str(self.shm_dir))
        logger.info("Model loaded successfully in %.2fs!", time.time() - t0)

        # Warm up model with a dummy image
        self._warmup()

        # Setup ZMQ
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        try:
            self.socket.bind(f"tcp://0.0.0.0:{self.port}")
            logger.info("ZMQ REP socket bound to tcp://0.0.0.0:%d", self.port)
        except Exception as e:
            logger.warning("Failed to bind TCP port %d (%s)", self.port, e)

        ipc_path = self.shm_dir / "valen.ipc"
        try:
            if ipc_path.exists():
                ipc_path.unlink()
            self.socket.bind(f"ipc://{ipc_path}")
            logger.info("ZMQ REP socket bound to ipc://%s", ipc_path)
        except Exception as e:
            logger.warning("Failed to bind IPC socket (%s)", e)

    def _warmup(self):
        logger.info("Warming up Valen decision head...")
        dummy_img = np.zeros((448, 896, 3), dtype=np.uint8)
        cv2.imwrite(str(self.tmp_img_path), dummy_img)
        ctx_text, questions = get_default_questions_for_phase(0)
        record = {
            "group_id": "warmup",
            "request": {
                "state": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ctx_text},
                                {"type": "image_url", "image_url": {"url": str(self.tmp_img_path)}},
                            ],
                        }
                    ]
                },
                "questions": questions,
            },
        }
        compiled = self.compiler.compile(record)
        with torch.no_grad():
            predict(self.model, compiled)
        logger.info("Warmup complete!")

    def run(self):
        logger.info("🚀 Valen Decision Server is ready and waiting for requests on port %d...", self.port)
        request_count = 0

        while True:
            try:
                # Receive multipart message
                frames = self.socket.recv_multipart()
                if not frames:
                    continue

                # Frame 0: JSON meta
                meta_json = frames[0].decode("utf-8")
                meta = json.loads(meta_json)

                # Health check / Ping
                if meta.get("ping", False):
                    self.socket.send_json({"status": "ok", "pong": True, "time": time.time()})
                    continue

                if len(frames) < 2:
                    self.socket.send_json({"status": "error", "error": "Missing image frame in multipart payload"})
                    continue

                # Frame 1: JPEG image bytes
                img_bytes = frames[1]
                t_start = time.perf_counter()

                # Save or write image to /dev/shm
                with open(self.tmp_img_path, "wb") as f:
                    f.write(img_bytes)

                phase = int(meta.get("phase", 0))
                custom_questions = meta.get("questions")
                custom_text = meta.get("context_text")

                if custom_questions:
                    questions = custom_questions
                    ctx_text = custom_text or "Observe the robot manipulation workspace."
                else:
                    ctx_text, questions = get_default_questions_for_phase(phase)

                record = {
                    "group_id": f"req_{request_count}",
                    "request": {
                        "state": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": ctx_text},
                                        {"type": "image_url", "image_url": {"url": str(self.tmp_img_path)}},
                                    ],
                                }
                            ]
                        },
                        "questions": questions,
                    },
                }

                # Compile & Predict
                compiled = self.compiler.compile(record)
                with torch.no_grad():
                    out = predict(self.model, compiled)

                latency_ms = (time.perf_counter() - t_start) * 1000.0

                ans = out["answers"]["task_phase"]
                choice = ans.get("choice", "")
                confidence = float(ans.get("confidence", 0.0))
                probs = ans.get("probabilities", {})

                response = {
                    "status": "success",
                    "phase": phase,
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": probs,
                    "cost_ms": round(latency_ms, 2),
                }

                self.socket.send_json(response)
                request_count += 1

                if phase in (2, 3) or request_count % 5 == 0 or confidence >= 0.70:
                    prob_val = probs.get(choice, 0.0)
                    logger.info(
                        "[%04d] Phase %d -> Choice: %-26s | Conf: %.3f (P=%5.1f%%) | Cost: %5.1fms",
                        request_count,
                        phase,
                        choice,
                        confidence,
                        prob_val * 100.0,
                        latency_ms,
                    )

            except Exception as e:
                logger.error("Error processing request: %s", e, exc_info=True)
                try:
                    self.socket.send_json({"status": "error", "error": str(e)})
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Valen Jev Remote Decision Server")
    parser.add_argument("--port", type=int, default=5559, help="ZMQ port to bind (default: 5559)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(VALEN_ROOT / "models" / "Valen-Preview-0923"),
        help="Path to Valen checkpoint directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(VALEN_ROOT / "models" / "Qwen3.5-2B"),
        help="Path to Qwen3.5-2B base model directory",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (default: cuda:0)")
    args = parser.parse_args()

    server = ValenServer(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        port=args.port,
        device=args.device,
    )
    server.run()


if __name__ == "__main__":
    main()
