# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerun visualization backend.

Live control-loop streaming to the Rerun viewer (:func:`log_rerun_data`). Callers usually select a
backend at runtime through the dispatch in :mod:`lerobot.utils.visualization_utils` rather than
importing from here directly. Requires the ``viz`` extra (``pip install 'lerobot[viz]'``).
"""

import numbers
import os
import sys

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def _to_numpy(x):
    """Convert input (e.g. torch.Tensor, list, tuple, np.ndarray) to numpy.ndarray or scalar."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x


def _is_scalar(x):
    arr = _to_numpy(x)
    return isinstance(arr, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(arr, np.ndarray) and arr.size == 1
    )


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    log_rerun_data.blueprint = None  # Reset blueprint cache for new session

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    # Ensure the active environment's bin directory is in PATH so rr.spawn() finds the rerun executable
    env_bin = os.path.join(sys.prefix, "bin")
    exec_bin = os.path.dirname(sys.executable)
    for b in (env_bin, exec_bin):
        if b and b not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")

    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """Build a Rerun blueprint laying out camera images, observation and action scalars in separate views.

    Camera images, observation and action scalars are arranged in a grid.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]) -> None:
    """Build and send the blueprint once, from the first observation and action data."""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    if not (observation_paths or action_paths or image_paths):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - PyTorch Tensors and NumPy arrays are automatically converted and handled.
    - 3D arrays/tensors that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D arrays/tensors are logged as a single `rr.Scalars` batch under one entity path.
    - Multi-dimensional action arrays are flattened and logged as a single `rr.Scalars` batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            arr = _to_numpy(v)
            if _is_scalar(arr):
                rr.log(key, rr.Scalars(float(arr)))
                observation_paths.add(key)
            elif isinstance(arr, np.ndarray):
                # Squeeze batch dimensions if present (e.g. [1, C, H, W] -> [C, H, W] or [1, D] -> [D])
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                elif arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)

                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))

                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.reshape(-1).astype(float)))
                    observation_paths.add(key)
                elif arr.ndim >= 2:
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        # At record time, the depth unit is inferred from the frame type.
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 and arr.min() >= 0.0:
                            arr_to_log = (arr * 255.0).astype(np.uint8)
                        else:
                            arr_to_log = arr.astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr
                        img_entity = rr.Image(arr_to_log).compress() if compress_images else rr.Image(arr_to_log)
                    elif arr.ndim == 2:
                        img_entity = rr.Image(arr)
                    else:
                        rr.log(key, rr.Scalars(arr.reshape(-1).astype(float)))
                        observation_paths.add(key)
                        continue

                    rr.log(key, img_entity)
                    image_paths.add(key)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            arr = _to_numpy(v)
            if _is_scalar(arr):
                rr.log(key, rr.Scalars(float(arr)))
                action_paths.add(key)
            elif isinstance(arr, np.ndarray):
                # Flatten any (incl. higher-dimensional) array into a single batched Scalars
                rr.log(key, rr.Scalars(arr.reshape(-1).astype(float)))
                action_paths.add(key)

    _ensure_blueprint(observation_paths, action_paths, image_paths)
