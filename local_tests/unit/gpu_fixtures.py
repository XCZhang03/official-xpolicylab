"""Shared real-GPU fixtures for native Isaac integration tests (not collected as tests)."""

import base64
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def gpu_runtime():
    root = Path(__file__).resolve().parents[2]
    python = Path(os.environ.get("ROBODOJO_PYTHON", root / "runtime/envs/robodojo/bin/python"))
    if not python.is_file() or not (root / "RoboDojo/task/RoboDojo/config/make_kong.yml").is_file():
        pytest.skip("GPU integration needs the bootstrapped RoboDojo runtime and source")
    if not shutil.which("nvidia-smi"):
        pytest.skip("NVIDIA driver unavailable")
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode:
            pytest.skip("NVIDIA driver inaccessible (a sandbox may hide host GPUs)")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = None if visible is None else set(visible.split(","))
        candidates = []
        for line in probe.stdout.splitlines():
            index, uuid, free, utilization = map(str.strip, line.split(","))
            if allowed is not None and index not in allowed and uuid not in allowed:
                continue
            if int(free) >= 24 * 1024 and int(utilization) < 20:
                candidates.append((int(free), uuid))
        if not candidates:
            pytest.skip("No visible idle GPU with >=24 GiB free for Isaac")
        gpu = max(candidates)[1]
        dependency = subprocess.run(
            [str(python), "-c", "import importlib.util; import torch; "
             "assert importlib.util.find_spec('isaaclab'); "
             "assert importlib.util.find_spec('isaacsim'); assert torch.cuda.is_available()"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
            capture_output=True, text=True, timeout=30,
        )
        if dependency.returncode:
            pytest.skip("Native Python needs Isaac Lab, Isaac Sim, and usable CUDA: " + dependency.stderr[-500:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"GPU preflight unavailable: {exc}")
    runtime = (root / "runtime").resolve()
    if not runtime.is_relative_to(Path("/mnt/ssd8")) or not os.access(runtime, os.W_OK):
        pytest.skip("GPU artifacts require writable SSD-backed runtime/ on /mnt/ssd8")
    output = runtime / "integration-tests"
    output.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gpu-integration-", dir=output))
    print(f"GPU integration artifacts: {output}")
    return root, str(python), gpu, output


def _check_images(observation, encode):
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        rgb = np.asarray(observation[camera])[..., :3]
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
        assert rgb.std() > 1, f"Blank render: {camera}"
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encode(rgb)))))
        np.testing.assert_array_equal(decoded, rgb)
        depth = np.asarray(observation[f"{camera}_depth_m"])
        assert depth.shape[:2] == rgb.shape[:2]
        assert np.any(np.isfinite(depth) & (depth > 0)), f"No valid depth: {camera}"


def _errors(actual, predicted):
    qa = np.asarray(actual["eef_quaternions_wxyz"], dtype=float)
    qp = np.asarray(predicted["eef_quaternions_wxyz"], dtype=float)
    qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
    qp /= np.linalg.norm(qp, axis=-1, keepdims=True)
    joints = [*range(6), *range(7, 13)]
    return {
        "joint_rad": float(np.max(np.abs(np.asarray(actual["states"])[joints] - np.asarray(predicted["states"])[joints]))),
        "gripper": float(np.max(np.abs(np.asarray(actual["states"])[[6, 13]] - np.asarray(predicted["states"])[[6, 13]]))),
        "eef_m": float(np.max(np.linalg.norm(np.asarray(actual["eef_positions"]) - np.asarray(predicted["eef_positions"]), axis=-1))),
        "eef_rad": float(np.max(2 * np.arccos(np.clip(np.abs(np.sum(qa * qp, axis=-1)), 0, 1)))),
    }


@pytest.fixture
def agent_gpu():
    """UUID of an idle visible GPU with 4 GiB free, for the agent container's CUDA smoke."""
    if not shutil.which('nvidia-smi'):
        pytest.skip('NVIDIA driver unavailable')
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if query.returncode:
        pytest.skip('GPU inaccessible')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None if visible is None else set(visible.split(','))
    candidates = []
    for line in query.stdout.splitlines():
        index, gpu, free, usage = map(str.strip, line.split(','))
        if (allowed is None or index in allowed or gpu in allowed) and int(free) >= 4096 and int(usage) < 20:
            candidates.append((int(free), gpu))
    if not candidates:
        pytest.skip('No idle visible GPU with 4 GiB free')
    return max(candidates)[1]
