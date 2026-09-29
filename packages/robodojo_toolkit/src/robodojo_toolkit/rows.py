"""Build 25 Hz 14-D joint rows for robodojo_step, and parse robodojo_* replies.

A row is [left joint1..6, left gripper, right joint1..6, right gripper]: absolute joint
targets in radians and gripper openings in [0, 1] (0 closed, 1 open). Each row is one
environment step. Official observations report the last commanded gripper opening.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

from .kinematics import DualArm

GRIPPER = {"left": 6, "right": 13}


def parse_reply(reply, *, images=True):
    """(meta, {camera: HxWx3 uint8}) from a robodojo_observe/status/step reply."""
    content = reply["content"]
    meta = json.loads(next(block["text"] for block in content if block["type"] == "text"))
    frames = {}
    if images:
        from PIL import Image
        for attachment in meta.get("attachments", []):
            if attachment.get("kind", "rgb") != "rgb":
                continue
            block = content[attachment["content_index"]]
            frames[attachment["camera"]] = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(block["data"]))).convert("RGB"))
    return meta, frames


def hold(state, count=1):
    """Keep the current targets for ``count`` steps (lets motion settle, waits)."""
    state = np.asarray(state, dtype=np.float32).reshape(14)
    return np.repeat(state[None], int(count), axis=0)


def set_gripper(state, arm, opening, count=10):
    """Ramp one gripper to ``opening`` over ``count`` steps, arms held."""
    state = np.asarray(state, dtype=np.float32).reshape(14)
    if not 0.0 <= float(opening) <= 1.0:
        raise ValueError("opening must be in [0, 1]")
    rows = hold(state, count)
    index = GRIPPER[arm]
    rows[:, index] = np.linspace(state[index], float(opening), int(count) + 1)[1:]
    return rows


def approach(arm, state, target, *, max_steps=50, position_tol_m=0.001, rotation_tol_rad=0.005, kinematics=None):
    """Bounded DLS steps toward ``target`` (the step_eef behaviour), predicted open-loop.

    Each row moves at most 0.05 rad per joint and 2 cm / 0.1 rad per step. Rows are
    predicted from kinematics, not measured: execute short chunks and re-observe near
    contact. Returns (rows, diagnostics); rows may be empty if already at the target.
    """
    kinematics = kinematics or DualArm()
    state = np.asarray(state, dtype=np.float32).reshape(14)
    offset = 0 if arm == "left" else 7
    joints = state[offset:offset + 6].astype(float)
    rows, diagnostics = [], {}
    for _ in range(int(max_steps)):
        next_joints, diagnostics = kinematics.step_toward(arm, joints, target)
        if (diagnostics["target_position_error_m"] <= position_tol_m
                and diagnostics["target_rotation_error_rad"] <= rotation_tol_rad):
            break
        joints = np.asarray(next_joints, dtype=float)
        row = state.copy()
        row[offset:offset + 6] = joints
        rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 14), diagnostics
