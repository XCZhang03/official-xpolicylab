---
name: wrist-camera-inspection
description: Obtain a clearer close-up of small or occluded task details by positioning an empty arm's wrist camera when the high-camera view is insufficient, and a second calibrated viewpoint for multi-view triangulation.
---

# Wrist-camera inspection

Use an empty arm as a movable inspection camera. Bringing its wrist camera closer
and adjusting the viewing angle can reveal details that are unclear in `cam_high`.
This is physical camera repositioning, not digital enlargement or a zoom API.

- Choose an arm that is not holding an object. Do not release a held object just
  to obtain a view; preserve the other arm's grasp and task state.
- Use the high-camera overview to choose a clear approach. Position it with the
  [motion-toolkit](../motion-toolkit/SKILL.md) planner, then use small,
  observation-guided `approach` adjustments if needed. Keep the arm and gripper
  clear of the table, task objects and other arm; planning success alone does not
  establish clearance from objects or the other arm.
- Inspect the fresh `cam_left_wrist` or `cam_right_wrist` image after each move.
  Adjust distance and angle until the relevant detail is clear, avoiding gripper
  occlusion. Stop approaching once the view is useful, or if clearance is uncertain.
- Wrist cameras resolve several times finer detail than `cam_high` up close.
  At 0.15–0.3 m that is 0.4–0.8 mm per pixel, against 2–3.5 mm. Their pose is
  known from the measured link6 pose, so a close view is also a precise
  calibrated ray.
- For 3-D positions, pair the close view with `cam_high` or the other wrist
  camera and triangulate (see
  [RGB position calibration](../rgb-position-calibration/SKILL.md#triangulate-from-several-views)).
  Choose the pose so the target is sharp and unoccluded, and so the ray meets
  the other camera's ray at a wide angle (at least 10–15°).
- One image alone is still not a metric 3-D measurement; it gives a ray. Keep
  each image with the `eef_*` pose from the same reply. Recheck after moving the
  camera.

Inspection motions consume episode steps. Move only when the clearer view can
resolve a task-relevant uncertainty, then return the empty arm to a clear pose
if it obstructs the next manipulation.
