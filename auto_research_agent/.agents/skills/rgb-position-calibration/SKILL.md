---
name: rgb-position-calibration
description: Estimate RoboDojo positions from RGB images without depth, using the static camera model derived from the task configs, multi-view triangulation across cam_high and the wrist cameras, known table and object geometry from task_source, and measured end-effector motion.
---

# RGB position calibration

Use current RGB images and three kinds of static knowledge:
- the camera model;
- the table geometry;
- object sizes from `task_source/`.

Ground motion with measured robot poses. This is an estimate-and-correct procedure,
not a depth measurement. Observations carry no depth or calibration. Do not seek
simulator files or hidden object state.

## Read the task-specific reference

Read [CALIBRATION.md](CALIBRATION.md) in this skill folder before using
table geometry. It supplies:
- the current task's reviewed table dimensions, top height and fixed 3-D
  reference points, or a statement that no reference is available;
- the static camera model: intrinsics `K` and camera poses for `cam_high`
  (fixed) and the wrist cameras (fixed relative to link6).
Do not assume dimensions, borrow geometry from another task, or seek simulator
files when a reference is absent. Use measured-motion feedback instead.

Reference coordinates use environment-origin metres. Confirm that the current
scene matches before using them; fixed geometry does not imply fixed image
pixels or object positions. Recheck each episode and environment.

`eef_positions` and `eef_quaternions_wxyz` are measured `link6` poses,
ordered [left, right], in the same environment frame. They are not fingertip,
grasp-centre, camera or object poses. Use achieved poses, not commanded targets;
the visible gripper feature may be offset from link6 and move when the gripper
opens or its orientation changes.

## Camera model: pixels to rays

The cameras are fixed properties of the task configuration. `task_source/`
contains the configs they come from:
- `source/env_cfg/camera/camera_config.yml`: cam_high pose;
- `source/env_cfg/camera/template.py`: lens parameters.

CALIBRATION.md gives the derived, simulator-verified matrices.
- **Pixels are square:** use fy = fx. Do not compute fy from the configured
  vertical aperture.
- **Pixel to ray:** a pixel back-projects to a ray. Intersect it with a known
  horizontal plane to get a 3-D point. The formulas are in CALIBRATION.md.
- **Plane height:** use the table top for points on the table, and the table top
  plus an object's height for its top face.
- **Wrist cameras:** compose the measured link6 pose with the fixed mount
  transform. Their poses are only as current as the observation they came with.
- **Check before use:** project a measured link6 position into a fresh image
  and confirm it lands on the gripper. After that, residual pixel error mostly
  comes from which image feature you picked and which height you assumed.

A ray–plane point is only as good as the assumed height; a wrong height shifts
the point along the ray. When the height is unknown or matters (elevated
surfaces, held or tilted objects, insertion), triangulate from two or more views
instead (below). Otherwise verify with a second camera or a measured-motion check
before contact.

## Triangulate from several views

The three cameras are calibrated in one environment frame. Two rays at the same
physical point fix its 3-D position with **no height assumption**.

**Use all three cameras.** They are all 640 x 480, but they resolve very
different detail:
- `cam_high` sees the whole scene from about 0.6–1 m with fx 288. That is about
  2–3.5 mm per pixel.
- Each wrist camera has fx 397 and works much closer. At 0.15–0.3 m that is
  0.4–0.8 mm per pixel, several times finer.

Use `cam_high` for the overview and for one ray. Use a wrist camera for the
precise ray. When both arms are free, or one arm is holding and the other is
empty, the two wrist cameras can also view the same target from two sides. Place
an empty arm's camera with the
[wrist-camera inspection](../wrist-camera-inspection/SKILL.md) procedure, then
triangulate with that view. A single empty wrist camera moved to two poses also
works if the target does not move between the two observations.

**Geometry**
- Rays that are nearly parallel give poor depth. Aim for at least 10–15°
  between rays; 30–90° is better. `cam_high` plus a nearby wrist camera usually
  gives a wide angle.
- The per-view error is roughly the distance from the camera to the target
  divided by fx, per pixel of error. The depth error is about that divided by
  the sine of the ray angle.

**Correspondence is the hard part**
- The pixels must mark the same physical point: a corner, an edge end, a
  handle tip, a slot end, a colour boundary. Silhouette or blob centroids are
  different points in different views.
- For a known object, fit its box or mesh pose from `task_source` to its outline
  in every available view together. The fitted pose is what you use. Scoring the
  projected outline in each view avoids hand-matching points and also gives
  yaw and tilt.
- Use only images and `eef_*` poses from the same reply. A wrist camera's pose
  goes stale as soon as its arm moves.

**Gemini as the pointer (test it first).** Hand-tuned feature detectors overfit
the explored scenes. In an autonomous controller, [gemini](../gemini/SKILL.md) may
generalize better at marking a named point on the target object in each view ("the
tip of the lever", "the centre of the slot opening"). Triangulation needs every
view to mark the **same physical point on the target object**, so test exactly
that in exploration before relying on it:
- **Cross-view consistency.** Ask for the same named point on the target object
  in `cam_high` and one or both wrist views of one scene. Triangulate the answers.
  Rays that meet within a few millimetres, with small reprojection errors, mean
  Gemini chose the same physical point in every view. A centimetre miss means it
  chose different points, for example another edge, the near versus the far
  corner, or the object's centre in one view and its tip in another.
- **Spread and accuracy.** Repeat identical calls to measure the spread per view.
  On a few frames, compare the triangulated point with a reference for the same
  target point from a careful model fit or edge measurement. Cover several
  layouts, viewpoints and object instances, not one scene.
- **Choose what to ask for.** Prefer points that look distinct from every view:
  tips, ends, corners with a unique role. Symmetric features (one of four
  identical corners, a rim) are easy to mismatch across views. If one phrasing
  mismatches, try a more specific description before giving up.
- **Expect the wrist view to matter most.** A pixel error costs about 0.5 mm in a
  close wrist view but about 3 mm in `cam_high`. Gemini's typical error of several
  to tens of pixels may be usable in a wrist view and too coarse in the high
  view.
- **In the controller, the residual is the check.** Reject an answer whose rays
  miss. Re-ask, add a view or fall back. Use an accepted point as a 3-D seed, and
  refine it locally with edges or a model fit and with closed-loop motion before
  contact. Each view costs a call, so count the calls against the session budget.
- Record the measured consistency per camera pair and target point in memory, and
  keep a non-Gemini fallback.

**Helper script:** [triangulation.py](triangulation.py) in this skill folder.
It needs only NumPy and carries the CALIBRATION.md camera models. For a controller,
copy it into `code/<project>/`; bundles cannot read the skill folder. In
development, import it from here:

```python
import sys; sys.path.insert(0, "/workspace/.agents/skills/rgb-position-calibration")
from triangulation import triangulate, project, pixel_ray, ray_plane, camera_pose

res = triangulate([
    {"camera": "cam_high", "pixel": [u1, v1], "meta": meta_a},
    {"camera": "cam_right_wrist", "pixel": [u2, v2], "meta": meta_b},
])
res["point"], res["ray_distances_m"], res["reprojection_px"], res["max_ray_angle_deg"], res["warnings"]
```

`meta` is the parsed reply the image came with; wrist cameras need it. The same
file also runs from the shell: `python triangulation.py views.json`. The core is
the least-squares point of rays (o_i, unit d_i), which solves
`sum(I - d_i d_i^T) p = sum(I - d_i d_i^T) o_i`.

**Accept a point only on evidence**
- The rays should pass within a few millimetres of the point.
- Reprojection errors should be a few pixels at most.
- The point must be in front of every camera.
- A large miss usually means a wrong correspondence or a pose from the wrong
  reply. Fix that; do not average it away.
- Agreement between two views can still be a coincidence. Confirm with a third
  view or a small measured motion before contact.
- Record the views, residuals and outcome in memory. The check above, projecting
  a measured link6 position onto the gripper in each camera, also bounds
  wrist-mount error.

## Use object sizes to point at objects

`task_source/assets/<Type>/<category>/<model>/metadata.json` gives each object's
bounding box in its own frame (`aligned_bbox.extents`, metres, z up for objects
resting upright), plus `active`/`passive` place and functional frames. `mesh.npz`
has its full triangle mesh. Use them to turn image evidence into 3-D targets:

- **Height:** an upright object's top face lies at the support height plus its
  z extent. Intersect the pixel ray of the top face's centre with that plane.
  Intersect a footprint or base edge ray with the support plane instead.
- **Scale and identity:** project the object's box or mesh at a candidate pose.
  Its image size must match what you see. Use this to tell similar objects apart,
  to reject a wrong height assumption, and to spot a toppled object.
- **Yaw:** estimate the object's orientation from long visible edges or the
  outline, and compare it with the projected box or mesh at that yaw.
- **Functional points:** once you have a pose estimate, map a functional frame
  (slot, handle, button) from object frame to environment frame. That gives a
  concrete grasp or insertion target. Check it with the wrist camera before
  contact.
- **Symmetric objects:** a symmetric footprint leaves yaw ambiguous; resolve it
  from visible asymmetric features.

These give estimates from the current images. Re-acquire every object from
fresh observations each episode and after any contact. Sizes are fixed facts and
may be copied into the bundle, but object poses never are.

## Start with the table, then refine with the arm

1. Inspect a fresh high-camera image. Identify the actual tabletop edges, not
   object edges, shadows or the table's underside. Establish which edges are
   x/y using the robot and a small known-direction motion if needed; do not
   assume image left/right or up/down equals an environment axis.
2. Use the known bounds for a coarse position estimate. The camera model
   already gives a table-plane mapping. As an independent check, when four
   tabletop corners are clearly identified and correctly matched to their XY
   coordinates, an image-to-table-plane homography also estimates points **on
   that plane**. Perspective generally rules out one global metres-per-pixel
   scale. Do not invent occluded corners to obtain a precise-looking fit.
3. Cross-check the estimate with the measured empty arm and fresh views. Choose
   a clear approach above the scene using the
   [motion-toolkit](../motion-toolkit/SKILL.md) planner. Allow for the entire
   gripper and any held object, not just link6; placing link6 above the supplied
   tabletop height alone is not clearance.
4. Near the intended operation, make small, independently directed XY motions
   at a fixed safe height, orientation and gripper opening. Compare the same
   visible gripper feature in fresh high-camera images with the **achieved**
   EEF displacement. This establishes local axis direction and scale. Use only
   moves with clear visibility and clearance; do not touch the table to calibrate.
5. Apply a small correction and observe again. Shorten corrections near objects,
   descend gradually only with sufficient clearance evidence, and use
   [wrist-camera inspection](../wrist-camera-inspection/SKILL.md) for ambiguous
   alignment or contact. Verify grasp/placement visually, not from coordinates
   alone. Use `robodojo_toolkit.pose_math` ([motion-toolkit](../motion-toolkit/SKILL.md)) to form motion targets.

A table-plane mapping does not localize elevated surfaces such as bowl rims,
bottle caps or grippers at a different height: perspective introduces parallax.
Triangulate such points from several views instead. A pixel overlap in one image
does not prove contact or 3-D alignment. Do not treat a table-plane XY estimate as
an exact EEF target.

## Optional local numerical fit

For small motions at fixed height/orientation, fit
`delta_pixel ≈ J * [delta_x, delta_y]` from at least two independent measured
motions of the same gripper feature. Use additional samples and a held-out small
move to check prediction error before trusting the fit. An ill-conditioned fit,
occlusion or inconsistent tracking calls for a better view, not larger motions.

This local fit estimates displacement, not an absolute link6-to-pixel origin or
object depth. A target feature must be at a comparable height for its pixel error
to give a useful correction. Refit after significant height/orientation changes,
large translations or camera movement; a moving wrist camera invalidates a
fixed-camera fit.

Record camera, episode/environment, measured poses, tracked feature, working
height/orientation and validation error in memory. Reuse a calibration only after
checking it against the current scene. Calibration motions consume normal steps;
do not reset solely to collect them.
