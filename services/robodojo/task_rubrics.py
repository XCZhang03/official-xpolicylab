"""Public task criteria, separate from private runtime evaluation state."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _snapshot():
    return json.loads(Path(__file__).with_suffix(".json").read_text())


def task_rubric_context(task: str, *, include_source_url: bool = True) -> str:
    canonical = task.lower()
    if canonical.endswith("_random"):
        canonical = canonical.removesuffix("_random")
    snapshot = _snapshot()
    rubric = snapshot["tasks"].get(canonical)
    if rubric is None:
        raise ValueError(f"No reviewed public scoring rubric for task {task!r}; refresh the wiki snapshot before launching")
    lines = ["## Public task scoring rubric", "", f"Task: `{task}`",
             f"Source: {rubric['source_url'] if include_source_url else 'bundled public wiki snapshot'}",
             f"Wiki snapshot: {snapshot['retrieved_at_utc']}", "",
             "These are public scoring criteria, not live feedback. Use them to",
             "understand the goal and judge visible evidence. Native scores, hidden",
             "state, and evaluator implementation remain unavailable. Formal evaluation",
             "still ends the attempt; this rubric does not authorize retries.", ""]
    if canonical != task.lower():
        lines += ["The randomized variant uses the base task's published rubric;",
                  "the wiki does not provide a separate scoring table for this variant.", ""]
    if canonical == "make_kong":
        # Reviewed visual-demo clarification; preserve the published table below.
        lines += ['Demo clarification: Use the supplied demonstration images to identify the "target tile" in the rubric, which is moved from the left pile to stand upright at the right end of the front row, next to its rightmost tile, with its symbol oriented the same way.', ""]
    elif canonical == "make_toast":
        lines += [
            "Demo clarification: Use the demonstration images to guide insertion "
            "depth and orientation, seat one bread slice in each toaster slot "
            "rather than on the rim, keep all four slices upright with exactly two "
            "slices on the shelf, and press the lever fully down and ensure it stays "
            "down after withdrawing the gripper and returning both arms home.", "",
        ]
    elif canonical == "pour_by_language":
        lines += [
            "Demo clarification: Complete the language-specified pours in order. "
            "Tilt each bottle far enough and hold it tilted long enough to drain it "
            "nearly completely into its assigned bowl before returning it upright. "
            "Aim to nearly fill the bowl without spilling, but also check that the "
            "bottle is nearly empty; some visible liquid in the bowl is not enough. "
            "After the liquid has successfully transferred into its assigned bowl, "
            "return that bottle upright, then return both robot arms to their initial "
            "poses before proceeding to the next pour. Repeat these checkpoints after "
            "each pour, including the final one; matching only the final filled-bowl "
            "image is not sufficient for full credit.", "",
            "Checkpoint detail: Upright and return-home checks depend on transitions. "
            "Both arms must leave their home regions between return-home checkpoints; "
            "they must be outside those regions at the same time before returning. "
            "While the working arm is away, raise the idle arm more than 15 cm above "
            "its initial end-effector height, with margin and safe clearance, to "
            "move it outside the home-position tolerance. A small lift or keeping "
            "one arm parked at home can prevent a new checkpoint.", "",
        ]
    elif canonical == "swap_blocks":
        lines += [
            "Demo clarification: Either block may move first. Move it onto the empty "
            "mat, move the other block onto the newly vacated mat, then move the first "
            "block onto the other block's original mat. Lift the moved block more than "
            "3 cm above its resting position on each move; sliding alone does not "
            "satisfy the lift checkpoints. Placement requires block and mat root "
            "positions to be less than 3 cm apart in 3D, not just aligned from above.", "",
            "Button detail: Press and release once after each placement, for exactly "
            "three presses total. A counted press moves from above 95% to below 50% "
            "of the button's joint travel. Although a release checkpoint accepts "
            "above 90%, the next press cannot be counted until the button rebounds "
            "above 95%. Clear the button and let it rebound fully between presses; "
            "a fourth counted press fails the task.", "",
            "Finish detail: After the third press, release the button and return "
            "both arms to their initial end-effector poses, within 15 cm on each "
            "position axis and 20 degrees in orientation. This is a final-state "
            "check, not a return-home transition; an unused arm may remain at home.", "",
        ]
    lines += ["| Published score | Condition |", "|---|---|"]
    for row in rubric["criteria"]:
        lines.append(f"| {row['points']} | {row['condition'].replace('|', '&#124;')} |")
    lines += ["", "If evaluation fails and the cause is unclear, verify each rubric condition "
              "and clarification point by point, comparing the supplied demonstration image, "
              "when present, with a current image of your final state."]
    return "\n".join(lines) + "\n"
