# Project storage

Use the global `$storage-organization` skill whenever work creates, downloads, moves,
or retains bulky artifacts.

Keep source code and lightweight project files in this workspace. Put active or
temporary artifacts on `/mnt/ssd8` and expose them at the paths expected by the code
with workspace symlinks. Reserve `/mnt/hdd1` for intentionally archived checkpoints,
and do not accumulate permanent checkpoint copies by default.
