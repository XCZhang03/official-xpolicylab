"""Launch the official RoboDojo evaluation environment behind local RPC."""

from __future__ import annotations

import argparse
import importlib
import traceback
from pathlib import Path


class NoPolicyConnection:
    """RoboDojo reset callback only; the MCP controller owns inference."""

    def __init__(self, **kwargs):
        del kwargs

    def call(self, func_name, **kwargs):
        del kwargs
        if func_name != "reset":
            raise RuntimeError("Native autonomous policy loop is disabled")

    def close(self):
        pass


def main():
    import cv2  # noqa: F401 - load before Isaac extension dependency paths
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="build_tower")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--episode-timeout-seconds", type=int, default=0,
                        help="Wall deadline after reset; 0 delegates timing to an external supervisor")
    parser.add_argument("--camera-depth", action="store_true")
    parser.add_argument("--camera-calibration", action="store_true")
    parser.add_argument(
        "--enforce-step-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="legacy mode selector: formal when enabled, exploration otherwise; native limits always apply",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episode_timeout_seconds < 0:
        parser.error('episode-timeout-seconds must be nonnegative')
    args.headless = True
    args.enable_cameras = True

    from env.global_configs import BENCHMARK, ROOT_DIR

    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    import yaml

    task_config_path = registry.task_config_path(
        str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), args.task
    )
    with open(task_config_path, encoding="utf-8") as stream:
        task_values = yaml.safe_load(stream) or {}
    enable_monitor = bool(task_values.get("Articulation"))
    if enable_monitor:
        from src.eval_client.physx_warning_monitor import get_monitor

        get_monitor().start(enabled=True)

    app = AppLauncher(args).app
    env = session = None
    try:
        from env.global_configs import ENV_CONFIG_PATH
        from omegaconf import OmegaConf
        from src.eval_client import eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        from .io import write_json
        from .rpc import serve
        from .session import RoboDojoSession

        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root / "arx_x5.yml"))
        evaluation.update(
            task_name=args.task,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            policy_name="Pi_05",
            additional_info="agent_mcp",
            seed=args.eval_seed,
            physx_monitor_enabled=enable_monitor,
        )
        values = {
            key: load_yaml(str(root / key / (evaluation["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
        if args.camera_depth:
            vision = evaluation.setdefault("observation", {}).setdefault("vision", {})
            vision["approximate_depth"] = False
            vision["depth"] = True
            for annotator in values["camera"].get("annotator", {}).values():
                if annotator.get("enabled", False):
                    annotator["distance_to_image_plane_capture"] = {
                        "type": "distance_to_image_plane",
                        "device": "cpu",
                    }
        if args.camera_calibration:
            vision = evaluation.setdefault("observation", {}).setdefault("vision", {})
            vision["intrinsic_matrix"] = True
            vision["extrinsic_matrix"] = True
        values.update(
            eval_cfg=evaluation,
            deploy_cfg={"port": 1, "policy_name": "Pi_05"},
            task_env=load_yaml(task_config_path),
        )
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]
        # Motion planning is lazy because cuRobo warmup is costly and ordinary
        # joint/EEF episodes do not need it.
        for robot in cfg.robot.robots:
            robot.need_planner = False
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "resolved_config.json",
            OmegaConf.to_container(cfg, resolve=True),
        )
        session = RoboDojoSession(
            env,
            args.output,
            args.task,
            enforce_step_limit=args.enforce_step_limit,
        )
        serve(session, args.port, episode_timeout_seconds=args.episode_timeout_seconds)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if session:
            session._write_summary("server_close")
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
