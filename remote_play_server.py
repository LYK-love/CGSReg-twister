from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wm_play" / "src"))
COMMON_SRC = ROOT.parent / "wm-play-common" / "src"
if COMMON_SRC.exists():
    sys.path.insert(0, str(COMMON_SRC))

from interactive.twister_adapter import build_twister_session, resolve_policy_checkpoint_path
from wm_play.cli import (
    add_atari_env_args,
    add_device_arg,
    add_pixel_policy_args,
    add_remote_server_args,
    add_wm_bootstrap_dataset_arg,
    add_wm_horizon_arg,
    add_wm_terminal_args,
    add_world_model_checkpoint_args,
    add_world_model_initial_source_arg,
    validate_remote_server_args,
)
from wm_play.server_summary import CheckpointEntry, print_remote_server_summary
from wm_play.web_server import run_web_server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TWISTER remote play server", allow_abbrev=False)
    add_atari_env_args(parser, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0)
    add_world_model_checkpoint_args(parser)
    parser.add_argument("--checkpoint", dest="wm_checkpoint", action="append",
                        help="Deprecated alias for --wm-checkpoint.")
    add_device_arg(parser)
    add_wm_horizon_arg(parser, default=512)
    add_world_model_initial_source_arg(parser)
    add_wm_bootstrap_dataset_arg(parser)
    add_wm_terminal_args(parser, default=True)
    add_pixel_policy_args(parser)
    add_remote_server_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_remote_server_args(args)
    if len(args.wm_name) not in (0, len(args.wm_checkpoint)):
        raise SystemExit("Provide either zero --wm-name values or exactly one name per --wm-checkpoint.")
    if args.policy_checkpoint and not args.additional_policy_controller:
        raise SystemExit("--policy-checkpoint requires --additional-policy-controller.")
    effective_policy_checkpoints = list(args.policy_checkpoint)
    effective_policy_names = list(args.policy_name) or [
        Path(path).expanduser().resolve().parent.name for path in effective_policy_checkpoints
    ]
    resolved_policy_checkpoints = [
        str(resolve_policy_checkpoint_path(x)) for x in effective_policy_checkpoints
    ]
    if len(args.policy_name) not in (0, len(effective_policy_checkpoints)):
        raise SystemExit("Provide either zero --policy-name values or exactly one name per --policy-checkpoint.")

    session = build_twister_session(
        env_name=args.env_name,
        seed=args.seed,
        checkpoint_args=[str(Path(x).expanduser().resolve()) for x in args.wm_checkpoint],
        wm_name_args=list(args.wm_name),
        policy_checkpoint_args=resolved_policy_checkpoints,
        policy_name_args=effective_policy_names,
        additional_policy_controller=args.additional_policy_controller,
        device=args.device,
        wm_horizon=args.wm_horizon,
        wm_respect_terminal=args.wm_respect_terminal,
        wm_initial_source=args.wm_initial_source,
        wm_bootstrap_dataset=args.wm_bootstrap_dataset or None,
    )

    print_remote_server_summary(
        project="TWISTER",
        controller=session.controller,
        tcp_host=args.web_host,
        tcp_port=args.web_port,
        client_command=f"open http://<server-ip>:{args.web_port}",
        real_env=True,
        wm_checkpoints=[
            CheckpointEntry(name=slot.name, path=str(slot.checkpoint))
            for slot in session.wm_slots
        ],
        policy_checkpoints=[
            CheckpointEntry(name=slot.name, path=str(slot.checkpoint))
            for slot in session.policy_slots
        ],
        extras=[
            ("env", args.env_name),
            ("seed", args.seed),
            ("device", args.device),
            ("cuda visible", os.environ.get("CUDA_VISIBLE_DEVICES")),
            ("wm horizon", args.wm_horizon if args.wm_checkpoint else None),
            ("wm initial source", args.wm_initial_source if args.wm_checkpoint else None),
            ("wm bootstrap dataset", args.wm_bootstrap_dataset if args.wm_checkpoint and args.wm_initial_source == "dataset" else None),
            ("wm terminal", "respect" if args.wm_respect_terminal else "ignore"),
        ],
        fps=args.fps,
        size=args.size,
        jpeg_quality=args.jpeg_quality,
        ram_panel=False,
    )

    try:
        run_web_server(args, session)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
