from __future__ import annotations

import argparse
from pathlib import Path


def weight_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate TWISTER Pong pixel-RL-in-WM reward-quantization sweep commands."
    )
    parser.add_argument(
        "--wm-checkpoint",
        default="callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt",
    )
    parser.add_argument("--thresholds", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--run-root", default="pong_pixel_rl_in_env/logdir")
    parser.add_argument("--run-prefix", default="twister-repro")
    parser.add_argument("--ac-updates", type=int, default=20000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--backup-every", type=int, default=15)
    parser.add_argument("--wm-horizon", type=int, default=512)
    parser.add_argument("--wm-initial-source", choices=("real", "prior", "dataset"), default="real")
    parser.add_argument("--wm-bootstrap-dataset", default="datasets/converted_from_diamond/pong/train")
    parser.add_argument("--respect-terminal", default="True")
    parser.add_argument("--eval-real-every", type=int, default=2000)
    parser.add_argument("--eval-real-video-every", type=int, default=10000)
    parser.add_argument("--eval-real-eps", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--wm-rollout-video-every", type=int, default=10000)
    parser.add_argument("--resume", default="False")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--conda-env-name", default="twister")
    parser.add_argument("--wandb-enabled", default="1")
    parser.add_argument("--wandb-project", default="rl-in-pixel-env")
    parser.add_argument("--wandb-entity", default="ssl-lab")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--host-tag", default="scorpio")
    parser.add_argument(
        "--output",
        default="scripts/experiments/pong_pixel_rl_rewq_sweep.commands.txt",
    )
    parser.add_argument(
        "--scheduler-logs-dir",
        default="pong_pixel_rl_in_env/scheduler_logs/rewq_sweep",
    )
    args = parser.parse_args(argv)

    thresholds = parse_float_csv(args.thresholds)
    output = Path(args.output)
    lines = [
        "# Auto-generated TWISTER Pong pixel-RL reward-quantization sweep for tiny-exp-scheduler.",
        "#",
        "# Run from /scorpio/home/luyukuan/projects/twister with conda env twister.",
        f"# WM checkpoint: {args.wm_checkpoint}",
        f"# Thresholds: {thresholds}",
        f"# AC updates: {args.ac_updates}",
        f"# Parallel envs: {args.envs}",
        f"# Backup every: {args.backup_every}",
        f"# WM horizon: {args.wm_horizon}",
        f"# WM initial source: {args.wm_initial_source}",
        f"# WM bootstrap dataset: {args.wm_bootstrap_dataset}",
        f"# Respect terminal: {args.respect_terminal}",
        f"# W&B enabled: {args.wandb_enabled}",
        f"# W&B project: {args.wandb_project}",
        f"# Total jobs: {len(thresholds)}",
        "#",
        "# Notes:",
        "# - Only WM reward is thresholded into {-1, 0, +1}.",
        "# - Terminal is not quantized; it remains a boolean end signal from the WM head.",
        "#",
        "# Usage:",
        f"#   tiny-exp-scheduler run {output} --cuda-devices auto --logs-dir {args.scheduler_logs_dir} --verbose --keep-job-tabs",
        "",
    ]
    for threshold in thresholds:
        rewq = weight_slug(threshold)
        run_name = (
            f"{args.run_root}/"
            f"{args.run_prefix}-h{args.wm_horizon}-rewq{rewq}"
        )
        env_vars = [
            f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
            f"CONDA_ENV_NAME={args.conda_env_name}",
            f"WANDB_ENABLED={args.wandb_enabled}",
            f"WANDB_PROJECT={args.wandb_project}",
            f"WANDB_ENTITY={args.wandb_entity}",
            f"WANDB_MODE={args.wandb_mode}",
            f"PIXEL_RL_AC_UPDATES={args.ac_updates}",
            f"PIXEL_RL_ENVS={args.envs}",
            f"PIXEL_RL_BACKUP_EVERY={args.backup_every}",
            f"PIXEL_RL_LOG_EVERY={args.log_every}",
            f"PIXEL_RL_SAVE_EVERY={args.save_every}",
            f"PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY={args.wm_rollout_video_every}",
            f"PIXEL_RL_WM_HORIZON={args.wm_horizon}",
            f"PIXEL_RL_WM_RESPECT_TERMINAL={args.respect_terminal}",
            f"PIXEL_RL_WM_INITIAL_SOURCE={args.wm_initial_source}",
            f"PIXEL_RL_WM_BOOTSTRAP_DATASET={args.wm_bootstrap_dataset}",
            f"PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD={threshold:g}",
            f"PIXEL_RL_EVAL_REAL_EVERY={args.eval_real_every}",
            f"PIXEL_RL_EVAL_REAL_VIDEO_EVERY={args.eval_real_video_every}",
            f"PIXEL_RL_EVAL_REAL_EPS={args.eval_real_eps}",
            f"PIXEL_RL_RESUME={args.resume}",
        ]
        lines.append(
            " ".join(env_vars)
            + " bash scripts/experiments/pong_pixel_rl_in_env.sh wm "
            + f"{args.wm_checkpoint} {run_name}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(thresholds)} commands to {output}")


if __name__ == "__main__":
    main()
