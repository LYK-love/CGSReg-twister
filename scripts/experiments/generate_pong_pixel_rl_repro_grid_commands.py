from __future__ import annotations

import argparse
from pathlib import Path


def slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_twister_lines(args, thresholds: list[float], horizons: list[int]) -> list[str]:
    output = Path(args.twister_output)
    lines = [
        "# Auto-generated TWISTER Pong reproduction pixel-RL WM grid.",
        "#",
        "# Reproduction checkpoint:",
        f"#   {args.twister_wm_checkpoint}",
        f"# Reward thresholds: {thresholds}",
        f"# Horizons: {horizons}",
        f"# Total jobs: {len(thresholds) * len(horizons)}",
        "#",
        "# Usage:",
        f"#   tiny-exp-scheduler run {output} --cuda-devices auto --logs-dir {args.twister_scheduler_logs_dir} --verbose --keep-job-tabs",
        "",
    ]
    for horizon in horizons:
        for threshold in thresholds:
            rewq = slug(threshold)
            run_name = (
                "pong_pixel_rl_in_env/logdir/"
                f"twister-repro-h{horizon}-rewq{rewq}"
            )
            envs = [
                f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
                f"CONDA_ENV_NAME={args.twister_conda_env_name}",
                f"WANDB_ENABLED={args.wandb_enabled}",
                f"WANDB_PROJECT={args.twister_wandb_project}",
                f"WANDB_ENTITY={args.wandb_entity}",
                f"WANDB_MODE={args.wandb_mode}",
                f"PIXEL_RL_AC_UPDATES={args.ac_updates}",
                f"PIXEL_RL_ENVS={args.envs}",
                f"PIXEL_RL_BACKUP_EVERY={args.backup_every}",
                f"PIXEL_RL_LOG_EVERY={args.log_every}",
                f"PIXEL_RL_SAVE_EVERY={args.save_every}",
                f"PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY={args.twister_wm_video_every}",
                f"PIXEL_RL_WM_HORIZON={horizon}",
                f"PIXEL_RL_WM_RESPECT_TERMINAL={args.wm_respect_terminal}",
                f"PIXEL_RL_WM_INITIAL_SOURCE={args.wm_initial_source}",
                f"PIXEL_RL_WM_BOOTSTRAP_DATASET={args.twister_bootstrap_dataset}",
                f"PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD={threshold:g}",
                f"PIXEL_RL_EVAL_REAL_EVERY={args.eval_real_every}",
                f"PIXEL_RL_EVAL_REAL_VIDEO_EVERY={args.eval_real_video_every}",
                f"PIXEL_RL_EVAL_REAL_EPS={args.eval_real_eps}",
                f"PIXEL_RL_RESUME={args.resume}",
            ]
            lines.append(
                " ".join(envs)
                + " bash scripts/experiments/pong_pixel_rl_in_env.sh wm "
                + f"{args.twister_wm_checkpoint} {run_name}"
            )
    return lines


def build_storm_lines(args, thresholds: list[float], horizons: list[int]) -> list[str]:
    output = Path(args.storm_output)
    lines = [
        "# Auto-generated STORM Pong reproduction pixel-RL WM grid.",
        "#",
        "# Run from /scorpio/home/luyukuan/projects/oc-storm.",
        "# Reproduction checkpoint:",
        f"#   {args.storm_wm_checkpoint}",
        f"# Reward thresholds: {thresholds}",
        f"# Horizons: {horizons}",
        f"# Total jobs: {len(thresholds) * len(horizons)}",
        "#",
        "# Usage:",
        f"#   tiny-exp-scheduler run {output} --cuda-devices auto --logs-dir {args.storm_scheduler_logs_dir} --verbose --keep-job-tabs",
        "",
    ]
    for horizon in horizons:
        for threshold in thresholds:
            rewq = slug(threshold)
            run_name = (
                "pong_pixel_rl_in_env/logdir/"
                f"backend=wm_wm_family=storm_ckpt=pong_atari100k_repro_base_"
                f"ac20k_envs{args.envs}_backup{args.backup_every}_"
                f"horizon{horizon}_rewq{rewq}_reset={args.wm_initial_source}_"
                f"host={args.host_tag}"
            )
            envs = [
                f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
                f"CONDA_ENV_NAME={args.storm_conda_env_name}",
                f"WANDB_ENABLED={args.wandb_enabled}",
                f"WANDB_PROJECT={args.storm_wandb_project}",
                f"WANDB_ENTITY={args.wandb_entity}",
                f"WANDB_MODE={args.wandb_mode}",
                "STORM_SIZE_CONFIG=base",
                f"PIXEL_RL_CONFIG_PATH={args.storm_config_path}",
                f"PIXEL_RL_AC_UPDATES={args.ac_updates}",
                f"PIXEL_RL_ENVS={args.envs}",
                f"PIXEL_RL_BACKUP_EVERY={args.backup_every}",
                f"PIXEL_RL_LOG_EVERY={args.log_every}",
                f"PIXEL_RL_SAVE_EVERY={args.save_every}",
                f"PIXEL_RL_WM_VIDEO_EVERY={args.storm_wm_video_every}",
                f"PIXEL_RL_WM_HORIZON={horizon}",
                "PIXEL_RL_WM_DISABLE_KV_CACHE=True",
                f"PIXEL_RL_WM_RESPECT_TERMINAL={args.wm_respect_terminal}",
                f"PIXEL_RL_WM_INITIAL_SOURCE={args.wm_initial_source}",
                f"PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD={threshold:g}",
                f"PIXEL_RL_EVAL_REAL_EVERY={args.eval_real_every}",
                f"PIXEL_RL_EVAL_REAL_VIDEO_EVERY={args.eval_real_video_every}",
                f"PIXEL_RL_EVAL_REAL_EPS={args.eval_real_eps}",
                f"PIXEL_RL_RESUME={args.resume}",
            ]
            lines.append(
                " ".join(envs)
                + " bash scripts/experiments/pong_pixel_rl_in_env.sh wm "
                + f"{args.storm_wm_checkpoint} {run_name}"
            )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate TWISTER and STORM Pong reproduction WM pixel-RL grid commands."
    )
    parser.add_argument("--thresholds", default="0.01,0.1,0.5")
    parser.add_argument("--horizons", default="128,512")
    parser.add_argument("--ac-updates", type=int, default=20000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--backup-every", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-real-every", type=int, default=2000)
    parser.add_argument("--eval-real-video-every", type=int, default=10000)
    parser.add_argument("--eval-real-eps", type=int, default=5)
    parser.add_argument("--resume", default="False")
    parser.add_argument("--wm-initial-source", choices=("real", "prior", "dataset"), default="real")
    parser.add_argument("--wm-respect-terminal", default="True")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--wandb-enabled", default="1")
    parser.add_argument("--wandb-entity", default="ssl-lab")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--host-tag", default="scorpio")

    parser.add_argument(
        "--twister-wm-checkpoint",
        default="callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt",
    )
    parser.add_argument("--twister-bootstrap-dataset", default="datasets/converted_from_diamond/pong/train")
    parser.add_argument("--twister-conda-env-name", default="twister")
    parser.add_argument("--twister-wandb-project", default="rl-in-pixel-env")
    parser.add_argument("--twister-wm-video-every", type=int, default=10000)
    parser.add_argument(
        "--twister-output",
        default="scripts/experiments/pong_pixel_rl_twister_repro_grid.commands.txt",
    )
    parser.add_argument(
        "--twister-scheduler-logs-dir",
        default="pong_pixel_rl_in_env/scheduler_logs/twister_repro_grid",
    )

    parser.add_argument(
        "--storm-wm-checkpoint",
        default="/data/luyukuan/projects/oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/ckpt/latest_agent.pth",
    )
    parser.add_argument("--storm-config-path", default="configs/atari_visual.py")
    parser.add_argument("--storm-conda-env-name", default="oc-storm")
    parser.add_argument("--storm-wandb-project", default="rl-in-pixel-env-storm")
    parser.add_argument("--storm-wm-video-every", type=int, default=10000)
    parser.add_argument(
        "--storm-output",
        default="/scorpio/home/luyukuan/projects/oc-storm/scripts/experiments/pong_pixel_rl_storm_repro_grid.commands.txt",
    )
    parser.add_argument(
        "--storm-scheduler-logs-dir",
        default="runs/pong_pixel_rl_in_env/scheduler_logs/storm_repro_grid",
    )
    args = parser.parse_args(argv)

    thresholds = parse_float_csv(args.thresholds)
    horizons = parse_int_csv(args.horizons)

    twister_lines = build_twister_lines(args, thresholds, horizons)
    storm_lines = build_storm_lines(args, thresholds, horizons)

    twister_output = Path(args.twister_output)
    storm_output = Path(args.storm_output)
    twister_output.parent.mkdir(parents=True, exist_ok=True)
    storm_output.parent.mkdir(parents=True, exist_ok=True)
    twister_output.write_text("\n".join(twister_lines) + "\n", encoding="utf-8")
    storm_output.write_text("\n".join(storm_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(thresholds) * len(horizons)} TWISTER jobs to {twister_output}")
    print(f"Wrote {len(thresholds) * len(horizons)} STORM jobs to {storm_output}")


if __name__ == "__main__":
    main()
