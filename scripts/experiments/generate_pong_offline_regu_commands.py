from __future__ import annotations

import argparse
from pathlib import Path


def weight_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate TWISTER Pong offline spatial-reg sweep commands.")
    parser.add_argument("--dataset-path", default="/SSD_RAID0/lyk/shared_replay/diamond_pong_for_simulus_twister/twister/train")
    parser.add_argument("--eval-dataset-path", default="")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--model-sizes", default="base")
    parser.add_argument("--wm-initial-source", choices=("real", "prior", "dataset"), default="real")
    parser.add_argument("--spatial-weights", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--mask-presets", default="mask1")
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-root", default="runs/pong_offline_regu_sweep/logdir")
    parser.add_argument("--run-prefix", default="twister_pong_offline_regu")
    parser.add_argument("--output", default="scripts/experiments/pong_offline_regu_base_mask1.commands.txt")
    parser.add_argument("--scheduler-logs-dir", default="runs/pong_offline_regu_sweep/scheduler_logs")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--conda-env", default="twister")
    parser.add_argument("--wandb-enabled", default="1")
    parser.add_argument("--wandb-project", default="twister")
    parser.add_argument("--wandb-entity", default="ssl-lab")
    parser.add_argument("--wandb-mode", default="online")
    args = parser.parse_args(argv)

    weights = parse_float_csv(args.spatial_weights)
    model_sizes = parse_str_csv(args.model_sizes)
    masks = parse_str_csv(args.mask_presets)
    output = Path(args.output)
    lines = [
        "# Auto-generated TWISTER offline Pong model-size and spatial-regularization jobs for tiny-exp-scheduler.",
        "#",
        "# Run from the TWISTER repository root with conda env twister.",
        f"# Dataset path: {args.dataset_path}",
        f"# Eval dataset path: {args.eval_dataset_path or '<disabled>'}",
        f"# Eval every: {args.eval_every}",
        f"# Eval batches: {args.eval_batches}",
        f"# Eval batch size: {args.eval_batch_size or args.batch_size}",
        f"# Model sizes: {model_sizes}",
        f"# WM initial source: {args.wm_initial_source}",
        f"# Mask presets: {masks}",
        f"# Spatial weights: {weights}",
        f"# Train steps: {args.train_steps}",
        f"# Batch size: {args.batch_size}",
        f"# Batch length: {args.batch_length}",
        f"# Torch CPU threads per job: {args.torch_num_threads}",
        f"# Log every: {args.log_every}",
        f"# W&B enabled: {args.wandb_enabled}",
        f"# W&B project: {args.wandb_project}",
        f"# Total jobs: {len(model_sizes) * len(masks) * len(weights)}",
        "#",
        "# Usage:",
        f"#   tiny-exp-scheduler run {output} --cuda-devices auto --logs-dir {args.scheduler_logs_dir} --verbose --keep-job-tabs",
        "",
    ]
    job_count = 0
    for model_size in model_sizes:
        for mask in masks:
            for weight in weights:
                run_name = f"{args.run_prefix}_model={model_size}_{mask}_spatial_{weight_slug(weight)}"
                wandb_args = [
                    f"--wandb-enabled {args.wandb_enabled}",
                    f"--wandb-project {args.wandb_project}",
                    f"--wandb-mode {args.wandb_mode}",
                ]
                if args.wandb_entity:
                    wandb_args.append(f"--wandb-entity {args.wandb_entity}")
                eval_args = []
                if args.eval_dataset_path and args.eval_every > 0:
                    eval_args = [
                        f"--eval-dataset-path {args.eval_dataset_path}",
                        f"--eval-every {args.eval_every}",
                        f"--eval-batches {args.eval_batches}",
                    ]
                    if args.eval_batch_size > 0:
                        eval_args.append(f"--eval-batch-size {args.eval_batch_size}")
                train_args = [
                    f"--dataset-path {args.dataset_path}",
                    f"--run-root {args.run_root}",
                    f"--run-name {run_name}",
                    f"--model-size {model_size}",
                    f"--wm-initial-source {args.wm_initial_source}",
                    f"--spatial-regu-weight {weight:g}",
                    f"--mask-preset {mask}",
                    f"--train-steps {args.train_steps}",
                    f"--batch-size {args.batch_size}",
                    f"--batch-length {args.batch_length}",
                    *eval_args,
                    f"--torch-num-threads {args.torch_num_threads}",
                    f"--log-every {args.log_every}",
                    *wandb_args,
                ]
                lines.append(
                    f"bash -lc 'cd {args.project_root} && "
                    f"OMP_NUM_THREADS={args.torch_num_threads} "
                    f"MKL_NUM_THREADS={args.torch_num_threads} "
                    f"OPENBLAS_NUM_THREADS={args.torch_num_threads} "
                    f"NUMEXPR_NUM_THREADS={args.torch_num_threads} "
                    f"TWISTER_TORCH_NUM_THREADS={args.torch_num_threads} "
                    f"conda run --no-capture-output -n {args.conda_env} python -u train_offline.py "
                    + " ".join(train_args)
                    + "'"
                )
                job_count += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {job_count} commands to {output}")


if __name__ == "__main__":
    main()
