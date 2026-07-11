from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nnet
from train_offline import (
    MASK_PRESETS,
    StaticEpisodeDataset,
    _build_log_scalars,
    _evaluate_world_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate frozen TWISTER denoiser/world-model checkpoints on a held-out dataset."
    )
    parser.add_argument("--ckpt-root", default="external_wm_ckpts/offline_ac_cpc_spatial")
    parser.add_argument("--weights", default="0,0.003,0.005,0.01,0.02,0.05,0.1,1,10")
    parser.add_argument("--ckpt-name", default="checkpoints/checkpoints_100000_wm_only.ckpt")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--env-name", default="atari100k-pong")
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--mask-preset", default="mask1", choices=sorted(MASK_PRESETS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dataset", type=str2bool, default=True)
    parser.add_argument("--native-replay-mask-mode", choices=("none", "motion"), default="motion")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/denoiser_eval/offline_ac_cpc_spatial_exp_repro_rb")
    parser.add_argument("--run-label", default="heldout_exp_repro_rb")
    parser.add_argument("--wandb-enabled", type=str2bool, default=False)
    parser.add_argument("--wandb-project", default="twister-denoiser-eval")
    parser.add_argument("--wandb-entity", default="ssl-lab")
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args(argv)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t", "on"}


def main(argv=None):
    args = parse_args(argv)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_num_threads, 4)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights = [_parse_weight(token) for token in _split_csv(args.weights)]
    seeds = [int(token) for token in _split_csv(args.seeds)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = StaticEpisodeDataset(
        args.dataset_path,
        args.batch_length,
        _num_actions(args.env_name),
        cache_dataset=args.cache_dataset,
        native_replay_mask_mode=args.native_replay_mask_mode,
    )
    print(
        f"[denoiser_eval] dataset_files={len(dataset.files)} "
        f"device={device} batches={args.eval_batches} batch_size={args.batch_size} "
        f"seeds={seeds}",
        flush=True,
    )

    jsonl_path = output_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for weight in weights:
            ckpt = Path(args.ckpt_root) / _weight_slug(weight) / args.ckpt_name
            if not ckpt.is_file():
                raise FileNotFoundError(f"Checkpoint not found for w={weight:g}: {ckpt}")
            for seed in seeds:
                _set_seed(seed)
                print(f"[denoiser_eval] loading w={weight:g} seed={seed} ckpt={ckpt}", flush=True)
                model = _load_model(args, weight, ckpt, device)
                scalars = _evaluate_world_model(
                    model,
                    dataset,
                    batch_size=args.batch_size,
                    batches=args.eval_batches,
                    device=device,
                    prefix="eval",
                )
                row = {
                    "run_label": args.run_label,
                    "weight": weight,
                    "weight_slug": _weight_slug(weight),
                    "seed": seed,
                    "checkpoint": str(ckpt),
                    "dataset_path": args.dataset_path,
                    "native_replay_mask_mode": args.native_replay_mask_mode,
                    "batch_size": args.batch_size,
                    "batch_length": args.batch_length,
                    "eval_batches": args.eval_batches,
                    **scalars,
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                _log_wandb(args, row)
                print(_format_progress(row), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = _summarize(rows)
    _write_csv(output_dir / "summary.csv", summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[denoiser_eval] wrote {jsonl_path}", flush=True)
    print(f"[denoiser_eval] wrote {output_dir / 'summary.csv'}", flush=True)


def _load_model(args, weight: float, ckpt: Path, device: torch.device):
    override = {
        "batch_size": int(args.batch_size),
        "L": int(args.batch_length),
        "model_size": args.model_size,
        "num_envs": 1,
        "eval_episodes": 0,
        "spatial_regu_enabled": True,
        "loss_spatial_regu_scale": float(weight),
        "spatial_regu_mask_weights": MASK_PRESETS[args.mask_preset],
        "load_replay_buffer_state_dict": False,
    }
    model = nnet.models.TWISTER(args.env_name, override_config=override).to(device)
    model.compile()
    model.load(str(ckpt), load_optimizer=False, verbose=False, strict=True)
    model.eval()
    return model


def _num_actions(env_name: str) -> int:
    model = nnet.models.TWISTER(
        env_name,
        override_config={
            "num_envs": 1,
            "eval_episodes": 0,
            "load_replay_buffer_state_dict": False,
        },
    )
    num_actions = int(model.env.num_actions)
    del model
    return num_actions


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalar_keys = sorted(
        key
        for key in rows[0].keys()
        if key.startswith("eval/") and all(isinstance(row.get(key), (int, float)) for row in rows)
    ) if rows else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["weight_slug"], []).append(row)
    summary = []
    for slug, group in sorted(grouped.items(), key=lambda item: _parse_weight(item[1][0]["weight"])):
        out: dict[str, Any] = {
            "weight": group[0]["weight"],
            "weight_slug": slug,
            "n": len(group),
        }
        for key in scalar_keys:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            out[f"{key}/mean"] = float(values.mean())
            out[f"{key}/std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = ["weight", "weight_slug", "n"]
    keys = preferred + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _log_wandb(args, row: dict[str, Any]) -> None:
    if not args.wandb_enabled:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        mode=args.wandb_mode,
        name=f"denoiser-eval-{row['weight_slug']}-{args.run_label}-seed{row['seed']}",
        group=args.run_label,
        config={key: value for key, value in row.items() if not key.startswith("eval/")},
        reinit=True,
    )
    run.log({key: value for key, value in row.items() if key.startswith("eval/")})
    run.finish()


def _format_progress(row: dict[str, Any]) -> str:
    keys = [
        "eval/loss",
        "eval/loss_model_image",
        "eval/loss_model_reward",
        "eval/loss_model_discount",
        "eval/loss_unscaled/spatial_regu",
        "eval/loss_scaled/spatial_regu",
    ]
    pieces = [f"w={row['weight_slug']}", f"seed={row['seed']}"]
    for key in keys:
        if key in row:
            pieces.append(f"{key}={float(row[key]):.5g}")
    return "[denoiser_eval] " + " ".join(pieces)


def _split_csv(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _parse_weight(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("w", "").replace("p", "."))


def _weight_slug(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"w{text}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
