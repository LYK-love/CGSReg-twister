from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import nnet


MASK_PRESETS = {
    "mask1": {"mask1": 1.0, "mask2": 0.0, "mask3": 0.0},
    "ball": {"mask1": 1.0, "mask2": 0.0, "mask3": 0.0},
    "mask1_mask3": {"mask1": 1.0, "mask2": 0.0, "mask3": 1.0},
    "ball_player": {"mask1": 1.0, "mask2": 0.0, "mask3": 1.0},
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def weight_slug(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline TWISTER Pong world-model training with spatial regularization.")
    parser.add_argument("--dataset-path", required=True, help="Directory or file containing offline episodes.")
    parser.add_argument("--env-name", default="atari100k-pong")
    parser.add_argument("--model-size", default="base", choices=("S", "base", "size50m", "size100m", "size200m", "size400m"))
    parser.add_argument("--wm-initial-source", choices=("real", "prior", "dataset"), default="real")
    parser.add_argument("--run-root", default="runs/pong_offline_regu_sweep/logdir")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-prefix", default="twister_pong_offline_regu")
    parser.add_argument("--spatial-regu-weight", type=float, default=0.0)
    parser.add_argument("--disable-spatial-regu", action="store_true")
    parser.add_argument("--mask-preset", default="mask1", choices=sorted(MASK_PRESETS))
    parser.add_argument("--train-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--eval-dataset-path", default="", help="Optional held-out offline dataset for periodic eval.")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N train steps when --eval-dataset-path is set.")
    parser.add_argument("--eval-batches", type=int, default=8, help="Number of sampled held-out batches per eval.")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Held-out eval batch size. Defaults to --batch-size.")
    parser.add_argument("--torch-num-threads", type=int, default=int(os.environ.get("TWISTER_TORCH_NUM_THREADS", "4")))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--cache-dataset", type=str2bool, default=True)
    parser.add_argument(
        "--native-replay-mask-mode",
        choices=("none", "motion"),
        default="none",
        help="Mask source for TWISTER native ReplayBuffer shards. 'motion' derives mask1 from consecutive RGB frame differences.",
    )
    parser.add_argument("--override-config", default="{}", help="JSON dict applied to TWISTER config.")
    parser.add_argument("--wandb-enabled", type=str2bool, default=str2bool(os.environ.get("WANDB_ENABLED", "0")))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "twister"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_num_threads, 4)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or (
        f"{args.run_prefix}_model={args.model_size}_{args.mask_preset}_spatial_{weight_slug(args.spatial_regu_weight)}"
    )
    run_dir = Path(args.run_root) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "logs"))
    wandb_run = _init_wandb(args, run_name, run_dir)

    override = json.loads(args.override_config)
    override.update({
        "batch_size": int(args.batch_size),
        "L": int(args.batch_length),
        "model_size": args.model_size,
        "wm_initial_source": args.wm_initial_source,
        "num_envs": 1,
        "eval_episodes": 0,
        "spatial_regu_enabled": not bool(args.disable_spatial_regu),
        "loss_spatial_regu_scale": float(args.spatial_regu_weight),
        "spatial_regu_mask_weights": MASK_PRESETS[args.mask_preset],
        "load_replay_buffer_state_dict": False,
    })
    model = nnet.models.TWISTER(args.env_name, override_config=override).to(device)
    model.compile()
    dataset = StaticEpisodeDataset(
        args.dataset_path,
        args.batch_length,
        model.env.num_actions,
        cache_dataset=args.cache_dataset,
        native_replay_mask_mode=args.native_replay_mask_mode,
    )
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = StaticEpisodeDataset(
            args.eval_dataset_path,
            args.batch_length,
            model.env.num_actions,
            cache_dataset=args.cache_dataset,
            native_replay_mask_mode=args.native_replay_mask_mode,
        )
    print(
        f"[offline_twister] dataset_files={len(dataset.files)} "
        f"torch_threads={torch.get_num_threads()} interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )

    start_step = 0
    latest = ckpt_dir / "latest.ckpt"
    if args.resume and latest.exists():
        model.load(str(latest), load_optimizer=True, verbose=True, strict=True)
        start_step = int(model.model_step.item())

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and model.config.precision != torch.float32)
    acc_step = 0
    try:
        for step in range(start_step + 1, args.train_steps + 1):
            inputs = [x.to(device, non_blocking=True) for x in dataset.sample(args.batch_size)]
            inputs[0] = model.preprocess_inputs(inputs[0], time_stacked=True)
            inputs = tuple(inputs)
            losses, metrics, acc_step = model.world_model.train_step(
                inputs=inputs,
                targets=None,
                precision=model.config.precision,
                grad_scaler=scaler,
                accumulated_steps=1,
                acc_step=acc_step,
                eval_training=True,
            )
            if step == 1 or step % args.log_every == 0:
                print(f"[offline_twister] step={step}/{args.train_steps} loss={float(losses['loss'].detach().cpu()):.4f}", flush=True)
                scalars = _build_log_scalars(
                    losses,
                    metrics,
                    model.world_model.infos,
                    model.config,
                    prefix="train",
                )
                _write_scalars(writer, scalars, step)
                writer.flush()
                if wandb_run is not None and scalars:
                    wandb_run.log(scalars, step=step)
                model.world_model.reset_infos()
            if _should_eval(args, eval_dataset, step):
                eval_scalars = _evaluate_world_model(
                    model,
                    eval_dataset,
                    batch_size=args.eval_batch_size or args.batch_size,
                    batches=args.eval_batches,
                    device=device,
                    prefix="eval",
                )
                _write_scalars(writer, eval_scalars, step)
                writer.flush()
                if wandb_run is not None and eval_scalars:
                    wandb_run.log(eval_scalars, step=step)
            if step % args.save_every == 0:
                _save_world_model_checkpoint(model, ckpt_dir / f"checkpoints_{step}.ckpt")
                _save_world_model_checkpoint(model, latest)

        _save_world_model_checkpoint(model, latest)
    finally:
        writer.close()
        if wandb_run is not None:
            wandb_run.finish()


def _init_wandb(args, run_name: str, run_dir: Path):
    if not args.wandb_enabled:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        mode=args.wandb_mode,
        name=run_name,
        dir=str(run_dir / "wandb"),
        config=vars(args),
    )


def _as_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        value = value.detach().float().mean().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_log_scalars(
    losses: dict[str, Any],
    metrics: dict[str, Any],
    infos: dict[str, Any],
    config: Any,
    prefix: str,
) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in losses.items():
        scalar = _as_scalar(value)
        if scalar is not None:
            scalars[key] = scalar
    for key, value in metrics.items():
        scalar = _as_scalar(value)
        if scalar is not None:
            scalars[key] = scalar
    for key, value in infos.items():
        scalar = _as_scalar(value)
        if scalar is not None:
            scalars[key] = scalar
    _add_loss_breakdown(losses, metrics, config, scalars)
    if not prefix:
        return scalars
    return {f"{prefix}/{key}": value for key, value in scalars.items()}


def _write_scalars(writer: SummaryWriter, scalars: dict[str, float], step: int) -> None:
    for key, value in scalars.items():
        writer.add_scalar(key, value, step)


def _should_eval(args, eval_dataset: "StaticEpisodeDataset | None", step: int) -> bool:
    return eval_dataset is not None and args.eval_every > 0 and step % args.eval_every == 0 and args.eval_batches > 0


def _evaluate_world_model(
    model,
    dataset: "StaticEpisodeDataset",
    batch_size: int,
    batches: int,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _ in range(int(batches)):
        inputs = [x.to(device, non_blocking=True) for x in dataset.sample(batch_size)]
        inputs[0] = model.preprocess_inputs(inputs[0], time_stacked=True)
        losses, metrics, _, _ = model.world_model.eval_step(tuple(inputs), None)
        scalars = _build_log_scalars(losses, metrics, {}, model.config, prefix="")
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    model.world_model.reset_infos()
    if was_training:
        model.train()
    return {f"{prefix}/{key}": totals[key] / counts[key] for key in sorted(totals)}


def _add_loss_breakdown(losses: dict[str, Any], metrics: dict[str, Any], config: Any, scalars: dict[str, float]) -> None:
    total = _as_scalar(losses.get("loss"))
    if total is None:
        return
    denom = max(abs(total), 1e-8)
    spatial = _as_scalar(losses.get("loss_spatial_regu"))
    if spatial is not None:
        scalars["loss_scaled/spatial_regu_total"] = spatial
        scalars["lossfrac/spatial_regu"] = abs(spatial) / denom
    for idx in (1, 2, 3):
        key = f"loss_scaled/spatial_regu/mask{idx}"
        value = _as_scalar(metrics.get(key))
        if value is not None:
            scalars[f"lossfrac/spatial_regu/mask{idx}"] = abs(value) / denom
    ac_cpc_scaled = 0.0
    ac_cpc_raw = []
    weights = _contrastive_loss_weights(config)
    for idx, weight in enumerate(weights):
        value = _as_scalar(losses.get(f"loss_model_contrastive_{idx}"))
        if value is None:
            continue
        ac_cpc_raw.append(value)
        ac_cpc_scaled += value * weight
        scalars[f"loss_scaled/ac_cpc/{idx}"] = value * weight
        scalars[f"lossfrac/ac_cpc/{idx}"] = abs(value * weight) / denom
    if ac_cpc_raw:
        scalars["loss_unscaled/ac_cpc_mean"] = sum(ac_cpc_raw) / len(ac_cpc_raw)
        scalars["loss_scaled/ac_cpc"] = ac_cpc_scaled
        scalars["lossfrac/ac_cpc"] = abs(ac_cpc_scaled) / denom


def _contrastive_loss_weights(config: Any) -> list[float]:
    steps = int(getattr(config, "contrastive_steps", 0))
    if steps <= 0:
        return []
    scale = float(getattr(config, "loss_contrastive_scale", 0.0))
    decay = float(getattr(config, "contrastive_exp_lambda", 0.75))
    normalizer = sum(decay ** idx for idx in range(steps))
    if normalizer <= 0:
        return [0.0 for _ in range(steps)]
    return [scale * (decay ** idx) / normalizer for idx in range(steps)]


def _save_world_model_checkpoint(model, path: Path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {
            key: value.state_dict() for key, value in model.optimizer.items()
        },
        "model_step": model.model_step,
        "grad_scaler_state_dict": None,
        "replay_buffer_state_dict": {},
    }, path)


class StaticEpisodeDataset:
    def __init__(
        self,
        root: str | os.PathLike[str],
        length: int,
        num_actions: int,
        cache_dataset: bool = True,
        native_replay_mask_mode: str = "none",
    ):
        root = Path(root).expanduser()
        self.root = root
        self.files = [root] if root.is_file() else sorted(
            p for suffix in ("*.pt", "*.pth", "*.torch", "*.npz")
            for p in root.rglob(suffix)
            if p.name != "info.pt"
        )
        if not self.files:
            raise FileNotFoundError(f"No offline episode files found under {root}")
        self.length = int(length)
        self.num_actions = int(num_actions)
        self.native_replay_mask_mode = native_replay_mask_mode
        first_payload = self._load(self.files[0])
        self._replay_buffer_mode = self._is_replay_buffer_payload(first_payload)
        self._cached_data = None
        self._native_replay_buffer = None
        if self._replay_buffer_mode:
            self._native_replay_buffer = self._load_native_replay_buffer()
            return
        if cache_dataset:
            self._cached_data = [self._prepare_episode(first_payload)] + [
                self._prepare_episode(self._load(path)) for path in self.files[1:]
            ]

    def sample(self, batch_size: int):
        items = [self._sample_one() for _ in range(int(batch_size))]
        return [torch.stack(parts, dim=0) for parts in zip(*items)]

    def _sample_one(self):
        if self._replay_buffer_mode:
            return self._sample_replay_buffer_one()
        for _ in range(max(16, len(self.files))):
            if self._cached_data is None:
                data = self._prepare_episode(self._load(random.choice(self.files)))
            else:
                data = random.choice(self._cached_data)
            states = data["states"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            is_firsts = data["is_firsts"]
            masks = data["masks"]
            usable = min(states.shape[0], actions.shape[0], rewards.shape[0], dones.shape[0], is_firsts.shape[0], masks.shape[0])
            if usable >= self.length:
                start = random.randint(0, usable - self.length)
                stop = start + self.length
                model_steps = torch.arange(start, stop, dtype=torch.long)
                return (
                    states[start:stop],
                    actions[start:stop],
                    rewards[start:stop],
                    dones[start:stop],
                    is_firsts[start:stop],
                    model_steps,
                    masks[start:stop],
                )
        raise ValueError("No sampled episode is long enough for the requested batch length.")

    def _sample_replay_buffer_one(self):
        if self._native_replay_buffer is None:
            raise RuntimeError("Native ReplayBuffer mode was not initialized.")
        states, actions, rewards, dones, is_firsts, model_steps = self._native_replay_buffer.sample()
        masks = self._native_replay_masks(states)
        return states, actions, rewards, dones, is_firsts, model_steps, masks

    def _load_native_replay_buffer(self):
        keys = []
        for path in self.files:
            payload = self._load(path)
            if not self._is_replay_buffer_payload(payload):
                raise ValueError(f"Mixed ReplayBuffer and episode payloads are not supported: {path}")
            keys.extend(payload.keys())
        if not keys:
            raise ValueError(f"No ReplayBuffer trajectory keys found under {self.root}")

        if self.root.is_file():
            buffer_dir = self.root.parent
        elif self.root.name == "ReplayBuffer":
            buffer_dir = self.root
        elif (self.root / "ReplayBuffer").is_dir():
            buffer_dir = self.root / "ReplayBuffer"
        else:
            buffer_dir = self.root

        replay_buffer = nnet.datasets.ReplayBuffer(
            batch_size=1,
            root=str(buffer_dir.parent),
            buffer_capacity=max(1, len(keys)),
            epoch_length=1,
            sample_length=self.length,
            buffer_name=buffer_dir.name,
            save_trajectories=False,
        )
        replay_buffer.load(sorted(keys))
        replay_buffer.traj_index.fill_(max(keys) + 1)
        replay_buffer.num_steps.fill_(len(keys))
        return replay_buffer

    def _native_replay_masks(self, states: torch.Tensor) -> torch.Tensor:
        if self.native_replay_mask_mode == "none":
            return torch.zeros(states.shape[0], 3, *states.shape[-2:], dtype=torch.uint8)
        if self.native_replay_mask_mode != "motion":
            raise ValueError(f"Unsupported native replay mask mode: {self.native_replay_mask_mode}")

        frames = states[:, :3].float()
        diff = torch.zeros(states.shape[0], *states.shape[-2:], dtype=torch.float32)
        step_diff = (frames[1:] - frames[:-1]).abs().mean(dim=1)
        diff[1:] = torch.maximum(diff[1:], step_diff)
        diff[:-1] = torch.maximum(diff[:-1], step_diff)
        mask1 = (diff > 8.0).float()[:, None]
        mask1 = F.max_pool2d(mask1, kernel_size=3, stride=1, padding=1)
        masks = torch.zeros(states.shape[0], 3, *states.shape[-2:], dtype=torch.uint8)
        masks[:, 0] = (mask1[:, 0] > 0.5).to(torch.uint8)
        return masks

    def _prepare_episode(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        states = self._states(data)
        actions = self._actions(data, len(states))
        rewards = self._vector(data, ("reward", "rewards"), len(states), torch.float32)
        dones = self._vector(data, ("done", "dones", "is_terminal", "terminal"), len(states), torch.float32)
        is_firsts = self._vector(data, ("is_first", "is_firsts", "first"), len(states), torch.float32)
        if is_firsts.abs().sum() == 0:
            is_firsts[0] = 1.0
        masks = self._masks(data, len(states), states.shape[-2:])
        usable = min(states.shape[0], actions.shape[0], rewards.shape[0], dones.shape[0], is_firsts.shape[0], masks.shape[0])
        return {
            "states": states[:usable],
            "actions": actions[:usable],
            "rewards": rewards[:usable],
            "dones": dones[:usable],
            "is_firsts": is_firsts[:usable],
            "masks": masks[:usable],
        }

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if path.suffix == ".npz":
            data = np.load(path)
            return {key: torch.as_tensor(data[key]) for key in data.files}
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            return loaded
        raise ValueError(f"Unsupported episode payload in {path}: {type(loaded)}")

    @staticmethod
    def _is_replay_buffer_payload(data: dict[str, Any]) -> bool:
        if not data:
            return False
        first_key = next(iter(data.keys()))
        first_value = data[first_key]
        return isinstance(first_key, int) and isinstance(first_value, (list, tuple)) and len(first_value) >= 6

    def _states(self, data):
        value = self._first(data, ("state", "states", "obs", "image", "frames"))
        if value is None:
            raise KeyError("Episode has no state/states/obs/image/frames key.")
        value = torch.as_tensor(value)
        if value.ndim == 4 and value.shape[-1] in (1, 3):
            value = value.permute(0, 3, 1, 2)
        if value.dtype != torch.uint8:
            if value.numel() and value.min() < 0:
                value = (value.float().add(1).div(2).mul(255))
            elif value.numel() and value.max() <= 1.5:
                value = value.float().mul(255)
            value = value.clamp(0, 255).to(torch.uint8)
        return value[:, :3].contiguous()

    def _actions(self, data, length):
        value = self._first(data, ("action", "actions", "act"))
        if value is None:
            idx = torch.zeros(length, dtype=torch.long)
        else:
            value = torch.as_tensor(value)
            if value.ndim > 1 and value.shape[-1] == self.num_actions:
                return value.float()
            idx = value.reshape(value.shape[0], -1)[:, 0].long()
        return torch.nn.functional.one_hot(idx.clamp(0, self.num_actions - 1), self.num_actions).float()

    @staticmethod
    def _vector(data, names, length, dtype):
        value = StaticEpisodeDataset._first(data, names)
        if value is None:
            return torch.zeros(length, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype).reshape(-1)

    @staticmethod
    def _masks(data, length, spatial_shape):
        masks = StaticEpisodeDataset._first(data, ("masks", "mask"))
        parts = []
        if masks is not None:
            masks = torch.as_tensor(masks)
            if masks.ndim == 3:
                masks = masks[:, None]
            elif masks.ndim == 4 and masks.shape[-1] <= 3:
                masks = masks.permute(0, 3, 1, 2)
            parts.append(masks)
        else:
            for key in ("mask1", "mask2", "mask3"):
                value = StaticEpisodeDataset._first(data, (key,))
                if value is not None:
                    mask = torch.as_tensor(value)
                    if mask.ndim == 3:
                        mask = mask[:, None]
                    parts.append(mask)
        if not parts:
            return torch.zeros(length, 3, *spatial_shape, dtype=torch.uint8)
        masks = torch.cat(parts, dim=1)
        if masks.shape[1] < 3:
            pad = torch.zeros(masks.shape[0], 3 - masks.shape[1], *masks.shape[-2:], dtype=masks.dtype)
            masks = torch.cat([masks, pad], dim=1)
        return (masks[:, :3] > 0.5).to(torch.uint8).contiguous()

    @staticmethod
    def _first(data, names):
        for name in names:
            if name in data:
                return data[name]
        return None


if __name__ == "__main__":
    main()
