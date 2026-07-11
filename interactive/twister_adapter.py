from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
from PIL import Image
import pygame
import torch
from torch.nn import functional as F

import nnet
from nnet import modules

try:
    from wm_play.api import PixelPolicy, PlaySession, PolicyAction, StepResult
    from wm_play.server_summary import print_runtime_event
    from wm_play.status import play_status_lines
except ImportError:
    @dataclass
    class StepResult:
        obs: Any
        reward: float
        done: bool
        trunc: bool
        info: dict[str, Any]

    class PlaySession:
        pass

    class PixelPolicy:
        pass

    @dataclass
    class PolicyAction:
        action: Any
        info: dict[str, Any] | None = None

    def print_runtime_event(label, value):
        del label, value

    def play_status_lines(status, extras=()):
        return [str(status), *[str(x) for x in extras]]


ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / "third_party" / "rl-in-pixel-env" / "src"
if COMMON_SRC.is_dir():
    sys.path.insert(0, str(COMMON_SRC))

from diamond_rl_env.actor_critic import ActorCriticConfig
from diamond_rl_env.policy import ActorCriticPolicy, load_actor_critic_policy
from interactive.sb3_atari_policy import is_sb3_atari_policy_checkpoint, load_sb3_pixel_policy


ATARI_ACTION_NAMES = [
    "noop", "fire", "up", "right", "left", "down",
    "upright", "upleft", "downright", "downleft",
    "upfire", "rightfire", "leftfire", "downfire",
    "uprightfire", "upleftfire", "downrightfire", "downleftfire",
]


def _keymap(action_names):
    wanted = {
        (pygame.K_SPACE,): "fire",
        (pygame.K_d,): "right",
        (pygame.K_a,): "left",
        (pygame.K_d, pygame.K_SPACE): "rightfire",
        (pygame.K_a, pygame.K_SPACE): "leftfire",
        (pygame.K_w,): "up",
        (pygame.K_s,): "down",
    }
    return {keys: action_names.index(name) for keys, name in wanted.items() if name in action_names}


def _env_id_to_twister_name(env_name: str) -> str:
    if env_name.startswith("atari100k-"):
        return env_name
    game = env_name
    for suffix in ("NoFrameskip-v4", "NoFrameskip-v0", "-v4", "-v0"):
        game = game.replace(suffix, "")
    return f"atari100k-{game.lower()}"


def _validate_initial_source(value: str) -> str:
    source = str(value).lower()
    if source not in {"real", "prior", "dataset"}:
        raise ValueError(f"Unknown wm initial source {value!r}; expected 'real', 'prior', or 'dataset'.")
    return source


def _policy_checkpoint_sort_key(path: Path) -> tuple[int, int, float, str]:
    match = re.search(r"(?:update|epoch)_?0*(\d+)", path.stem)
    if match:
        return (1, int(match.group(1)), path.stat().st_mtime, path.name)
    return (0, -1, path.stat().st_mtime, path.name)


def resolve_policy_checkpoint_path(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.is_file():
        return root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Policy checkpoint path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Policy checkpoint path is neither file nor directory: {root}")

    for rel in (
        "latest.pt",
        "pixel_rl_ckpt/latest.pt",
        "policy_ckpt/latest.pt",
        "ckpt/latest.pt",
        "checkpoints/latest.pt",
    ):
        candidate = root / rel
        if candidate.is_file():
            return candidate.resolve()

    candidates: list[Path] = []
    for child in ("pixel_rl_ckpt", "policy_ckpt", "ckpt", "checkpoints"):
        candidate_dir = root / child
        if candidate_dir.is_dir():
            candidates.extend(p for p in candidate_dir.glob("*.pt") if p.is_file())
    if not candidates:
        candidates = [p for p in root.glob("*.pt") if p.is_file()]
    if not candidates:
        candidates = [p for p in root.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No .pt policy checkpoint found under: {root}")
    return max(candidates, key=_policy_checkpoint_sort_key).resolve()


class _BootstrapFrameDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if self.root.is_file():
            self.files = [self.root]
        else:
            self.files = sorted(path for path in self.root.rglob("*.pt") if path.name != "info.pt")
        if not self.files:
            raise ValueError(f"No bootstrap episodes found under {self.root}")

    @staticmethod
    def _first(data: dict[str, Any], names: tuple[str, ...]):
        for name in names:
            if name in data:
                return data[name]
        return None

    def sample(self) -> torch.Tensor:
        for _ in range(max(16, len(self.files))):
            data = torch.load(random.choice(self.files), map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                continue
            obs = self._first(data, ("obs", "image", "frames", "observation"))
            if obs is None and isinstance(data.get("observations"), dict):
                obs = self._first(data["observations"], ("image", "obs", "frames"))
            if obs is None:
                continue
            obs = torch.as_tensor(obs)
            if obs.ndim != 4 or obs.shape[0] == 0:
                continue
            frame = obs[int(torch.randint(obs.shape[0], ()).item())]
            if frame.shape[-1] == 3 and frame.shape[0] != 3:
                frame = frame.permute(2, 0, 1)
            if frame.dtype != torch.uint8:
                frame = frame.float()
                if frame.numel() and frame.min() < 0:
                    frame = (frame + 1.0) / 2.0
                if frame.numel() and frame.max() <= 1.5:
                    frame = frame * 255.0
                frame = frame.clamp(0, 255).to(torch.uint8)
            return frame[:3].contiguous()
        raise RuntimeError(f"Could not sample a bootstrap frame from {self.root}")


def _make_model(env_name: str, device: torch.device, checkpoint: str | None = None):
    model = nnet.models.TWISTER(
        _env_id_to_twister_name(env_name),
        override_config={
            "num_envs": 1,
            "eval_episodes": 0,
            "load_replay_buffer_state_dict": False,
        },
    )
    model.compile()
    model.to(device)
    if checkpoint:
        model.load(str(Path(checkpoint).expanduser().resolve()), load_optimizer=False, verbose=True, strict=True)
    model.eval()
    return model


def _make_env(env_name: str, seed: int):
    model = nnet.models.TWISTER(
        _env_id_to_twister_name(env_name),
        override_config={"num_envs": 1, "eval_episodes": 0},
    )
    params = dict(model.config.env_params)
    params["seed"] = int(seed)
    env = model.config.env_class(**params)
    del model
    return env


def _action_names(env) -> list[str]:
    for obj in (getattr(env, "env", None), getattr(getattr(env, "env", None), "env", None)):
        meanings = getattr(obj, "get_action_meanings", None)
        if callable(meanings):
            return [str(x).lower().replace("no-op", "noop") for x in meanings()]
    return ATARI_ACTION_NAMES[: int(env.num_actions)]


def _state_to_frame(state: torch.Tensor) -> np.ndarray:
    state = state.detach().cpu()
    if state.ndim == 4:
        state = state[0]
    return state[:3].permute(1, 2, 0).contiguous().numpy().astype(np.uint8, copy=False)


def _frame_to_state_tensor(frame: Any, device: torch.device) -> torch.Tensor:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    return torch.as_tensor(array, dtype=torch.uint8, device=device).permute(2, 0, 1).unsqueeze(0).contiguous()


def _decode_frame(model, state) -> np.ndarray:
    dist = model.decoder_network(state["stoch"].flatten(-2, -1))
    tensor = dist.mode().detach().float().clamp(-0.5, 0.5).add(0.5).mul(255).to(torch.uint8)
    if tensor.ndim == 5:
        tensor = tensor[:, 0]
    return tensor[0].permute(1, 2, 0).contiguous().cpu().numpy()


@dataclass
class WMSlot:
    name: str
    checkpoint: Path
    model: Any


@dataclass
class PolicySlot:
    name: str
    checkpoint: Path
    model: Any = None
    pixel_policy: PixelPolicy | None = None
    state: Any = None
    action: torch.Tensor | None = None


@dataclass
class PixelRLPolicy(PixelPolicy):
    name: str
    policy: ActorCriticPolicy

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: Any) -> PolicyAction:
        image = _obs_to_pixel_rl_tensor(obs, self.policy.device)
        action, value, _ = self.policy.act(image)
        return PolicyAction(
            action=int(action.reshape(-1)[0].detach().cpu().item()),
            info={
                "source": "rl_in_pixel_env_policy",
                "policy_slot_name": self.name,
                "value": float(value.reshape(-1)[0].detach().cpu().item()),
            },
        )


def _obs_to_pixel_rl_tensor(obs: Any, device: torch.device) -> torch.Tensor:
    frame = np.asarray(obs)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    tensor = torch.as_tensor(frame, device=device)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image obs with 3 dims, got {tuple(tensor.shape)}.")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous().float()
    if tensor.shape[-2:] != (64, 64):
        tensor = F.interpolate(tensor, size=(64, 64), mode="nearest")
    if tensor.numel() and tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0, 1).mul(2).sub(1)


def load_pixel_rl_policy(
    checkpoint: str | Path,
    *,
    name: str,
    device: torch.device,
    num_actions: int,
    deterministic: bool = True,
) -> PixelRLPolicy:
    checkpoint = resolve_policy_checkpoint_path(checkpoint)
    cfg = ActorCriticConfig(num_actions=int(num_actions))
    policy = load_actor_critic_policy(
        checkpoint,
        cfg=cfg,
        device=device,
        deterministic=deterministic,
        module_name="policy",
    )
    return PixelRLPolicy(name=name, policy=policy)


class TwisterRealEnv:
    name = "real"

    def __init__(self, env):
        self.env = env
        self.return_ = 0.0
        self.step_count = 0

    def reset(self):
        obs = self.env.reset()
        self.return_ = 0.0
        self.step_count = 0
        frame = _state_to_frame(obs.state)
        return frame, {"reward": 0.0, "done": False, "trunc": False}

    def step(self, action: int):
        obs = self.env.step(torch.tensor(int(action), dtype=torch.long))
        reward = float(obs.reward.item())
        done = bool(obs.done.item())
        self.return_ += reward
        self.step_count += 1
        frame = _state_to_frame(obs.state)
        return StepResult(frame, reward, done, False, {"backend": "real"})

    def close(self):
        close = getattr(getattr(self.env, "env", None), "close", None)
        if callable(close):
            close()


class TwisterWMEnv:
    def __init__(
        self,
        slot: WMSlot,
        seed: int,
        env_name: str,
        horizon: int,
        respect_terminal: bool,
        initial_source: str = "real",
        bootstrap_dataset: str | None = None,
    ):
        self.slot = slot
        self.model = slot.model
        self.initial_source = _validate_initial_source(initial_source)
        self.real_env = _make_env(env_name, seed) if self.initial_source == "real" else None
        self.bootstrap_dataset = None
        if self.initial_source == "dataset":
            if not bootstrap_dataset:
                raise ValueError("TWISTER dataset initial source requires --wm-bootstrap-dataset.")
            self.bootstrap_dataset = _BootstrapFrameDataset(bootstrap_dataset)
        self.horizon = int(horizon)
        self.respect_terminal = bool(respect_terminal)
        self.state = None
        self.steps = 0
        self.return_ = 0.0

    @torch.no_grad()
    def reset(self):
        if self.initial_source == "prior":
            return self._reset_from_prior()
        if self.initial_source == "dataset":
            return self._reset_from_dataset()
        return self._reset_from_real()

    def _reset_from_prior(self):
        device = self.model.device
        self.state = self.model.rssm.initial(1, 1, torch.float32, device, detach_learned=True)
        if self.state["hidden"] is not None:
            self.state["hidden"] = self.model.rssm.slice_hidden(self.state["hidden"])
        self.steps = 0
        self.return_ = 0.0
        return _decode_frame(self.model, self.state), {"backend": "wm", "wm_initial_source": "prior"}

    def _reset_from_real(self):
        obs = self.real_env.reset()
        return self._reset_from_state(obs.state.unsqueeze(0), "real")

    def _reset_from_dataset(self):
        assert self.bootstrap_dataset is not None
        return self._reset_from_state(self.bootstrap_dataset.sample().unsqueeze(0), "dataset")

    def bootstrap_from_observation(self, obs: Any):
        return self._reset_from_state(_frame_to_state_tensor(obs, self.model.device), "observation")

    def _reset_from_state(self, raw_state: torch.Tensor, initial_source: str):
        device = self.model.device
        raw = raw_state.to(device)
        states = self.model.preprocess_inputs(raw, time_stacked=False)
        latent = self.model.encoder_network(states)
        latent = {key: value.unsqueeze(1) for key, value in latent.items()}
        prev_state = self.model.rssm.initial(1, 1, states.dtype, device, detach_learned=True)
        prev_action = torch.zeros(1, 1, self.model.env.num_actions, dtype=states.dtype, device=device)
        posts, _ = self.model.rssm(
            states=latent,
            prev_states=prev_state,
            prev_actions=prev_action,
            is_firsts=torch.zeros(1, 1, dtype=states.dtype, device=device),
        )
        posts["hidden"] = self.model.rssm.slice_hidden(posts["hidden"])
        self.state = posts
        self.steps = 0
        self.return_ = 0.0
        return _decode_frame(self.model, self.state), {"backend": "wm", "wm_initial_source": initial_source}

    @torch.no_grad()
    def step(self, action: int):
        assert self.state is not None
        device = self.model.device
        action_onehot = torch.nn.functional.one_hot(
            torch.tensor([int(action)], dtype=torch.long, device=device),
            self.model.env.num_actions,
        ).float()
        mask = modules.return_mask(
            seq_len=1,
            hidden_len=self.model.rssm.get_hidden_len(self.state["hidden"]),
            left_context=self.model.rssm.att_context_left,
            right_context=0,
            dtype=action_onehot.dtype,
            device=device,
        )
        next_state = self.model.rssm.forward_img(self.state, action_onehot.unsqueeze(1), mask)
        next_state["hidden"] = self.model.rssm.slice_hidden(next_state["hidden"])
        feat = self.model.rssm.get_feat(next_state)
        raw_reward = float(self.model.reward_network(feat).mode().reshape(-1)[0].detach().cpu())
        cont = float(self.model.continue_network(feat).base_dist.probs.reshape(-1)[0].detach().cpu())
        done = bool(cont <= 0.5 and self.respect_terminal)
        self.steps += 1
        trunc = self.steps >= self.horizon
        self.return_ += raw_reward
        self.state = next_state
        frame = _decode_frame(self.model, self.state)
        return StepResult(
            frame,
            raw_reward,
            done,
            trunc,
            {"backend": "wm", "raw_reward": raw_reward, "continuation": cont, "wm_name": self.slot.name},
        )

    def close(self):
        if self.real_env is None:
            return
        close = getattr(getattr(self.real_env, "env", None), "close", None)
        if callable(close):
            close()


class TwisterPlaySession(PlaySession):
    def __init__(
        self,
        env_name: str,
        seed: int,
        real_env,
        wm_slots,
        policy_slots,
        horizon: int,
        respect_terminal: bool,
        wm_initial_source: str,
        wm_bootstrap_dataset: str | None,
    ):
        self.env_name = env_name
        self.real = TwisterRealEnv(real_env)
        self.wm_slots = list(wm_slots)
        self.policy_slots = list(policy_slots)
        self.wm_initial_source = wm_initial_source
        self.wm_bootstrap_dataset = wm_bootstrap_dataset
        self.wm_envs = [
            TwisterWMEnv(
                slot,
                seed + 1000 + index,
                env_name,
                horizon,
                respect_terminal,
                wm_initial_source,
                wm_bootstrap_dataset,
            )
            for index, slot in enumerate(self.wm_slots)
        ]
        self.current_backend_index = 0
        self.policy_slot_index = 0
        self.action_names = _action_names(real_env)
        self.keymap = _keymap(self.action_names)
        self.controller = "human"
        self.current_obs = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_trunc = False
        self.last_info = {}
        self.last_policy_info = {}

    @property
    def active(self):
        return self.real if self.current_backend_index == 0 else self.wm_envs[self.current_backend_index - 1]

    @property
    def horizon(self):
        return None if self.current_backend_index == 0 else int(self.active.horizon)

    def reset(self):
        self.current_obs, self.last_info = self.active.reset()
        self._reset_policy_slots()
        self.last_reward = 0.0
        self.last_done = False
        self.last_trunc = False
        self.last_policy_info = {}

    def switch_backend(self, direction: int):
        count = 1 + len(self.wm_envs)
        self.current_backend_index = (self.current_backend_index + int(direction)) % count
        print_runtime_event("backend", "real" if self.current_backend_index == 0 else self.wm_slots[self.current_backend_index - 1].name)
        self.reset()

    def switch_controller(self):
        if self.controller == "human" and not self.policy_slots:
            print_runtime_event("controller", self.controller)
            return
        controllers = ["human"] + [slot.name for slot in self.policy_slots]
        index = controllers.index(self.controller) if self.controller in controllers else 0
        index = (index + 1) % len(controllers)
        self.controller = controllers[index]
        self.policy_slot_index = max(0, index - 1)
        print_runtime_event("controller", self.controller)

    def switch_policy(self, direction: int):
        if not self.policy_slots:
            print_runtime_event("policy", "none")
            return
        self.policy_slot_index = (self.policy_slot_index + int(direction)) % len(self.policy_slots)
        if self.controller != "human":
            self.controller = self.active_policy_slot.name
        print_runtime_event(
            "policy",
            f"{self.policy_slot_index + 1}/{len(self.policy_slots)} ({self.active_policy_slot.name})",
        )

    def adjust_horizon(self, delta: int):
        current = self.horizon
        if current is not None:
            self.set_horizon(current + int(delta))

    def set_horizon(self, horizon: int):
        if self.current_backend_index == 0:
            return
        self.active.horizon = max(1, int(horizon))
        print_runtime_event("wm horizon", self.active.horizon)
        self.reset()

    def choose_action(self, human_action: int):
        if not self.policy_slots or self.current_obs is None:
            self.last_policy_info = {"source": "human"}
            return int(human_action)

        action = int(human_action)
        observed = {}
        if self.controller != "human":
            slot = self.active_policy_slot
            if slot.pixel_policy is not None:
                result = slot.pixel_policy.act(self.current_obs)
                action = int(result.action)
                info = dict(result.info or {})
            else:
                latent = self._observe_policy_slot(slot)
                action, info = self._policy_action(slot, latent)
                observed[id(slot)] = latent
            self.last_policy_info = info
        else:
            self.last_policy_info = {"source": "human"}

        for slot in self.policy_slots:
            if slot.pixel_policy is not None:
                continue
            latent = observed.get(id(slot))
            if latent is None:
                latent = self._observe_policy_slot(slot)
            slot.state = latent
            slot.action = self._action_onehot(slot.model, action)
        return action

    def step(self, action: int):
        result = self.active.step(int(action))
        info = dict(result.info or {})
        info.update(self.last_policy_info or {})
        result = StepResult(result.obs, result.reward, result.done, result.trunc, info)
        self.current_obs = result.obs
        self.last_reward = float(result.reward)
        self.last_done = bool(result.done)
        self.last_trunc = bool(result.trunc)
        self.last_info = info
        if result.done or result.trunc:
            self.reset()
        return result

    def header(self, action: int, info: dict[str, Any]):
        backend = "real" if self.current_backend_index == 0 else self.wm_slots[self.current_backend_index - 1].name
        status = {
            "env_name": self.env_name,
            "env_kind": backend,
            "control": self.controller,
            "step": getattr(self.active, "steps", getattr(self.active, "step_count", 0)),
            "reward": self.last_reward,
            "return": getattr(self.active, "return_", 0.0),
            "action_name": self.action_names[int(action)] if int(action) < len(self.action_names) else str(action),
            "done": self.last_done,
            "trunc": self.last_trunc,
        }
        if "continuation" in info:
            status["continuation"] = info["continuation"]
        extras = []
        if self.policy_slots:
            index = min(max(self.policy_slot_index, 0), len(self.policy_slots) - 1)
            name = info.get("policy_slot_name", self.policy_slots[index].name)
            extras.append(("Policy", f"{index + 1}/{len(self.policy_slots)} ({name})"))
        if "policy_entropy" in info:
            extras.append(("Policy ent", f"{float(info['policy_entropy']):.3f}"))
        return play_status_lines(status, extras)

    def render_frame(self, size: int, header_lines: list[str]):
        del header_lines
        frame = self.current_obs
        if frame is None:
            frame, _ = self.active.reset()
        return Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").resize((size, size), Image.NEAREST)

    def record_metadata(self) -> dict[str, Any]:
        backend = "real" if self.current_backend_index == 0 else self.wm_slots[self.current_backend_index - 1].name
        policy = self.active_policy_slot.name if self.policy_slots else ""
        return {
            "project": "TWISTER",
            "env_name": self.env_name,
            "backend": backend,
            "backend_index": int(self.current_backend_index),
            "controller": self.controller,
            "policy_index": int(self.policy_slot_index),
            "policy_label": policy,
            "wm_initial_source": self.wm_initial_source,
        }

    def close(self):
        self.real.close()
        for env in self.wm_envs:
            env.close()

    @property
    def active_policy_slot(self) -> PolicySlot:
        return self.policy_slots[self.policy_slot_index]

    def _reset_policy_slots(self):
        for slot in self.policy_slots:
            if slot.pixel_policy is not None:
                slot.pixel_policy.reset()
                continue
            device = slot.model.device
            slot.state = slot.model.rssm.initial(
                1, 1, torch.float32, device, detach_learned=True
            )
            slot.action = torch.zeros(
                1, slot.model.env.num_actions, dtype=torch.float32, device=device
            )

    @torch.no_grad()
    def _observe_policy_slot(self, slot: PolicySlot):
        model = slot.model
        device = model.device
        if slot.state is None or slot.action is None:
            self._reset_policy_slots()
        raw = _frame_to_state_tensor(self.current_obs, device)
        states = model.preprocess_inputs(raw, time_stacked=False)
        latent = model.encoder_network(states)
        latent = {key: value.unsqueeze(1) for key, value in latent.items()}
        posts, _ = model.rssm(
            states=latent,
            prev_states=slot.state,
            prev_actions=slot.action.unsqueeze(1),
            is_firsts=torch.zeros(1, 1, dtype=slot.action.dtype, device=device),
        )
        posts["hidden"] = model.rssm.slice_hidden(posts["hidden"])
        return posts

    @torch.no_grad()
    def _policy_action(self, slot: PolicySlot, latent) -> tuple[int, dict[str, Any]]:
        feat = slot.model.rssm.get_feat(latent).squeeze(1)
        dist = slot.model.policy_network(feat)
        action_onehot = dist.mode()
        action = int(action_onehot.argmax(dim=-1).reshape(-1)[0].item())
        entropy = dist.entropy().reshape(-1)[0].detach().cpu()
        return action, {
            "source": "twister_policy",
            "policy_slot_name": slot.name,
            "policy_entropy": float(entropy),
        }

    def _action_onehot(self, model, action: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.tensor([int(action)], dtype=torch.long, device=model.device),
            model.env.num_actions,
        ).float()


def build_twister_session(
    *,
    env_name: str,
    seed: int,
    checkpoint_args: list[str],
    wm_name_args: list[str],
    policy_checkpoint_args: list[str] | None = None,
    policy_name_args: list[str] | None = None,
    additional_policy_controller: bool = False,
    device: str = "cuda",
    wm_horizon: int = 512,
    wm_respect_terminal: bool = True,
    wm_initial_source: str = "real",
    wm_bootstrap_dataset: str | None = None,
) -> TwisterPlaySession:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    real_env = _make_env(env_name, seed)
    slots = []
    models_by_path = {}
    slots_by_path = {}
    for index, checkpoint in enumerate(checkpoint_args):
        path = Path(checkpoint).expanduser().resolve()
        name = wm_name_args[index] if index < len(wm_name_args) else path.parent.name
        model = _make_model(env_name, device_obj, str(path))
        slot = WMSlot(name=name, checkpoint=path, model=model)
        slots.append(slot)
        models_by_path[path] = model
        slots_by_path[path] = slot

    policy_slots = []
    if additional_policy_controller:
        policy_checkpoint_args = list(policy_checkpoint_args or [])
        policy_name_args = list(policy_name_args or [])
        for index, checkpoint in enumerate(policy_checkpoint_args):
            path = resolve_policy_checkpoint_path(checkpoint)
            if index < len(policy_name_args):
                name = policy_name_args[index]
            elif path in slots_by_path:
                name = slots_by_path[path].name
            else:
                name = path.parent.name
            model = models_by_path.get(path)
            if model is None:
                if is_sb3_atari_policy_checkpoint(path):
                    policy = load_sb3_pixel_policy(
                        path,
                        name=name,
                        device=device_obj,
                    )
                    policy_slots.append(PolicySlot(name=name, checkpoint=path, pixel_policy=policy))
                    continue
                try:
                    model = _make_model(env_name, device_obj, str(path))
                    models_by_path[path] = model
                    policy_slots.append(PolicySlot(name=name, checkpoint=path, model=model))
                except Exception:
                    policy = load_pixel_rl_policy(
                        path,
                        name=name,
                        device=device_obj,
                        num_actions=int(real_env.num_actions),
                    )
                    policy_slots.append(PolicySlot(name=name, checkpoint=path, pixel_policy=policy))
            else:
                policy_slots.append(PolicySlot(name=name, checkpoint=path, model=model))

    return TwisterPlaySession(
        env_name,
        seed,
        real_env,
        slots,
        policy_slots,
        wm_horizon,
        wm_respect_terminal,
        wm_initial_source,
        wm_bootstrap_dataset,
    )
