from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

import nnet
from nnet import modules

from .rewards import quantize_pong_reward


def _action_to_int(action: Any) -> int:
    try:
        arr = np.asarray(action).reshape(-1)
        if arr.size:
            return int(arr[0])
    except Exception:
        pass
    return int(action)


def _env_id_to_twister_name(env_name: str) -> str:
    name = str(env_name)
    if name.startswith("atari100k-"):
        return name
    game = name
    for suffix in ("NoFrameskip-v4", "NoFrameskip-v0", "-v4", "-v0"):
        game = game.replace(suffix, "")
    return f"atari100k-{game.lower()}"


def _validate_initial_source(value: str) -> str:
    source = str(value).lower()
    if source not in {"real", "prior", "dataset"}:
        raise ValueError(f"Unknown wm initial source {value!r}; expected 'real', 'prior', or 'dataset'.")
    return source


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


class TwisterAtariPixelEnv:
    def __init__(self, env, num_actions: int):
        self.env = env
        self.num_actions = int(num_actions)

    def reset(self):
        obs = self.env.reset()
        return self._obs(obs.state, reward=0.0, first=True, last=False, terminal=False), {}

    def step(self, action: int):
        obs = self.env.step(torch.tensor(_action_to_int(action), dtype=torch.long))
        done = bool(obs.done.item())
        return (
            self._obs(obs.state, reward=float(obs.reward.item()), first=False, last=done, terminal=done),
            float(obs.reward.item()),
            done,
            False,
            {},
        )

    def close(self):
        close = getattr(getattr(self.env, "env", None), "close", None)
        if callable(close):
            close()

    @staticmethod
    def _obs(state, reward, first, last, terminal):
        frame = _chw_uint8_to_hwc(state)
        return {
            "image": frame,
            "reward": np.float32(reward),
            "is_first": bool(first),
            "is_last": bool(last),
            "is_terminal": bool(terminal),
        }


class VectorPixelEnv:
    def __init__(self, envs, device, raw_reward_threshold=0.5):
        self.envs = list(envs)
        self.device = torch.device(device)
        self.raw_reward_threshold = float(raw_reward_threshold)
        self.num_envs = len(self.envs)
        self.num_actions = self.envs[0].num_actions
        self._completed = []
        self._raw_rewards = []
        self._episode_scores = [0.0 for _ in self.envs]
        self._episode_agent_scores = [0.0 for _ in self.envs]
        self._episode_opponent_scores = [0.0 for _ in self.envs]
        self._episode_raw_scores = [0.0 for _ in self.envs]
        self._episode_raw_agent_scores = [0.0 for _ in self.envs]
        self._episode_raw_opponent_scores = [0.0 for _ in self.envs]
        self._episode_has_raw_reward = [False for _ in self.envs]
        self._episode_lengths = [0 for _ in self.envs]

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)
        return self._obs_to_tensor(obs), {"raw": infos}

    def step(self, actions):
        actions = actions.detach().cpu().numpy().astype(np.int32)
        next_obs, rewards, ends, truncs, final_obs, raw_infos = [], [], [], [], [], []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            ob, rew, end, trunc, info = env.step(int(action))
            rewards.append(rew)
            ends.append(end)
            truncs.append(trunc)
            raw_infos.append(info)
            self._record_transition(index, rew, info)
            if end or trunc:
                self._complete_episode(index)
                final_obs.append(ob)
                ob, _ = env.reset()
            next_obs.append(ob)
        info = {"raw": raw_infos}
        if final_obs:
            info["final_observation"] = self._obs_to_tensor(final_obs)
        return (
            self._obs_to_tensor(next_obs),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(ends, dtype=torch.float32, device=self.device),
            torch.as_tensor(truncs, dtype=torch.float32, device=self.device),
            info,
        )

    def pop_episode_stats(self):
        completed = self._completed
        self._completed = []
        return completed

    def pop_episode_videos(self, max_videos=1):
        return []

    def pop_raw_reward_stats(self):
        if not self._raw_rewards:
            return {}
        values = np.asarray(self._raw_rewards, dtype=np.float32)
        self._raw_rewards = []
        return {
            "raw_reward_mean": float(values.mean()),
            "raw_reward_abs_mean": float(np.abs(values).mean()),
            "raw_reward_nonzero_rate": float((np.abs(values) >= self.raw_reward_threshold).mean()),
            "raw_agent_reward_mean": float(np.maximum(values, 0.0).mean()),
            "raw_opponent_reward_mean": float(np.maximum(-values, 0.0).mean()),
            "raw_agent_reward_nonzero_rate": float((values >= self.raw_reward_threshold).mean()),
            "raw_opponent_reward_nonzero_rate": float((values <= -self.raw_reward_threshold).mean()),
        }

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _record_transition(self, index, rew, info):
        rew = float(rew)
        self._episode_scores[index] += rew
        self._episode_agent_scores[index] += max(rew, 0.0)
        self._episode_opponent_scores[index] += max(-rew, 0.0)
        self._episode_lengths[index] += 1
        if isinstance(info, dict) and "raw_reward" in info:
            raw_reward = float(info["raw_reward"])
            self._episode_raw_scores[index] += raw_reward
            self._episode_raw_agent_scores[index] += max(raw_reward, 0.0)
            self._episode_raw_opponent_scores[index] += max(-raw_reward, 0.0)
            self._episode_has_raw_reward[index] = True
            self._raw_rewards.append(raw_reward)

    def _complete_episode(self, index):
        completed = {
            "score": self._episode_scores[index],
            "agent_score": self._episode_agent_scores[index],
            "opponent_score": self._episode_opponent_scores[index],
            "length": self._episode_lengths[index],
        }
        if self._episode_has_raw_reward[index]:
            completed["raw_score"] = self._episode_raw_scores[index]
            completed["raw_agent_score"] = self._episode_raw_agent_scores[index]
            completed["raw_opponent_score"] = self._episode_raw_opponent_scores[index]
        self._completed.append(completed)
        self._episode_scores[index] = 0.0
        self._episode_agent_scores[index] = 0.0
        self._episode_opponent_scores[index] = 0.0
        self._episode_raw_scores[index] = 0.0
        self._episode_raw_agent_scores[index] = 0.0
        self._episode_raw_opponent_scores[index] = 0.0
        self._episode_has_raw_reward[index] = False
        self._episode_lengths[index] = 0

    def _obs_to_tensor(self, obs):
        images = [np.asarray(x["image"]) for x in obs]
        arr = np.stack(images)
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        return tensor.div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()


class BatchedTwisterWMPixelEnv:
    def __init__(
        self,
        *,
        env_name,
        seed,
        checkpoint,
        horizon,
        num_envs,
        device,
        reward_threshold,
        respect_terminal,
        initial_source="real",
        bootstrap_dataset=None,
    ):
        if not checkpoint:
            raise ValueError("pixel RL backend=wm requires --wm-checkpoint.")
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.threshold = float(reward_threshold)
        self.respect_terminal = bool(respect_terminal)
        self.initial_source = _validate_initial_source(initial_source)
        self.twister_env_name = _env_id_to_twister_name(env_name)
        self.seed = int(seed)
        self.model = _load_twister_model(self.twister_env_name, checkpoint, self.device)
        self.model.eval()
        self.num_actions = int(self.model.env.num_actions)
        self.real_envs = []
        if self.initial_source == "real":
            self.real_envs = [_make_twister_env(self.twister_env_name, self.seed + i) for i in range(self.num_envs)]
        self.bootstrap_dataset = None
        if self.initial_source == "dataset":
            if not bootstrap_dataset:
                raise ValueError("TWISTER dataset initial source requires --wm-bootstrap-dataset.")
            self.bootstrap_dataset = _BootstrapFrameDataset(bootstrap_dataset)
        self._step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_state = None
        self.prev_action = None
        self.current_frames = None
        self._completed = []
        self._raw_rewards = []
        self._episode_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_agent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_opponent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_agent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_opponent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_has_raw_reward = np.zeros(self.num_envs, dtype=bool)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    @torch.no_grad()
    def reset(self):
        self._reset_indices(torch.arange(self.num_envs, device=self.device))
        return self._frames_to_obs_tensor(self.current_frames), {}

    @torch.no_grad()
    def step(self, actions):
        actions = actions.to(device=self.device, dtype=torch.long).reshape(-1)
        action_onehot = torch.nn.functional.one_hot(actions, num_classes=self.num_actions).to(torch.float32)
        mask = modules.return_mask(
            seq_len=1,
            hidden_len=self.model.rssm.get_hidden_len(self.prev_state["hidden"]),
            left_context=self.model.rssm.att_context_left,
            right_context=0,
            dtype=action_onehot.dtype,
            device=self.device,
        )
        next_state = self.model.rssm.forward_img(
            prev_states=self.prev_state,
            prev_actions=action_onehot.unsqueeze(1),
            mask=mask,
        )
        next_state["hidden"] = self.model.rssm.slice_hidden(next_state["hidden"])
        feats = self.model.rssm.get_feat(next_state)
        raw_rewards = self.model.reward_network(feats).mode().reshape(-1).float()
        continue_prob = self.model.continue_network(feats).base_dist.probs.reshape(-1)
        predicted_terms = continue_prob <= 0.5
        rewards_np = np.asarray([quantize_pong_reward(x, self.threshold) for x in raw_rewards.detach().cpu().numpy()], dtype=np.float32)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        ends = predicted_terms.to(dtype=torch.float32) if self.respect_terminal else torch.zeros_like(rewards)
        self._step += 1
        truncs = (self._step >= self.horizon).to(dtype=torch.float32)
        frames = _decode_frames(self.model, next_state)

        self._record_batch(rewards_np, raw_rewards.detach().cpu().numpy())
        self.current_frames = frames
        self.prev_state = next_state
        self.prev_action = action_onehot

        info = {"raw": [
            {
                "raw_reward": float(raw),
                "terminal_predicted": bool(term),
                "continue_prob": float(cont),
                "terminal_ignored": bool(term and not self.respect_terminal),
            }
            for raw, term, cont in zip(
                raw_rewards.detach().cpu().numpy().tolist(),
                predicted_terms.detach().cpu().numpy().tolist(),
                continue_prob.detach().cpu().numpy().tolist(),
            )
        ]}
        dead = torch.logical_or(ends.bool(), truncs.bool())
        if dead.any():
            dead_indices = dead.nonzero(as_tuple=False).flatten()
            final_frames = self.current_frames[dead_indices.detach().cpu().numpy()]
            info["final_observation"] = self._frames_to_obs_tensor(final_frames)
            for index in dead_indices.detach().cpu().numpy().tolist():
                self._complete_episode(index)
            self._reset_indices(dead_indices)

        return self._frames_to_obs_tensor(self.current_frames), rewards, ends, truncs, info

    def pop_episode_stats(self):
        completed = self._completed
        self._completed = []
        return completed

    def pop_episode_videos(self, max_videos=1):
        return []

    def pop_raw_reward_stats(self):
        if not self._raw_rewards:
            return {}
        values = np.asarray(self._raw_rewards, dtype=np.float32)
        self._raw_rewards = []
        return {
            "raw_reward_mean": float(values.mean()),
            "raw_reward_abs_mean": float(np.abs(values).mean()),
            "raw_reward_nonzero_rate": float((np.abs(values) >= self.threshold).mean()),
            "raw_agent_reward_mean": float(np.maximum(values, 0.0).mean()),
            "raw_opponent_reward_mean": float(np.maximum(-values, 0.0).mean()),
            "raw_agent_reward_nonzero_rate": float((values >= self.threshold).mean()),
            "raw_opponent_reward_nonzero_rate": float((values <= -self.threshold).mean()),
        }

    def close(self):
        for env in self.real_envs:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()

    def _reset_indices(self, indices):
        indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
        posts, dtype = self._initial_posts(indices_cpu)
        frames = _decode_frames(self.model, posts)
        if self.current_frames is None:
            self.current_frames = np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)
            self.prev_state = self.model.rssm.initial(
                batch_size=self.num_envs,
                seq_length=1,
                dtype=dtype,
                device=self.device,
                detach_learned=True,
            )
            if self.prev_state["hidden"] is not None:
                self.prev_state["hidden"] = self.model.rssm.slice_hidden(self.prev_state["hidden"])
            self.prev_action = torch.zeros(self.num_envs, self.num_actions, dtype=dtype, device=self.device)
        self.current_frames[indices_cpu] = frames
        for key, value in posts.items():
            if key == "hidden":
                self.prev_state[key] = _assign_hidden(self.prev_state[key], value, indices)
            else:
                self.prev_state[key][indices] = value
        self.prev_action[indices] = 0.0
        self._step[indices] = 0

    def _initial_posts(self, indices_cpu):
        if self.initial_source == "prior":
            return self._prior_posts(len(indices_cpu))
        if self.initial_source == "dataset":
            return self._dataset_conditioned_posts(len(indices_cpu))
        return self._real_conditioned_posts(indices_cpu)

    def _prior_posts(self, batch_size):
        posts = self.model.rssm.initial(
            batch_size=int(batch_size),
            seq_length=1,
            dtype=torch.float32,
            device=self.device,
            detach_learned=True,
        )
        if posts["hidden"] is not None:
            posts["hidden"] = self.model.rssm.slice_hidden(posts["hidden"])
        return posts, torch.float32

    def _real_conditioned_posts(self, indices_cpu):
        raw_states = []
        for index in indices_cpu:
            obs = self.real_envs[int(index)].reset()
            raw_states.append(obs.state)
        states = torch.stack(raw_states, dim=0).to(self.device)
        states = self.model.preprocess_inputs(states, time_stacked=False)
        latent = self.model.encoder_network(states)
        latent = {key: value.unsqueeze(1) for key, value in latent.items()}
        prev_state = self.model.rssm.initial(
            batch_size=len(indices_cpu),
            seq_length=1,
            dtype=states.dtype,
            device=self.device,
            detach_learned=True,
        )
        prev_action = torch.zeros(len(indices_cpu), 1, self.num_actions, dtype=states.dtype, device=self.device)
        posts, _ = self.model.rssm(
            states=latent,
            prev_states=prev_state,
            prev_actions=prev_action,
            is_firsts=torch.zeros(len(indices_cpu), 1, dtype=states.dtype, device=self.device),
        )
        posts["hidden"] = self.model.rssm.slice_hidden(posts["hidden"])
        return posts, states.dtype

    def _dataset_conditioned_posts(self, batch_size):
        if self.bootstrap_dataset is None:
            raise RuntimeError("TWISTER bootstrap dataset is not initialized.")
        raw_states = [self.bootstrap_dataset.sample() for _ in range(int(batch_size))]
        states = torch.stack(raw_states, dim=0).to(self.device)
        states = self.model.preprocess_inputs(states, time_stacked=False)
        latent = self.model.encoder_network(states)
        latent = {key: value.unsqueeze(1) for key, value in latent.items()}
        prev_state = self.model.rssm.initial(
            batch_size=int(batch_size),
            seq_length=1,
            dtype=states.dtype,
            device=self.device,
            detach_learned=True,
        )
        prev_action = torch.zeros(int(batch_size), 1, self.num_actions, dtype=states.dtype, device=self.device)
        posts, _ = self.model.rssm(
            states=latent,
            prev_states=prev_state,
            prev_actions=prev_action,
            is_firsts=torch.zeros(int(batch_size), 1, dtype=states.dtype, device=self.device),
        )
        posts["hidden"] = self.model.rssm.slice_hidden(posts["hidden"])
        return posts, states.dtype

    def _record_batch(self, rewards, raw_rewards):
        self._episode_scores += rewards
        self._episode_agent_scores += np.maximum(rewards, 0.0)
        self._episode_opponent_scores += np.maximum(-rewards, 0.0)
        self._episode_lengths += 1
        self._episode_raw_scores += raw_rewards
        self._episode_raw_agent_scores += np.maximum(raw_rewards, 0.0)
        self._episode_raw_opponent_scores += np.maximum(-raw_rewards, 0.0)
        self._episode_has_raw_reward[:] = True
        self._raw_rewards.extend(raw_rewards.tolist())

    def _complete_episode(self, index):
        completed = {
            "score": float(self._episode_scores[index]),
            "agent_score": float(self._episode_agent_scores[index]),
            "opponent_score": float(self._episode_opponent_scores[index]),
            "length": int(self._episode_lengths[index]),
        }
        if self._episode_has_raw_reward[index]:
            completed["raw_score"] = float(self._episode_raw_scores[index])
            completed["raw_agent_score"] = float(self._episode_raw_agent_scores[index])
            completed["raw_opponent_score"] = float(self._episode_raw_opponent_scores[index])
        self._completed.append(completed)
        self._episode_scores[index] = 0.0
        self._episode_agent_scores[index] = 0.0
        self._episode_opponent_scores[index] = 0.0
        self._episode_raw_scores[index] = 0.0
        self._episode_raw_agent_scores[index] = 0.0
        self._episode_raw_opponent_scores[index] = 0.0
        self._episode_has_raw_reward[index] = False
        self._episode_lengths[index] = 0

    def _frames_to_obs_tensor(self, frames):
        tensor = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        return tensor.div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()


def make_real_pixel_envs(env_name, seed, num_envs, device):
    twister_env_name = _env_id_to_twister_name(env_name)
    envs = []
    for index in range(num_envs):
        env = _make_twister_env(twister_env_name, int(seed) + index)
        envs.append(TwisterAtariPixelEnv(env, env.num_actions))
    return VectorPixelEnv(envs, device)


def make_wm_pixel_envs(
    *,
    env_name,
    seed,
    checkpoint,
    horizon,
    num_envs,
    device,
    reward_threshold,
    respect_terminal,
    initial_source="real",
    bootstrap_dataset=None,
):
    return BatchedTwisterWMPixelEnv(
        env_name=env_name,
        seed=seed,
        checkpoint=checkpoint,
        horizon=horizon,
        num_envs=num_envs,
        device=device,
        reward_threshold=reward_threshold,
        respect_terminal=respect_terminal,
        initial_source=initial_source,
        bootstrap_dataset=bootstrap_dataset,
    )


def _make_twister_env(env_name, seed):
    model = nnet.models.TWISTER(
        env_name=env_name,
        override_config={"num_envs": 1, "eval_episodes": 0},
    )
    params = dict(model.config.env_params)
    params["seed"] = int(seed)
    env = model.config.env_class(**params)
    del model
    return env


def _load_twister_model(env_name, checkpoint, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"TWISTER WM checkpoint not found: {checkpoint}")
    model = nnet.models.TWISTER(
        env_name=env_name,
        override_config={
            "num_envs": 1,
            "eval_episodes": 0,
            "load_replay_buffer_state_dict": False,
        },
    )
    model.compile()
    model.to(device)
    model.load(str(checkpoint), load_optimizer=False, verbose=True, strict=True)
    return model


def _decode_frames(model, state):
    dist = model.decoder_network(state["stoch"].flatten(-2, -1))
    tensor = dist.mode().detach().float().clamp(-0.5, 0.5).add(0.5).mul(255).to(torch.uint8)
    if tensor.ndim == 5 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    return tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _chw_uint8_to_hwc(state):
    state = state.detach().cpu()
    if state.ndim == 4:
        state = state[0]
    return state.permute(1, 2, 0).contiguous().numpy().astype(np.uint8, copy=False)


def _assign_hidden(dst_hidden, src_hidden, indices):
    if dst_hidden is None:
        return src_hidden
    updated = []
    for dst_blk, src_blk in zip(dst_hidden, src_hidden):
        k, v = dst_blk
        sk, sv = src_blk
        k = k.clone()
        v = v.clone()
        k[indices] = sk
        v[indices] = sv
        updated.append((k, v))
    return updated
