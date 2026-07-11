from __future__ import annotations


def quantize_pong_reward(reward: float, threshold: float = 0.001) -> float:
    reward = float(reward)
    threshold = float(threshold)
    if reward >= threshold:
        return 1.0
    if reward <= -threshold:
        return -1.0
    return 0.0
