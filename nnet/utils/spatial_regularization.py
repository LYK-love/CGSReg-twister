from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpatialRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    mask_weights: dict[str, float] = field(
        default_factory=lambda: {"mask1": 1.0, "mask2": 0.0, "mask3": 0.0}
    )
    image_channels: int = 3

    def __post_init__(self):
        normalized = {
            "mask1": float(self.mask_weights.get("mask1", 1.0)),
            "mask2": float(self.mask_weights.get("mask2", 0.0)),
            "mask3": float(self.mask_weights.get("mask3", 0.0)),
        }
        unknown = sorted(set(self.mask_weights) - set(normalized))
        if unknown:
            raise ValueError(f"Unknown spatial regularization mask keys: {unknown}")
        non_binary = {k: v for k, v in normalized.items() if v not in (0.0, 1.0)}
        if non_binary:
            raise ValueError(
                "Spatial regularization mask weights are binary switches; "
                f"got {non_binary}. Use weight for continuous scaling."
            )
        self.mask_weights = normalized


def compute_spatial_regularization(
    pred: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor | None,
    config: SpatialRegularizationConfig,
    return_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not config.enabled or masks is None or masks.shape[2] == 0:
        return pred.new_zeros(()), {}

    pred = pred[:, :, : config.image_channels]
    target = target[:, :, : config.image_channels]
    masks = masks.float()
    if masks.ndim == 4:
        masks = masks.unsqueeze(2)
    if masks.shape[-2:] != pred.shape[-2:]:
        raise ValueError(f"Mask shape {tuple(masks.shape)} does not match image shape {tuple(pred.shape)}")

    sqerr = (pred - target.detach()) ** 2
    total_area = sqerr.shape[2] * sqerr.shape[3] * sqerr.shape[4]
    total_sqerr_sum = sqerr.sum(dim=(2, 3, 4)).clamp_min(1e-8)

    metrics: dict[str, torch.Tensor] = {}
    total_mask_mean = pred.new_zeros(pred.shape[:2])
    total_mask_sum = pred.new_zeros(pred.shape[:2])
    for idx in range(min(3, masks.shape[2])):
        key = f"mask{idx + 1}"
        mask_weight = float(config.mask_weights.get(key, 0.0))
        mask = (masks[:, :, idx:idx + 1] > 0.5).to(dtype=sqerr.dtype)
        mask = mask.expand_as(sqerr)
        masked_sqerr = sqerr * mask
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        mask_sum = masked_sqerr.sum(dim=(2, 3, 4))
        mask_mean = mask_sum / denom
        mask_loss = mask_mean * total_area
        mask_scaled = mask_loss.mean() * mask_weight * config.weight

        total_mask_mean = total_mask_mean + mask_mean * mask_weight
        total_mask_sum = total_mask_sum + mask_sum * mask_weight

        if return_metrics:
            metrics[f"spatial_regu/mask_mean/{key}"] = mask_mean.mean()
            metrics[f"spatial_regu/mask_sum/{key}"] = mask_sum.mean()
            metrics[f"spatial_regu/mask_sum_ratio/{key}"] = (mask_sum / total_sqerr_sum).mean()
            metrics[f"spatial_regu/mask_weight/{key}"] = pred.new_tensor(mask_weight)
            metrics[f"spatial_regu/mask_mean_weighted/{key}"] = (mask_mean * mask_weight).mean()
            metrics[f"spatial_regu/mask_sum_weighted/{key}"] = (mask_sum * mask_weight).mean()
            metrics[f"spatial_regu/important_ratio/{key}"] = mask.mean()
            metrics[f"loss_unscaled/spatial_regu/{key}"] = mask_loss.mean()
            metrics[f"loss_scaled/spatial_regu/{key}"] = mask_scaled

    spatial_unscaled = total_mask_mean * total_area
    spatial_loss = spatial_unscaled.mean()
    spatial_scaled = spatial_loss * config.weight
    if not return_metrics:
        return spatial_scaled, {}

    metrics["loss_unscaled/spatial_regu"] = spatial_loss
    metrics["loss_scaled/spatial_regu"] = spatial_scaled
    metrics["spatial_regu_stat/mask_mean"] = total_mask_mean.mean()
    metrics["spatial_regu_stat/mask_sum"] = total_mask_sum.mean()
    metrics["spatial_regu_stat/mask_sum_ratio"] = (total_mask_sum / total_sqerr_sum).mean()
    metrics["spatial_regu/num_masks"] = pred.new_tensor(float(min(3, masks.shape[2])))
    metrics["spatial_regu/total_mask_weight"] = pred.new_tensor(sum(config.mask_weights.values()))
    metrics["spatial_regu/total_area"] = pred.new_tensor(float(total_area))

    return spatial_scaled, metrics
