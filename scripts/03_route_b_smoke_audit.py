from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "models/pretrained/timm/vit_base_r50_s16_224.orig_in21k/model.safetensors"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/model_audit/route_b_smoke_audit.json"

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.pretrained_hybrid_vit import (  # noqa: E402
    FIRST_CONV_KEY,
    POS_EMBED_KEY,
    PretrainedHybridViTSegmenter,
    adapt_first_conv_weight,
    count_parameters,
    interpolate_pos_embed_2d,
    load_checkpoint_tensor_dict,
)


def tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    finite = torch.isfinite(tensor)
    return {
        "shape": list(tensor.shape),
        "finite": bool(finite.all().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def collect_trainable_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def check_optimizer_param_uniqueness(model: nn.Module) -> dict[str, object]:
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    param_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
    return {
        "trainable_param_objects": len(trainable),
        "optimizer_param_objects": len(param_ids),
        "unique_optimizer_param_objects": len(set(param_ids)),
        "covers_all_trainable_once": len(trainable) == len(param_ids) == len(set(param_ids)),
    }


def run_audit(weights_path: Path) -> dict[str, object]:
    torch.manual_seed(731)

    original_state = load_checkpoint_tensor_dict(weights_path)
    expected_conv4 = adapt_first_conv_weight(original_state[FIRST_CONV_KEY], in_chans=4)
    expected_pos226 = interpolate_pos_embed_2d(original_state[POS_EMBED_KEY], (15, 15))

    model = PretrainedHybridViTSegmenter(weights_path=weights_path)
    model.eval()

    loaded_conv = model.encoder.patch_embed.backbone.stem.conv.weight
    loaded_pos = model.encoder.pos_embed

    hook_outputs: dict[str, dict[str, object]] = {}

    def hook(name: str):
        def capture(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            hook_outputs[name] = tensor_stats(output.detach())

        return capture

    hooks = [
        model.encoder.patch_embed.backbone.stem.conv.register_forward_hook(hook("stem.conv")),
        model.encoder.patch_embed.backbone.stem.pool.register_forward_hook(hook("stem.pool")),
        model.encoder.patch_embed.backbone.stages[2].register_forward_hook(hook("resnet.stage2")),
        model.encoder.patch_embed.proj.register_forward_hook(hook("hybrid.proj")),
        model.encoder.blocks[8].register_forward_hook(hook("vit.block8")),
        model.encoder.blocks[11].register_forward_hook(hook("vit.block11")),
    ]

    x_zero = torch.zeros(1, 4, 240, 240)
    x_rand = torch.randn(1, 4, 240, 240)

    with torch.no_grad():
        zero_logits = model(x_zero)
        feature_grid = model.forward_features_grid(x_rand)
        logits_1 = model(x_rand)
        logits_2 = model(x_rand)

    for handle in hooks:
        handle.remove()

    trainable_names = collect_trainable_names(model)
    frozen_norms_with_running_stats = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and not any(param.requires_grad for param in module.parameters())
    ]
    sigmoid_modules = [name for name, module in model.named_modules() if isinstance(module, nn.Sigmoid)]

    pos_before = loaded_pos.detach().clone()
    with torch.no_grad():
        _ = model(x_zero)
        _ = model(x_zero)
    pos_changed_during_forward = max_abs_diff(pos_before, model.encoder.pos_embed) > 0.0

    return {
        "weights_path": str(weights_path),
        "route": "B_native_240_with_4ch_stem_and_15x15_pos_embed",
        "parameter_counts": count_parameters(model),
        "adapted_weight_shapes": {
            "first_conv": list(loaded_conv.shape),
            "pos_embed": list(loaded_pos.shape),
        },
        "weight_source_checks": {
            "first_conv_matches_expected_rgb_mean_scaled_3_over_4": max_abs_diff(
                loaded_conv, expected_conv4
            ),
            "pos_embed_matches_expected_bicubic_14_to_15": max_abs_diff(
                loaded_pos, expected_pos226
            ),
            "cls_pos_embed_unchanged": max_abs_diff(
                loaded_pos[:, :1], original_state[POS_EMBED_KEY][:, :1]
            ),
            "pos_embed_changed_during_two_forwards": pos_changed_during_forward,
        },
        "runtime_shapes_and_stats": {
            "hooks": hook_outputs,
            "feature_grid": tensor_stats(feature_grid),
            "zero_logits": tensor_stats(zero_logits),
            "random_logits": tensor_stats(logits_1),
        },
        "interface_checks": {
            "input_shape": [1, 4, 240, 240],
            "feature_grid_shape_ok": list(feature_grid.shape) == [1, 768, 15, 15],
            "logits_shape_ok": list(logits_1.shape) == [1, 1, 240, 240],
            "zero_forward_finite": bool(torch.isfinite(zero_logits).all().item()),
            "random_forward_finite": bool(torch.isfinite(logits_1).all().item()),
            "eval_forward_deterministic": max_abs_diff(logits_1, logits_2) == 0.0,
            "decoder_has_sigmoid": bool(sigmoid_modules),
            "sigmoid_modules": sigmoid_modules,
        },
        "freezing_checks": {
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_name_count": len(trainable_names),
            "pos_embed_trainable": bool(model.encoder.pos_embed.requires_grad),
            "frozen_batchnorm_modules_with_running_stats": frozen_norms_with_running_stats,
            "has_batchnorm_modules": any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in model.modules()
            ),
        },
        "optimizer_checks": check_optimizer_param_uniqueness(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Route B pretrained hybrid ViT segmenter.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = run_audit(args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    checks = audit["interface_checks"]
    weight_checks = audit["weight_source_checks"]
    optimizer_checks = audit["optimizer_checks"]

    failures = []
    for key in [
        "feature_grid_shape_ok",
        "logits_shape_ok",
        "zero_forward_finite",
        "random_forward_finite",
        "eval_forward_deterministic",
    ]:
        if not checks[key]:
            failures.append(key)
    if checks["decoder_has_sigmoid"]:
        failures.append("decoder_has_sigmoid")
    if weight_checks["first_conv_matches_expected_rgb_mean_scaled_3_over_4"] != 0.0:
        failures.append("first_conv_weight_source")
    if weight_checks["pos_embed_matches_expected_bicubic_14_to_15"] != 0.0:
        failures.append("pos_embed_weight_source")
    if weight_checks["cls_pos_embed_unchanged"] != 0.0:
        failures.append("cls_pos_embed_changed")
    if weight_checks["pos_embed_changed_during_two_forwards"]:
        failures.append("pos_embed_repeated_or_mutated_during_forward")
    if not optimizer_checks["covers_all_trainable_once"]:
        failures.append("optimizer_trainable_param_coverage")

    print(f"Wrote audit: {args.output}")
    print(f"Route B logits shape OK: {checks['logits_shape_ok']}")
    print(f"Route B finite random forward: {checks['random_forward_finite']}")
    print(f"Route B deterministic eval forward: {checks['eval_forward_deterministic']}")
    print(f"Trainable parameter names: {audit['freezing_checks']['trainable_parameter_name_count']}")
    print(f"Parameter counts: {audit['parameter_counts']}")
    if failures:
        raise SystemExit(f"Route B audit failed checks: {failures}")


if __name__ == "__main__":
    main()

