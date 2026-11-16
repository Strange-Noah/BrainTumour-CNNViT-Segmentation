"""Port of the frozen summer Route B model with UCSD slice adaptation."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cnn_vit_model import PretrainedHybridViTSegmenter


def load_model(checkpoint, backbone=None, diagnostic_input=None):
    """Retain compatible weights but reset the task head for UCSD label 3."""
    model = PretrainedHybridViTSegmenter(
        weights_path=backbone, num_classes=1, trainable_vit_blocks=0,
        trainable_resnet_stages=(), train_stem=False, train_pos_embed=False,
    )
    report = {"initialization": "pretrained_backbone" if backbone else "random",
              "loaded_parameters": [], "excluded_parameters": []}
    if checkpoint:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = data.get("model_state_dict", data.get("state_dict", data))
        classes = int(state["decoder.head.weight"].shape[0])
        if classes not in (1, 4):
            raise ValueError("Unknown summer checkpoint head")
        binary_head = model.decoder.head
        model.decoder.head = torch.nn.Conv2d(binary_head.in_channels, classes, 1)
        model.load_state_dict(state, strict=True)
        report["summer_strict_load"] = True
        report["summer_native_classes"] = classes
        if diagnostic_input is not None:
            model.eval()
            with torch.inference_mode():
                report["summer_native_forward_shape"] = list(model(diagnostic_input).shape)
        model.decoder.head = binary_head
        expected = model.state_dict()
        matched = {}
        for key, value in state.items():
            if key.startswith("decoder.head.") or key not in expected or expected[key].shape != value.shape:
                report["excluded_parameters"].append(key)
            else:
                matched[key] = value
        if not matched or not any(key.startswith("encoder.") for key in matched):
            raise ValueError("Summer checkpoint has no compatible encoder weights")
        model.load_state_dict(matched, strict=False)
        report["initialization"] = "summer_encoder_and_compatible_decoder"
        report["loaded_parameters"] = sorted(matched)
    # Small first run trains the decoder head only. Owners may unfreeze native stages.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("decoder.head.")
    return model, report


def preprocess(images, z):
    """4-channel slice normalization from summer, then bilinear resize to 240."""
    image = np.array(images[:, :, :, z], dtype=np.float32, copy=True)
    brain = np.any(image != 0, axis=0)
    for channel in image:
        values = channel[brain]
        if values.size:
            channel[brain] = (values - values.mean()) / max(float(values.std()), 1e-6)
        channel[~brain] = 0
    return F.interpolate(torch.from_numpy(image[None]), size=(240, 240),
                         mode="bilinear", align_corners=False)


def slice_target(mask, z):
    target = torch.from_numpy(mask[:, :, z].astype(np.float32)[None, None])
    return F.interpolate(target, size=(240, 240), mode="nearest")


def infer(model, images, device):
    """All axial slices, no label input, restored to original T1c voxel grid."""
    model.eval()
    mask = np.zeros(images.shape[1:], dtype=np.uint8)
    with torch.inference_mode():
        for z in range(images.shape[-1]):
            logits = model(preprocess(images, z).to(device))
            restored = F.interpolate(logits, size=images.shape[1:3],
                                     mode="bilinear", align_corners=False)
            mask[:, :, z] = (torch.sigmoid(restored)[0, 0].cpu().numpy() >= 0.5)
    return mask
