from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


FIRST_CONV_KEY = "patch_embed.backbone.stem.conv.weight"
POS_EMBED_KEY = "pos_embed"


def interpolate_pos_embed_2d(
    pos_embed: torch.Tensor,
    new_grid_size: tuple[int, int],
    *,
    old_grid_size: tuple[int, int] | None = None,
    num_prefix_tokens: int = 1,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate ViT absolute image-position embeddings to a new 2D grid."""
    if pos_embed.ndim != 3:
        raise ValueError(f"Expected pos_embed with shape (1, N, C), got {tuple(pos_embed.shape)}")
    if pos_embed.shape[0] != 1:
        raise ValueError("Only batch-size-1 absolute positional embeddings are supported.")

    prefix = pos_embed[:, :num_prefix_tokens]
    image_pos = pos_embed[:, num_prefix_tokens:]
    old_token_count = image_pos.shape[1]

    if old_grid_size is None:
        old_side = int(old_token_count**0.5)
        old_grid_size = (old_side, old_side)

    old_h, old_w = old_grid_size
    new_h, new_w = new_grid_size
    if old_h * old_w != old_token_count:
        raise ValueError(
            f"old_grid_size={old_grid_size} does not match {old_token_count} image tokens."
        )

    image_pos = image_pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
    image_pos = F.interpolate(
        image_pos,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
    )
    image_pos = image_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    return torch.cat([prefix, image_pos], dim=1)


def adapt_first_conv_weight(
    weight: torch.Tensor,
    in_chans: int,
    *,
    scale_to_preserve_sum: bool = True,
) -> torch.Tensor:
    """Adapt an RGB first-conv kernel to a different input-channel count."""
    if weight.ndim != 4:
        raise ValueError(f"Expected conv weight with shape (O, I, H, W), got {tuple(weight.shape)}")
    old_in_chans = weight.shape[1]
    if old_in_chans == in_chans:
        return weight
    if in_chans <= 0:
        raise ValueError(f"in_chans must be positive, got {in_chans}")

    mean_kernel = weight.mean(dim=1, keepdim=True)
    adapted = mean_kernel.repeat(1, in_chans, 1, 1)
    if scale_to_preserve_sum:
        adapted = adapted * (old_in_chans / float(in_chans))
    return adapted


def adapt_hybrid_vit_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    target_first_conv_shape: tuple[int, ...],
    target_pos_embed_shape: tuple[int, ...],
    target_grid_size: tuple[int, int],
    num_prefix_tokens: int = 1,
) -> dict[str, torch.Tensor]:
    """Adapt pretrained 224/RGB ViT-R50 weights for 240/4-channel Route B."""
    adapted = dict(state_dict)

    if FIRST_CONV_KEY in adapted and tuple(adapted[FIRST_CONV_KEY].shape) != target_first_conv_shape:
        adapted[FIRST_CONV_KEY] = adapt_first_conv_weight(
            adapted[FIRST_CONV_KEY],
            in_chans=target_first_conv_shape[1],
        )

    if POS_EMBED_KEY in adapted and tuple(adapted[POS_EMBED_KEY].shape) != target_pos_embed_shape:
        adapted[POS_EMBED_KEY] = interpolate_pos_embed_2d(
            adapted[POS_EMBED_KEY],
            new_grid_size=target_grid_size,
            num_prefix_tokens=num_prefix_tokens,
        )

    return adapted


def load_checkpoint_tensor_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type at {checkpoint_path}: {type(checkpoint)!r}")
    return checkpoint


class UpsampleSegmentationDecoder(nn.Module):
    """Light decoder for 15x15 or 14x14 ViT feature grids."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: Iterable[int] = (384, 192, 96, 48),
        output_size: tuple[int, int] = (240, 240),
    ):
        super().__init__()
        channels = [in_channels, *hidden_channels]
        blocks: list[nn.Module] = []
        for src, dst in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(src, dst, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=_largest_valid_group_count(dst), num_channels=dst),
                    nn.GELU(),
                ]
            )

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels[-1], out_channels, kernel_size=1)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return self.head(x)


def _largest_valid_group_count(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class PretrainedHybridViTSegmenter(nn.Module):
    """Route B segmenter: 4-channel 240x240 MRI -> 240x240 tumour logits."""

    def __init__(
        self,
        *,
        model_name: str = "vit_base_r50_s16_224.orig_in21k",
        weights_path: str | Path | None = None,
        in_chans: int = 4,
        img_size: int = 240,
        num_classes: int = 1,
        decoder_channels: Iterable[int] = (384, 192, 96, 48),
        trainable_vit_blocks: int = 4,
        trainable_resnet_stages: Iterable[int] = (2,),
        train_stem: bool = True,
        train_pos_embed: bool = True,
    ):
        super().__init__()

        import timm

        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
            img_size=img_size,
        )
        self.num_prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1))
        self.grid_size = tuple(int(v) for v in self.encoder.patch_embed.grid_size)
        self.embed_dim = int(self.encoder.embed_dim)
        self.img_size = (img_size, img_size)

        if weights_path is not None:
            self.load_pretrained_route_b(weights_path)

        self.decoder = UpsampleSegmentationDecoder(
            in_channels=self.embed_dim,
            out_channels=num_classes,
            hidden_channels=decoder_channels,
            output_size=self.img_size,
        )

        self.configure_trainable_layers(
            trainable_vit_blocks=trainable_vit_blocks,
            trainable_resnet_stages=trainable_resnet_stages,
            train_stem=train_stem,
            train_pos_embed=train_pos_embed,
        )

    def load_pretrained_route_b(self, weights_path: str | Path) -> None:
        state_dict = load_checkpoint_tensor_dict(weights_path)
        first_conv = self.encoder.patch_embed.backbone.stem.conv.weight
        state_dict = adapt_hybrid_vit_state_dict(
            state_dict,
            target_first_conv_shape=tuple(first_conv.shape),
            target_pos_embed_shape=tuple(self.encoder.pos_embed.shape),
            target_grid_size=self.grid_size,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Unexpected pretrained-weight load mismatch after adaptation: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def configure_trainable_layers(
        self,
        *,
        trainable_vit_blocks: int,
        trainable_resnet_stages: Iterable[int],
        train_stem: bool,
        train_pos_embed: bool,
    ) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

        if train_pos_embed:
            self.encoder.pos_embed.requires_grad = True

        if train_stem:
            for param in self.encoder.patch_embed.backbone.stem.parameters():
                param.requires_grad = True

        stages = self.encoder.patch_embed.backbone.stages
        for stage_idx in trainable_resnet_stages:
            for param in stages[int(stage_idx)].parameters():
                param.requires_grad = True

        if trainable_vit_blocks > 0:
            for block in self.encoder.blocks[-trainable_vit_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.encoder.norm.parameters():
                param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward_features_grid(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        image_tokens = features[:, self.num_prefix_tokens :, :]
        expected_tokens = self.grid_size[0] * self.grid_size[1]
        if image_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} image tokens for grid {self.grid_size}, "
                f"got {image_tokens.shape[1]}."
            )
        batch = image_tokens.shape[0]
        return image_tokens.transpose(1, 2).reshape(
            batch,
            self.embed_dim,
            self.grid_size[0],
            self.grid_size[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(f"Expected spatial size {self.img_size}, got {tuple(x.shape[-2:])}")
        feature_grid = self.forward_features_grid(x)
        return self.decoder(feature_grid)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
