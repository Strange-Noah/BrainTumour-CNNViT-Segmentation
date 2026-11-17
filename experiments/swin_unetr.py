"""Yilong entry point for a task-specific MONAI Swin UNETR checkpoint."""
import os

_MODEL = None
_MODEL_KEY = None


def predict(sample):
    global _MODEL, _MODEL_KEY
    import os
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from experiments.swin_unetr_common import build_model, normalize_modalities

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = int(os.environ.get("SWIN_FEATURE_SIZE", "24"))
    roi = int(os.environ.get("SWIN_ROI_SIZE", "96"))
    if roi % 32:
        raise ValueError("SWIN_ROI_SIZE must be divisible by 32")
    checkpoint_path = os.environ.get("SWIN_CHECKPOINT")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Set SWIN_CHECKPOINT to the trained Swin UNETR state dict")

    model_key = (os.path.abspath(checkpoint_path), feature_size, str(device))
    if _MODEL is None or _MODEL_KEY != model_key:
        model = build_model(feature_size, use_checkpoint=False).to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _MODEL = model
        _MODEL_KEY = model_key
    model = _MODEL
    data = normalize_modalities(sample["images"])
    with torch.inference_mode():
        kwargs = {"device": torch.device("cpu")} if device.type == "cuda" else {}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                torch.from_numpy(data[None]),
                (roi, roi, roi), 1, model, overlap=0.25,
                sw_device=device, **kwargs
            )
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
