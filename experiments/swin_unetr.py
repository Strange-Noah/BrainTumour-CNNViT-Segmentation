"""Yilong entry point for a task-specific MONAI Swin UNETR checkpoint."""
import os


def predict(sample):
    import numpy as np
    import torch
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(in_channels=4, out_channels=2, feature_size=48).to(device)
    model.load_state_dict(torch.load(os.environ["SWIN_CHECKPOINT"], map_location=device, weights_only=True), strict=True)
    model.eval()
    data = sample["images"].copy()
    # Example train-compatible per-volume normalization. Record/replace with owner's recipe.
    for channel in data:
        foreground = channel != 0
        if foreground.any():
            channel[foreground] = (channel[foreground] - channel[foreground].mean()) / max(channel[foreground].std(), 1e-6)
    with torch.inference_mode():
        logits = sliding_window_inference(torch.from_numpy(data[None]).to(device),
                                          (96, 96, 96), 1, model, overlap=0.25)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
