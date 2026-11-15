"""Zihan/Kaile entry point for a task-specific TorchScript CNN–ViT export."""
import os


def predict(sample):
    import numpy as np
    import torch
    # Owner export must embed trained preprocessing and return original-grid logits.
    model = torch.jit.load(os.environ["CNN_VIT_SCRIPT"], map_location="cpu").eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(sample["images"]).unsqueeze(0))
    if scores.ndim != 5 or scores.shape[0] != 1 or scores.shape[1] != 2:
        raise ValueError("Expected task-specific [1,2,X,Y,Z] logits")
    return scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
