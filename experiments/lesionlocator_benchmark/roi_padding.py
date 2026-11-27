"""Spatial padding for inputs rejected by the official TrackNet tensor-axis order."""
from torch.nn.functional import pad


def spatial_padding(x, target_shape):
    widths = tuple(int(n)-s for s, n in zip(x.shape[-3:], target_shape))
    if any(n < 0 for n in widths):
        raise ValueError("ROI exceeds the official patch")
    return tuple(i for n in widths for i in (n//2, n-n//2))


def pad_spatial(x, widths):
    d0,d1,h0,h1,w0,w1 = widths
    return pad(x, (w0,w1,h0,h1,d0,d1), mode="constant", value=0)


def unpad_spatial(x, widths):
    d0,d1,h0,h1,w0,w1 = widths
    return x[..., d0:x.shape[-3]-d1, h0:x.shape[-2]-h1, w0:x.shape[-1]-w1]
