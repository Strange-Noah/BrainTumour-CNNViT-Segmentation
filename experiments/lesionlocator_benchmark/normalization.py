"""Offline intensity characterization under the fixed official CT transform."""
import numpy as np
from scipy import ndimage


def characterize(image, foreground, parameters):
    x = np.asarray(image, dtype=np.float32)
    if not np.isfinite(x).all() or x.shape != foreground.shape:
        raise ValueError("Invalid offline intensity input")
    low, high, mean, std = (parameters[key] for key in
                             ("percentile_00_5", "percentile_99_5", "mean", "std"))
    if not np.isfinite([low, high, mean, std]).all() or std <= 0 or low >= high:
        raise ValueError("Invalid fixed CT normalization")
    nonzero = x != 0
    shell = ndimage.binary_dilation(foreground, iterations=2) & ~foreground & nonzero

    def stats(y):
        if not y.size:
            return {"n": 0}
        normalized = (np.clip(y, low, high)-mean)/std
        return dict(n=int(y.size), quantiles=np.quantile(y, [0, .005, .5, .995, 1]).tolist(),
                    low_clip=float(np.mean(y < low)), high_clip=float(np.mean(y > high)),
                    raw_mean=float(y.mean()), normalized_mean=float(normalized.mean()))

    g, b = stats(x[foreground]), stats(x[shell])
    contrast = {key: g[key]-b[key] for key in ("raw_mean", "normalized_mean")} if g["n"] and b["n"] else {}
    return dict(full=stats(x.ravel()), nonzero=stats(x[nonzero]), lesion=g,
                nonzero_perilesional_shell=b, signed_mean_contrast=contrast)
