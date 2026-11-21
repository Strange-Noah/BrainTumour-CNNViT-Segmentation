"""Non-secret source repository configuration."""

import os
import re


def source_archive_prefix():
    repo = os.environ.get(
        "NOVO_REPOSITORY",
        "Strange-Noah/BrainTumour-CNNViT-Segmentation",
    )
    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo
    ):
        raise ValueError("Invalid GitHub repository name")
    return f"https://github.com/{repo}/archive/"
