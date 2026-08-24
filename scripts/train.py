from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTE_B_SCRIPT = PROJECT_ROOT / "scripts/04_train_route_b.py"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for scripts/train.py --config. Install pyyaml first.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")
    return loaded


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "-1", "0"}:
        return None
    return int(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config wrapper for PBL BrainTumour training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "run_outputs/route_b_sanity_gpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=100)
    parser.add_argument("--max-val-steps-per-epoch", type=int, default=25)
    parser.add_argument("--max-train-patients", type=optional_int, default=None)
    parser.add_argument("--max-val-patients", type=optional_int, default=8)
    parser.add_argument("--max-slices-per-val-patient", type=optional_int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--trainable-vit-blocks", type=int, default=4)
    parser.add_argument("--trainable-resnet-stages", type=str, default="2")
    parser.add_argument("--train-stem", dest="train_stem", action="store_true", default=True)
    parser.add_argument("--no-train-stem", dest="train_stem", action="store_false")
    parser.add_argument("--train-pos-embed", dest="train_pos_embed", action="store_true", default=True)
    parser.add_argument("--no-train-pos-embed", dest="train_pos_embed", action="store_false")
    parser.add_argument("--lr-new", type=float, default=3e-4)
    parser.add_argument("--lr-pretrained", type=float, default=3e-5)
    parser.add_argument("--lr-pos-embed", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-train-sampler", dest="balanced_train_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-train-sampler", dest="balanced_train_sampler", action="store_false")
    parser.add_argument("--train-sampling-mode", choices=["patient_local", "random"], default="patient_local")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-log-every", type=int, default=500)
    parser.add_argument("--val-log-every", type=int, default=200)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = load_yaml(config_path)

    model_name = config.get("model", {}).get("name")
    if model_name != "pretrained_hybrid_vit_route_b":
        raise SystemExit(f"scripts/train.py currently supports Route B only, got model.name={model_name!r}")

    split_manifest = project_path(config["data"]["split_manifest"])
    data_root = project_path(config.get("data", {}).get("root", "data/processed/Task01_BrainTumour"))
    weights = project_path(config["pretrained"]["weights"])

    if args.full_train and not args.build_only and args.max_train_patients is not None:
        raise SystemExit("Full training must use all training patients; unset --max-train-patients.")

    epochs = args.epochs
    batch_size = args.batch_size
    max_train_steps = "none" if args.full_train else args.max_train_steps_per_epoch
    max_val_steps = "none" if args.full_train else args.max_val_steps_per_epoch
    max_val_patients = "none" if args.full_train else args.max_val_patients
    max_slices_per_val_patient = "none" if args.full_train else args.max_slices_per_val_patient

    argv = [
        str(ROUTE_B_SCRIPT),
        "--split-manifest",
        str(split_manifest),
        "--data-root",
        str(data_root),
        "--weights",
        str(weights),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--trainable-vit-blocks",
        str(args.trainable_vit_blocks),
        "--trainable-resnet-stages",
        str(args.trainable_resnet_stages),
        "--lr-new",
        str(args.lr_new),
        "--lr-pretrained",
        str(args.lr_pretrained),
        "--lr-pos-embed",
        str(args.lr_pos_embed),
        "--weight-decay",
        str(args.weight_decay),
        "--gpu-ids",
        str(args.gpu_ids),
        "--train-sampling-mode",
        str(args.train_sampling_mode),
        "--train-log-every",
        str(args.train_log_every),
        "--val-log-every",
        str(args.val_log_every),
    ]
    if args.train_stem:
        argv.append("--train-stem")
    else:
        argv.append("--no-train-stem")
    if args.train_pos_embed:
        argv.append("--train-pos-embed")
    else:
        argv.append("--no-train-pos-embed")
    if args.balanced_train_sampler:
        argv.append("--balanced-train-sampler")
    else:
        argv.append("--no-balanced-train-sampler")
    if args.data_parallel:
        argv.append("--data-parallel")
    if args.amp:
        argv.append("--amp")
    if max_train_steps is not None:
        argv += ["--max-train-steps-per-epoch", str(max_train_steps)]
    if max_val_steps is not None:
        argv += ["--max-val-steps-per-epoch", str(max_val_steps)]
    if args.max_train_patients is not None:
        argv += ["--max-train-patients", str(args.max_train_patients)]
    if max_val_patients is not None:
        argv += ["--max-val-patients", str(max_val_patients)]
    if max_slices_per_val_patient is not None:
        argv += ["--max-slices-per-val-patient", str(max_slices_per_val_patient)]
    if args.full_train:
        argv.append("--require-full-val")
    if args.build_only:
        argv.append("--build-only")

    sys.argv = argv
    runpy.run_path(str(ROUTE_B_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
