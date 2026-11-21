# Novo Nordisk — U-Net 2.5D Workstream

This is the owner's long-lived model-development branch for
longitudinal brain metastasis segmentation on UCSD-BMETS.

The project-wide dataset, shared interfaces and public project entry
are maintained on [main](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/main).

## Workstream

- Model acquisition and SMP audit: [experiments/smp_audit](experiments/smp_audit/)
- U-Net implementation: [experiments/unet25d](experiments/unet25d/)
- Full-cohort training: [cohort](experiments/unet25d/cohort/)
- Experiment versions: [run index](experiments/unet25d/runs/README.md)
- Failed baseline v1: [results](experiments/unet25d/runs/v1/README.md)
- Historical migration evidence: [archive](docs/archive/repository-migration/)

The v1 training campaign is preserved as a failed research baseline,
not an accepted segmentation model or longitudinal deployment.

## Development policy

Continue all related model development on this branch.
Use commits, scientific configuration hashes, run directories and
immutable source tags to distinguish experiment versions.

Do not create permanent branches for individual tasks or experiments.
Do not merge the complete workstream into main.

## Data and execution

Use the frozen project dataset and patient split.
Keep medical images, checkpoints and per-patient results in private R2.
GPU authorization is explicit and separate from repository changes.

Credential-bearing automated execution is dispatched from the trusted
main workflow using an explicitly approved immutable workstream source.

See [AGENTS.md](AGENTS.md) for repository governance.
