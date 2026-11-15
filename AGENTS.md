# Novo Nordisk project instructions

- Read [the current handoff](docs/longitudinal/handoff.md) and [verification ledger](docs/longitudinal/verification.md) before changing resources.
- Use feature branches for coordinated development. Team writers may push and merge normally. Preserve summer-archive. Avoid force-pushes and history rewrites.
- Share one UCSD-BMETS version and patient/visit manifest. Assign partitions by patient and freeze them before experiments.
- Keep imaging, clinical tables, checkpoints and per-patient outputs in the approved private storage. Commit code, schemas, hashes and aggregate verification evidence to GitHub.
- Use individual Kaggle accounts and bucket-scoped R2 credentials. Store secrets in platform secret stores and exclude them from logs and artifacts.
- Use unique experiment IDs and immutable raw keys. Record source commit, data version, input hashes, config, environment and output hashes.
- Develop models and loaders while ingestion progresses. Record verified evidence. Do not launch paid compute or full training without authorization.
- Publish the longitudinal viewer from reviewed GitHub code to the project owner's new Space. Preserve the historical TLOMMP Space.
