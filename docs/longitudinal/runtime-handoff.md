# Current runtime handoff after promotion

Read [final governance verification](final-governance.md) before changing repository settings. The approved longitudinal project is on main. Branch new segmentation, tissue segmentation evaluation, lesion correspondence and viewer work from main. Use PRs with passing validate CI, then owner-authorized merge. Summer code is recoverable on the frozen summer-archive branch.

The manual TCIA pilot runs from main. Both credential environments now allow only main. Confirm the run SHA, use the ingestion environment's bucket-scoped writer for the first job and the verification environment's reader for the independent second runner. Existing environment owner approval and self-review restrictions mean a trusted collaborator dispatches and the owner approves the credential jobs. Set TCIA_PILOT_ENABLED only when ready. Resume steps and exact command details are in the [runtime runbook](handoff.md).

The pilot must persist all 43 source-size-matched files, verify their hashes against the original Aspera pilot and write the completion receipt. Confirm the second job hashes every persisted R2 object. The pilot's known inventory and byte digests are embedded in cloud/tcia_pilot.py. Keep Kaggle dataset credentials read-only.

The stable output bucket is novo-nordisk-pbl-outputs. The Kaggle CPU smoke uses its separate R2_OUTPUT credentials. Verify both output NIfTI and report in a fresh session and record the pinned source SHA and hashes.

Full ingestion awaits the complete per-file release inventory and persistent pilot evidence. Use bounded patient batches and immutable completion receipts. Publisher checksum availability and interrupted-transfer recovery still require validation. R2 costs and staging conventions are recorded in the [acquisition audit](tcia.md).

Hugging Face deployment remains deferred pending eligible plan and authorization. No paid compute or full model training is authorized.
