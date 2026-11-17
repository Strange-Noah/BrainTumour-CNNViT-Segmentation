# Preserve completed CPU preparation without rerunning it

On 2026-11-17 the owner confirmed that the interactive CPU preparation completed successfully. Agent inspection of the official Kaggle API found current version **7**, private, GPU disabled, script source containing only the availability preflight, and an empty output file list (both latest and explicit /7). The API-visible version is therefore not evidence of saved prepared files. This does not establish whether files remain in the live interactive session; browser control timed out, so that filesystem was not independently inspected. [Sanitized metadata evidence](evidence/cpu-persistence-inspection.json).

## Minimal preservation steps

1. Keep the existing authenticated **CPU** session open. Confirm `/kaggle/working/novo-unet25d` still contains `prepared/`, `hf/`, `config.json`, `cache-receipts.json` and `cache_integrity.stage.json`. The last record should say `status: complete`. Do not rerun the preparation cell.
2. Keep the notebook **private**. Open **Save Version**, select **Quick Save**, and under **Advanced Settings** enable **Save output files** (wording may vary). This saves the existing `/kaggle/working` outputs; do not choose Save & Run All. Retain the prepared directory and its offline model cache. Wait for the output upload to finish before stopping the session.
3. Open the saved version's Output/Files area and confirm the prepared directory and three receipt/config records appear. Send only the saved version number/link. No separate public dataset, new credentials, or GPU allocation is needed.

Kaggle documents notebook versions as snapshots and supports saving current output files through Quick Save: [notebook documentation](https://www.kaggle.com/docs/notebooks), [Kaggle Advanced Save Options announcement](https://www.kaggle.com/discussions/product-feedback/161328). Saving notebook code or displayed text alone does not establish file persistence. If your UI does not offer output saving or the upload reports a limit/error, keep the CPU session alive and report that exact option/error; do not rerun preparation to work around it.

## Agent verification after saving

List the exact version's files through the official API. Mount that private notebook output as input to a separate private CPU kernel. Run `python -m experiments.unet25d.verify_mount /kaggle/input/<actual-output-mount>/novo-unet25d` from the pinned project source. This streams all receipt hashes, validates configuration, verifies the eight training/one validation visit indexes and the pinned offline encoder weight cache, with no R2 secret lookup and no GPU. Actual mount paths must be discovered, not assumed. Only after that check passes may the authorized bounded GPU campaign be launched. Record CPU source SHA separately from any reviewed GPU safety fixes; configuration and prepared-file receipts must agree.

Current GPU allocation/session/device time remains zero. Full training is not authorized. Prepared medical data stays private; no imaging or credential values are included in this report.

## R2 fallback — 2026-11-18

Quick Save remains queued according to the owner. Use `persist_cache.py upload /kaggle/working/novo-unet25d` in the existing authenticated CPU session. This performs **no preprocessing, source imaging downloads, GPU imports or training**. The dataset bucket is never opened; only the private output bucket is used.

The uploader retains the original `launch.json` source SHA and unchanged configuration bytes. Its allowlist is: prepared files explicitly named by existing integrity receipts; those receipts; configuration; completed integrity-stage record; original launch provenance and canonical manifest; the pinned encoder snapshot's config and safetensors. It excludes raw source cache, unrelated working files, environment dumps, tokens and logs. Original receipts and pretrained/dataset hashes must pass before transfer starts.

Each transfer gets a UUID prefix `novo-nordisk/experiments/unet25d/prepared-cache/<cache_id>/` in `novo-nordisk-pbl-outputs`. The UUID and inventory fingerprint are saved locally in `.r2-cache-transfer.json`. Files become content-addressed 32 MiB chunks; each chunk is create-once and GET/SHA-256 checked. Restart the same cell to resume: matching chunks are readback-verified and not uploaded again; conflicts or changed source files fail instead of overwriting. Only one chunk is buffered, never an entire volume. SDK retries are bounded. Keep the source directory and transfer state intact until verification completes.

The completion manifest is written after all files pass. A new Python process then uses only the remote manifest and R2 objects, reads every chunk and recomputes each whole-file hash, without consulting local cache files. It writes `independent-verification.json` privately only after success. This is independent process/readback verification using the existing output-scoped credentials, not a separately credentialed reader. Interrupted or failed verification never authorizes GPU allocation.

The final console JSON contains only verification status, cache ID, file/byte counts and hashes. Share that receipt, not data or credentials. The local `r2-cache-upload.json` also retains the private prefix and manifest hash for recovery. The chunked R2 cache will need CPU-side reconstruction before it can serve as a mounted file cache; uploading alone does not establish Kaggle mountability. Do not allocate GPUs for cache restoration.

Direct R2 is preferred because the authenticated session already has its output writer. Private Kaggle Dataset creation would add a separate publication/authentication step. Neither option requires preprocessing again; this fallback avoids the stuck Quick Save entirely. Offline tests cover repeated-upload reuse, refusal to overwrite conflicts, and remote corruption detection. Live upload and fresh-process readback remain pending execution in the owner's authenticated CPU session. No credentials or cache objects are committed to GitHub.
