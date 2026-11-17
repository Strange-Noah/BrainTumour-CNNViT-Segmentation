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
