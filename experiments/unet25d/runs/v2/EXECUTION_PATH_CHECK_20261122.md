# Phase1 continuation: execution-path check

2026-11-22. Owner additionally authorized trying alternative execution methods. Scientific source remains `9c13b991de9231277270dd115b2fa05248cac18d`; science, checkpoint and approved free budget remain unchanged.

## Confirmed state

The unique continuation is [44950328254-s2](https://www.kaggle.com/code/mathewzhou/novo-unet25d-v2-44950328254-s2), submitted by [successful CPU recovery44999857941](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44999857941). Provider CLI still reports QUEUED. No resumed optimizer updates or new CUDA allocation are established. Original s1 remains ERROR. This is not a completed Phase1 or a ten-minute health acceptance.

Independent v6 private package and lease checks bind source9c13b99, science61504c0 and checkpoint SHA `e76428bec9e1d43e8f851fe0703b5c330125ee53c9a710c499d4d2fbf03e380a` (step46,784). Lease expires2026-11-23 09:59:33 BJT. A12h session requires allocation before roughly2026-11-22 21:54 BJT with the runtime's extra300s startup validity check; publishing a new version alone would not repair a pinned queued input.

Official SDK account read: GPU used24,931.598s, reserved0s, total108,000s; paid scaling disabled; refresh2026-11-28. No evidence of hidden running GPU reservation or exhausted weekly quota. This does not establish provider capacity or prove why scheduling waits.

## Actual alternative-path checks

- CLI: supported genuine notebook Save & Run All, private, GPU enabled, NvidiaTeslaT4, immutable source independently pulled. Current call is a successful submission, not allocation.
- Official [Kaggle MCP](https://www.kaggle.com/docs/mcp): authenticated HTTP200 tools discovery and actual read-only notebook-info/status calls succeeded. The status tool returned an empty default response; CLI remains the actual QUEUED evidence. Notebook metadata confirms private notebook, GPU/internet enabled and NvidiaTeslaT4. `save_notebook` maps to the same save-and-run request family as CLI, so changing the transport alone is not demonstrated to change scheduling.
- MCP and installed official SDK expose `create_notebook_session` / `create_kernel_session`. This creates an interactive session and returns an Operation; it does not document automatic execution of the notebook's cells. The [official Jupyter Server route](https://www.kaggle.com/docs/notebooks) supports external clients once an actual connection exists. No connection credential or executable interactive session was obtained, and no extra GPU was allocated merely to speculate about the Operation response.
- Safe cancellation remains unresolved: documented cancellation requires the actual kernel-session ID. Public notebook/status responses did not supply it. Notebook ID146590272 and local version number1 are not substituted for an execution ID. The authorized private web-page request returned404 and no explicit session ID.
- Existing browser route was tested, without reading cookies or changing browser security: Chrome version responded, but window control returned macOS Apple Events permission error `-1743`. No browser control permission was granted or bypassed, and no manual operation was requested.

## Safety and next automatic behavior

No alternative GPU submission, cancellation, deletion, accelerator/precision/batch change, restricted priority override or paid request was issued. Starting another route before safely controlling the queued execution could produce duplicate GPU consumption and confuse checkpoint ownership; no such outcome is claimed safe.

The existing trusted-main collector remains enabled for the exact registered source. On allocation, the submitted code will check identities and resume the passed-Gate1 checkpoint directly, without waiting for another owner message. Observe actual updates, then independently recover a new formal checkpoint before claiming resumed unattended health. On genuine ERROR, preserve evidence and stop rather than launch speculative retries. Existing Phase1 endpoint remains AWAIT_PHASE2_DESIGN; Phase2/test access remain unauthorized.

[Repair, checksums, CPU proof and early scientific limitations](PHASE1_TRANSPORT_RECOVERY.md). [Existing two-exam diagnostic aggregate](EARLY_DIAGNOSTIC_AGGREGATE.json) is poor and is not full-cohort validation.
