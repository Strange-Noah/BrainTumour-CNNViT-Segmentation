# Controlled queue replacement — cancellation blocked

[Original private job](https://www.kaggle.com/code/mathewzhou/novo-unet25d-v2-44906471093-s1) remained CLI `QUEUED` at this operation's initial fresh read (2026-11-21 17:17 UTC / 2026-11-22 01:17 BJT). No replacement GPU was submitted and no existing notebook/output was deleted. Gate1 and actual formal updates remain unobserved.

## Exactly one refreshed private input

[CPU run 44929363445](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44929363445) succeeded using the existing trusted main dispatcher, registered immutable runtime `3ddc324e114d0a0cd740234b5ba26ec750238d2a` and existing private credentials. It ran 18 passing CPU tests, emitted `v2_gate0_reused` and published `mathewzhou/novo-unet25d-v2-input` **version 3** with `private_input_verified=true`. No MRI preprocessing/ingestion or GPU allocation occurred.

- Lease expires **2026-11-22 11:19:54 UTC / 19:19:54 BJT**.
- Source and science unchanged: science `1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110`.
- Catalogue SHA `04b59724c5cc6ac654f85d51ad19e8a8a0766944d8211ecfd03a3495a418de1e`.
- Plan file SHA `4029aee0a723fce9794ee2ce5ee05e61f7c357656029bb55f8d0ec02d87d8860` (file-byte SHA; distinguish from the canonical plan digest in SAMPLING_AUDIT).
- Local independent metadata readback verified lease/config bytes against the exact v3 package receipt. Compared v2/v3 package receipts: catalogue, config, plan, inventory and sampling audit file SHAs all identical; source/science identical. Existing publisher independently read back the exact package, config, full plan and inventory.
- Existing 220,384 mandatory / 73,462 replay positions preserved. Only private lease/package-publication metadata refreshed.

Publishing version 3 is NOT claimed to change the old queued notebook's mounted input. Provider input binding remains unverified. No stream of refresh versions is authorized or scheduled.

## Concrete cancellation blocker

Installed official Kaggle SDK exposes `cancel_kernel_session(ApiCancelKernelSessionRequest)` requiring an integer **kernel_session_id**. It does not accept owner/slug. The existing public GetKernel response exposes metadata/source and the status response does not expose a cancellable session identity. Raw response-field inspection confirmed no session-ID field. Notebook metadata ID must not be substituted for a session ID. No guessed cancellation request was issued.

Kaggle's [public CLI issue #1169](https://github.com/Kaggle/kaggle-cli/issues/1169) documents this API boundary. There is no verified `kaggle kernels cancel` CLI command available here. No authenticated browser-control tool is available in this runtime. Therefore safe termination/non-overlap cannot currently be established autonomously.

Minimum unblock: owner cancels only the original queued execution through Kaggle Active Events, preserving notebook/history, or supplies that execution's actual kernel_session_id. Agent must independently recheck provider terminal status before any replacement. If the original starts RUNNING first, abandon replacement and observe the original.

## Recovery status and next operation

The existing active record/recovery routing was not redirected or disabled; it still refers to the original job. No new source pin or scientific setting was introduced. Automatic recovery capability is configured, but no successful v2 checkpoint recovery is claimed. There is no trained v2 checkpoint or Gate1 result to archive yet.

After verified termination, use the owner's already authorized single replacement only: recheck v3 remaining validity, quota and cumulative reservation; preserve the old evidence and reconcile its unused reservation without resetting the original campaign budget; bind a distinct v2 notebook to the verified fresh input; update the existing active record before unattended operation. This routing transition is pending, not claimed complete. Never bypass campaign reservation/source authorization or start a second job while old state is nonterminal/ambiguous.

GPU submissions in this operation: **0**. Input refreshes: **1**. No paid resources, v1 changes, teammate changes or permanent branch added.
