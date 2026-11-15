# Experiment Protocol v1

All CNN–ViT (Zihan / Kaile), Swin UNETR (Yilong) and nnU-Net v2 (Qiu) runs use the attached 300-patient mapping, 210 train / 45 validation / 45 test, stratified by visit count. The exact digest and allocation are in split.json. Each visit is independent for segmentation. All visits from a patient share a partition.

Inputs are T1, T1c, T2, FLAIR in that channel order. Use the complete-four-modality cohort, the same original scans and eligibility for all models. The shared loader linearly resamples nonmatching MRI grids to T1c and rejects misaligned source labels. Model-specific pipelines may instead load cached originals using Dataset.fetch, with declared geometry restoration.

Primary task is enhancing tissue (raw label 3). Secondary is tumor core (raw 1 or 3). Keep source labels intact. Native normalization, cropping, patching, augmentation and resampling are allowed. No ground-truth-derived inference crops and no fitting preprocessing on held-out data.

Shared evaluator conventions are 26-connected components, no size filtering, one-to-one maximum-cardinality lesion matching at positive voxel overlap. HD95 uses the maximum of both directed 95th percentile nearest-surface distances after full affine transformation into mm. Two empty masks have Dice 1, HD95 0. One empty mask has Dice 0, undefined HD95 (JSON null). No truth lesions yields undefined sensitivity. Predicted unmatched components are false positives per scan. Signed and absolute volume error use determinant-based mm³. Aggregate defined visit metrics per patient, retaining defined counts, then report macro patient means. Never discard undefined counts silently.

Export binary task masks with prediction 1 meaning the declared task and 0 background, restored to original T1c. Sidecars retain canonical patient/exam, model version, source commit, split digest and raw mapping. Candidate IDs are visit-local connected components. Centroids are physical world mm, volumes are mm³. Longitudinal identities require tracking work.

Install research/requirements.txt. Model owners implement their own predictor taking a 4×X×Y×Z array and reference geometry and returning an original-grid binary mask. Use research.evaluate.export for handoff and research.evaluate.evaluate for shared metrics. Export one visit at a time. Full training requires coordination with model owners.
