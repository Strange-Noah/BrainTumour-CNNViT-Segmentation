# Owner experiments

Install shared requirements with `python -m pip install -r research/requirements.txt`. Install each model's own GPU requirements separately. The cloud loader consumes the frozen split and the canonical manifest. Inputs have shape 4×X×Y×Z. Return a binary original-T1c-grid mask for the selected task.

- Zihan / Kaile own CNN–ViT. Port the summer model from summer-archive in a feature branch and document checkpoint compatibility, channel order and inverse geometry. The summer shape and glioma-label assumptions require adaptation.
- Yilong owns Swin UNETR. Use the selected MONAI version and native preprocessing, record pretrained checkpoint provenance and restore predictions with the inverse transform.
- Qiu owns nnU-Net v2. Create the native dataset from the shared split. Use the frozen validation/test lists for final inference rather than interpreting nnU-Net's internal cross-validation folds as the project split.

Create a Python module in your feature branch with `predict(sample)`. Its sample contains images, reference, raw_labels, target and visit metadata. Inference code must read only images and reference, without labels or target. Load your checkpoint explicitly and fail if absent. Record architecture and weights in model-version. For validation, keep the reference mask out of the predictor inputs in production adapters.

```sh
python -m research.run_visit --manifest /private/manifest.jsonl \
  --cache /private/cache --predictor my_model.predict:predict \
  --model-version MODEL-CHECKPOINT-SHA --git-sha CODE-SHA --metadata /private/run-metadata.json \
  --partition train --output /private/outputs/unique-run
```

For loader/evaluator smoke only, use `experiments.reference_smoke:predict` and model-version `reference-mask-smoke`. This echoes the provided target to check the data and export path. It is not a learned segmentation experiment. The first actual architecture runs await the respective owners' checkpoints and coordination. Full training is not launched automatically.

Runnable inference examples are provided as `experiments.cnn_vit:predict`, `experiments.swin_unetr:predict` and `experiments.nnunet:predict`. Each requires an explicitly supplied checkpoint trained for the selected binary task. The Swin training and evaluation workflow is documented in `experiments/SWINUNETR_BMETS.md`.

CNN–ViT uses `CNN_VIT_SCRIPT`, a TorchScript export whose preprocessing is embedded. Swin uses `SWIN_CHECKPOINT`, a strict MONAI 4-input/2-output state dict. Set `SWIN_FEATURE_SIZE` and `SWIN_ROI_SIZE` to match the training run. The owner records the compatible MONAI version and normalization recipe. nnU-Net uses `NNUNET_DATASET_ID`, `NNUNET_CONFIG`, `NNUNET_FOLD` and the normal nnU-Net environment paths. Its native preprocessing and inverse restoration run through nnUNetv2_predict. Install the selected framework in the owner's GPU environment and use the same research.run_visit command with the corresponding predictor.


The owners confirmed independent integration. Use enhancing tissue (raw label 3) for the first run; core (1+3) is optional secondary work. Owners choose checkpoint, framework and GPU environment when ready. Supply a private JSON file to --metadata containing checkpoint_ref (a Kaggle path, R2 key or local path), framework_version and model_config. Record native preprocessing in model_config. The run sidecar retains these fields with the exact code commit and frozen split. Keep credentials out of this file and publish aggregate metrics only in GitHub.
