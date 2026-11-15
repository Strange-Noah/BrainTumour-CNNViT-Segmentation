# Owner experiments

Install shared requirements with `python -m pip install -r research/requirements.txt`. Install each model's own GPU requirements separately. The cloud loader consumes the frozen split and the canonical manifest. Inputs have shape 4×X×Y×Z. Return a binary original-T1c-grid mask for the selected task.

- Zihan / Kaile own CNN–ViT. Port the summer model from summer-archive in a feature branch and document checkpoint compatibility, channel order and inverse geometry. The summer shape and glioma-label assumptions require adaptation.
- Yilong owns Swin UNETR. Use the selected MONAI version and native preprocessing, record pretrained checkpoint provenance and restore predictions with the inverse transform.
- Qiu owns nnU-Net v2. Create the native dataset from the shared split. Use the frozen validation/test lists for final inference rather than interpreting nnU-Net's internal cross-validation folds as the project split.

Create a Python module in your feature branch with `predict(sample)`. Its sample contains images, reference, raw_labels, target and visit metadata. Inference code must read only images and reference, without labels or target. Load your checkpoint explicitly and fail if absent. Record architecture and weights in model-version. For validation, keep the reference mask out of the predictor inputs in production adapters.

```sh
python -m research.run_visit --manifest /private/manifest.jsonl \
  --cache /private/cache --predictor my_model.predict:predict \
  --model-version MODEL-CHECKPOINT-SHA --git-sha CODE-SHA \
  --partition train --output /private/outputs/unique-run
```

For loader/evaluator smoke only, use `experiments.reference_smoke:predict` and model-version `reference-mask-smoke`. This echoes the provided target to check the data and export path. It is not a learned segmentation experiment. The first actual architecture runs await the respective owners' checkpoints and coordination. Full training is not launched automatically.
