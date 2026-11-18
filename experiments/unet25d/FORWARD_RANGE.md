# Forward-range diagnosis and continuation

[Second GPU run 44241159112](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44241159112) used integrated source `ace4e237c5e160c15efd024e8e65c6406f713398` and the already verified private cache. It stopped on **nonfinite forward loss**, with no gradient-overflow skips. Short single/DDP stages ran, but sustained single/DDP, evaluation and resume acceptance remain open. No full training was launched.

## Confirmed CPU cloud diagnosis

[Private diagnostic notebook](https://www.kaggle.com/code/mathewzhou/novo-unet25d-forward-diagnostic) versions 2 and 3 completed without a GPU or optimizer updates. They reused the failing private checkpoint and exact next prepared draw. Model weights and AdamW states were finite; 20 decoder BN running-stat buffers were nonfinite. Diagnostic forward passes reset decoder running statistics; training uses batch statistics. No damaged checkpoint is an accepted recovery checkpoint.

The input was finite (maximum absolute value 6.620194). FP32 forward loss was 0.32153696. Under CPU FP16 autocast, the first nonfinite output was `decoder.blocks.0.conv1.0`. Its FP32 activation reached 67,976, exceeding FP16 maximum 65,504. This establishes a forward-range failure, which lowering GradScaler cannot fix.

Computing only `decoder.blocks.0.conv1` (Conv/BN/ReLU) in FP32 made the same draw's forward and scaled backward finite: loss 0.32153288 and maximum finite activation 67,962.67. The rest remains under FP16 autocast. This is CPU evidence for the correction, **not** sustained CUDA acceptance. [Aggregate diagnostic evidence and original receipt hashes](evidence/forward-range.json). Nonfinite JSON numbers are represented as null; identifiers and images are excluded.

## Focused implementation

The precision boundary changes no parameters, state-dict keys, architecture, learning rates, sampling or preprocessing. Checkpoints record the runtime precision, and both GPU resume and fresh CPU recovery reject a different precision contract. Restart this bounded benchmark from the official pretrained initialization rather than the failed diagnostic checkpoint.

PR #22 also omitted incrementing the reported successful-update counter. Actual AdamW checkpoint state held 1,010 updates while metadata reported zero; draw position and throughput sample counts were recorded independently. The correction increments only successful updates and checks optimizer/metadata agreement before saving. Existing failed checkpoints are unsuitable for recovery acceptance.

Six targeted tests passed: actual GradScaler skip/recovery, fatal guards, AdamW counter agreement through a skipped update, identical model state/parameter contracts and autocast boundary, resumable transfer, and remote corruption detection. Existing geometry/gradient/BN CPU contracts also passed. These tests do not establish GPU stability.

## Budget and remaining acceptance

Observed account quota after the two attempts: 0.24 hours used / 29.76 remaining. Account quota accounting differs from aggregate device-hours. The next dispatch must use **gpu_session_seconds=2400**, within the original 60-minute campaign envelope; this is not a new budget. Reuse private dataset version 1; never prepare or publish the cache again.

After this precise source change is reviewed and integrated into main, run the existing manual benchmark workflow. Require >=600 steady seconds for single and DDP, RAM stability, loader measurements, evaluation, fresh-process continuation, and private R2 output/readback. Stop after the benchmark. No paid compute or full-cohort training is authorized.
