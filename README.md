# InfiniteTalk optimization integration

This is a focused integration package for the Script2Video `image2video_infinitetalk` worker at source revision `c3a429582a587ddfc714de6986071e8b7b3a6376`. It contains the modified inference module, the `main_5090` environment recipe, the original benchmark runner, and selected measurement records. It does not contain the full Script2Video application.

## Optimized path

Set `INFINITETALK_OPTIMIZED=1` in the worker process before model initialization. The flag selects all five code changes below; the environment recipe supplies the sixth dependency change.

| Change | Implementation |
| --- | --- |
| FP8 Wan and FP16 InfiniteTalk adapter | `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` with `fp8_e4m3fn_fast`; `Wan2_1-InfiniTetalk-Single_fp16.safetensors`; LoRA merge enabled |
| 49-frame windows | `frame_window_size=49`, retaining the 9-frame motion overlap |
| SageAttention 2.2 | `integration/env_1_main_5090_uv.sh` builds `sageattention==2.2.0` against the recipe's PyTorch/CUDA stack |
| Wan model reuse | The initialized model is reused only when the request's loader arguments match; otherwise it reloads |
| TeaCache | Relative L1 threshold `0.26`, steps `1..2`, main-device cache, coefficients enabled, mode `e` |
| Identical padded VAE encode cache | One input/latent pair per request, exact `torch.equal` check, VAE internal-cache clear and cloned latent on a hit |

With the flag unset, the inference module uses the original Q8 model and 81-frame path. The supplied environment recipe still installs SageAttention 2.2, so use the original recipe when reproducing the full original baseline.

## Apply to Script2Video

From a checkout of the source revision above, back up the two target files and copy these integration files to:

```text
integration/infinitetalk_infer.py  -> ml_modules/comfyui_experiments_v10/infinitetalk_infer.py
integration/env_1_main_5090_uv.sh -> envs/env_1_main_5090_uv.sh
```

Build or update `main_5090` with that recipe. [SageAttention's installation guidance](https://github.com/thu-ml/SageAttention/blob/main/README.md) requires CUDA 12.8 or newer for Blackwell; the benchmark used an RTX 5090, CUDA 12.8, and PyTorch `2.10.0.dev20250923+cu128`. Put the two selected files from [MODEL_REFERENCES.md](MODEL_REFERENCES.md) into the existing ComfyUI `models/diffusion_models/` directory. Keep the deployment's other existing model files in their current locations. Then set `INFINITETALK_OPTIMIZED=1` before starting the worker.

The worker request and response schema is unchanged. The flag is a process setting, not an API request parameter. The benchmark validated the Wan reuse path for one request per worker instance at 832×480; consecutive requests and changed loader settings need runtime validation before production rollout.

## Benchmark and checks

The original `benchmark/run_canonical.py` uses a local Script2Video checkout in `benchmark/source/` and local input files in `benchmark/inputs/`. It is included to show the tested guarded overrides; the actual integration code is in `integration/`. The source checkout, human image/audio input, output videos, and installed environments are not distributed here. The reports in `benchmark/evidence/` retain their original local artifact paths as provenance; the [FP8 report](https://s3.renderplatform.com/agent-assets/Raqeeb/report_fp8_vs_gguf/index.html) and [window-size report](https://s3.renderplatform.com/agent-assets/Raqeeb/report_fp8_window49/index.html) include viewable video comparisons.

The selected single-run results measured 267.065 seconds for the original Q8 path and 158.016 seconds with all six optimizations for the 463-frame sample. The VAE cache produced byte-identical final videos in four tested samples relative to the same configuration without that cache. FP8, 49-frame windows, SageAttention 2.2, and TeaCache can change the video; inspect mouth motion in addition to pixel metrics. These are benchmark results, not a measurement of the integrated deployment file.

Run static and focused cache checks with:

```bash
python -m py_compile integration/infinitetalk_infer.py benchmark/run_canonical.py
bash -n integration/env_1_main_5090_uv.sh
python -m unittest discover -s tests -v
```

The unit tests need PyTorch. End-to-end generation requires the Script2Video source, model weights, compatible GPU environment, and input media. No end-to-end generation of the integrated file has been run in this package.

## Publication scope

This package excludes model weights, installed Python environments, caches, unrelated Script2Video modules, and benchmark media. One copied upstream worker file contains an embedded credential, so the full copied source tree is deliberately excluded. Replace or revoke that credential in the upstream deployment separately. Third-party dependencies and model licenses remain governed by their respective projects; no new license is asserted for the Script2Video-derived code here.
