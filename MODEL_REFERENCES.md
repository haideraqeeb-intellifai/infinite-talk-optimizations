# Model references

No model parameters, checkpoints, adapters, or optimizer states are included in this repository or Git LFS.

The optimized path selects these two additional files. Download them from their model publisher and place both in `ml_modules/comfyui_experiments_v10/models/diffusion_models/`. Confirm the SHA-256 before use. The hashes match the files used in the benchmark.

| File and purpose | Source | Size | SHA-256 |
| --- | --- | ---: | --- |
| `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` — Wan video transformer | [Kijai/WanVideo_comfy, revision 89ee5a2](https://huggingface.co/Kijai/WanVideo_comfy/blob/89ee5a220d9f24199fe5ad553e34a6ec6fd453d4/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors) | 16,993,877,896 bytes | `996dbad030df09b0b3c8e764f0fb5a81b98b220ab89524d6a9369e9ed882791f` |
| `Wan2_1-InfiniTetalk-Single_fp16.safetensors` — InfiniteTalk adapter | [Kijai/WanVideo_comfy, InfiniteTalk directory](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors) | 5,125,258,232 bytes | `bf212c5c647eaf1c019b2158d2edf4a360d97e38b746c4bac994fee485e09fc3` |

The FP16 adapter link tracks `main`; use the SHA-256 to select the tested bytes. Review the publisher's model card for current license and access terms before downloading. The existing Script2Video deployment also requires its unchanged LoRA, VAE, CLIP vision, text encoder, Wav2Vec, MelBandRoformer, FaceFusion, and upscaler files. Their model setup remains in Script2Video; this package changes neither their selection nor their storage. The selected benchmark files and their hashes are recorded in `benchmark/evidence/optimized_timings.json`.
