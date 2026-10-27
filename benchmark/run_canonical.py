#!/usr/bin/env python3
"""Run the copied Script2Video InfiniteTalk worker with local storage adapters."""

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RUNS = ROOT / "runs"
INPUTS = ROOT / "inputs"
FACE_ENHANCE = os.environ.get("FACE_ENHANCE", "1") == "1"
LORA_RANK = os.environ.get("LORA_RANK", "64")
COMPUTE_MODE = os.environ.get("COMPUTE_MODE", "gguf")
COMPILE_MODE = os.environ.get("COMPILE_MODE", "on")
SEGMENT_ONLY = os.environ.get("SEGMENT_ONLY", "0") == "1"
FRAME_WINDOW_SIZE = int(os.environ.get("FRAME_WINDOW_SIZE", "81"))
PROFILE_TRANSFORMER = os.environ.get("PROFILE_TRANSFORMER", "0") == "1"
PROFILE_STAGES = os.environ.get("PROFILE_STAGES", "0") == "1"
PROFILE_DETAIL = os.environ.get("PROFILE_DETAIL", "0") == "1"
REUSE_INITIAL_WAN_MODEL = os.environ.get("REUSE_INITIAL_WAN_MODEL", "0") == "1"
TEACACHE_THRESHOLD = os.environ.get("TEACACHE_THRESHOLD", "")
TEACACHE_END_STEP = int(os.environ.get("TEACACHE_END_STEP", "-1"))
VAE_OPT_MODE = os.environ.get("VAE_OPT_MODE", "none")
VAE_CACHE_AUDIT = os.environ.get("VAE_CACHE_AUDIT", "0") == "1"
if VAE_OPT_MODE not in {"none", "compile_decoder", "compile_encoder", "direct_gpu_encode", "cache_identical_encode"}:
    raise ValueError("VAE_OPT_MODE must be none, compile_decoder, compile_encoder, direct_gpu_encode, or cache_identical_encode")
if TEACACHE_THRESHOLD:
    TEACACHE_THRESHOLD = float(TEACACHE_THRESHOLD)
    if not 0 < TEACACHE_THRESHOLD <= 1:
        raise ValueError("TEACACHE_THRESHOLD must be in (0, 1]")
    if TEACACHE_END_STEP not in {-1, 1, 2, 3}:
        raise ValueError("TEACACHE_END_STEP must be -1 or a step index from 1 to 3")
BENCHMARK_TAG = os.environ.get("BENCHMARK_TAG", "")
if BENCHMARK_TAG and not re.fullmatch(r"[a-z0-9_]+", BENCHMARK_TAG):
    raise ValueError("BENCHMARK_TAG must contain only lowercase letters, digits, and underscores")
SAMPLE_ID = os.environ.get("SAMPLE_ID", "")
if SAMPLE_ID and not re.fullmatch(r"[a-z0-9_]+", SAMPLE_ID):
    raise ValueError("SAMPLE_ID must contain only lowercase letters, digits, and underscores")
if PROFILE_DETAIL and not (PROFILE_STAGES and PROFILE_TRANSFORMER):
    raise ValueError("PROFILE_DETAIL requires PROFILE_STAGES=1 and PROFILE_TRANSFORMER=1")
if LORA_RANK not in {"16", "32", "64"}:
    raise ValueError("LORA_RANK must be 16, 32, or 64")
if COMPUTE_MODE not in {"gguf", "fp8", "int8"}:
    raise ValueError("COMPUTE_MODE must be gguf, fp8, or int8")
if COMPILE_MODE not in {"on", "off"}:
    raise ValueError("COMPILE_MODE must be on or off")
if FRAME_WINDOW_SIZE < 1 or (FRAME_WINDOW_SIZE - 1) % 4:
    raise ValueError("FRAME_WINDOW_SIZE must be 4n+1")
RUN_NAME = "with_face" if FACE_ENHANCE else "without_face"
if LORA_RANK != "64":
    RUN_NAME += f"_rank{LORA_RANK}"
if COMPUTE_MODE != "gguf":
    RUN_NAME += f"_{COMPUTE_MODE}"
elif PROFILE_TRANSFORMER:
    RUN_NAME += "_profile"
if PROFILE_STAGES:
    RUN_NAME += "_stages"
if PROFILE_DETAIL:
    RUN_NAME += "_detail"
if SEGMENT_ONLY:
    RUN_NAME += f"_segment_compile_{COMPILE_MODE}"
if FRAME_WINDOW_SIZE != 81:
    RUN_NAME += f"_frames{FRAME_WINDOW_SIZE}"
if BENCHMARK_TAG:
    RUN_NAME += f"_{BENCHMARK_TAG}"
if REUSE_INITIAL_WAN_MODEL:
    RUN_NAME += "_reuse_wan"
if TEACACHE_THRESHOLD:
    RUN_NAME += f"_tea{round(TEACACHE_THRESHOLD * 1000):03d}"
    if TEACACHE_END_STEP != -1:
        RUN_NAME += f"_end{TEACACHE_END_STEP}"
if VAE_OPT_MODE != "none":
    RUN_NAME += f"_vae_{VAE_OPT_MODE}"
if VAE_CACHE_AUDIT:
    RUN_NAME += "_audit"
if SAMPLE_ID:
    RUN_NAME += f"_{SAMPLE_ID}"
RUN_DIR = RUNS / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE))


class LocalInputStorage:
    def download_from_google_storage(self, source, destination, keep_format=True):
        target = Path(destination + Path(source).suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target)


class LocalOutputStorage:
    def upload_file_public(self, source, destination):
        target = RUN_DIR / "final.mp4"
        shutil.copy2(source, target)
        return str(target)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    report = {
        "worker_source": str(SOURCE / "workers/image2video_infinitetalk/server.py"),
        "worker_source_sha256": sha256(SOURCE / "workers/image2video_infinitetalk/server.py"),
        "inference_source_sha256": sha256(SOURCE / "ml_modules/comfyui_experiments_v10/infinitetalk_infer.py"),
        "environment_recipe_sha256": sha256(SOURCE / "envs/env_1_main_5090_uv.sh"),
        "face_enhance_requested": FACE_ENHANCE,
        "lora_rank": int(LORA_RANK),
        "compute_mode": COMPUTE_MODE,
        "compile_mode": COMPILE_MODE,
        "segment_only": SEGMENT_ONLY,
        "frame_window_size": FRAME_WINDOW_SIZE,
        "scope": "canonical Script2Video worker.initialize_model/run_message with local storage adapters; model download skipped because weights are already local",
        "benchmark_tag": BENCHMARK_TAG,
        "sample_id": SAMPLE_ID,
        "reuse_initial_wan_model": REUSE_INITIAL_WAN_MODEL,
        "teacache_threshold": TEACACHE_THRESHOLD or None,
        "teacache_end_step": TEACACHE_END_STEP if TEACACHE_THRESHOLD else None,
        "vae_opt_mode": VAE_OPT_MODE,
        "vae_cache_audit": VAE_CACHE_AUDIT,
    }
    if PROFILE_STAGES:
        report["stage_events"] = []
        report["subprocess_events"] = []
    try:
        import huggingface_hub
        # The worker logs in at import time; local weights make this network call unnecessary.
        huggingface_hub.login = lambda *args, **kwargs: None

        import torch
        from workers.image2video_infinitetalk.debug import messages
        from workers.image2video_infinitetalk.schema import schema_image2video_infinitetalk

        worker = importlib.import_module("workers.image2video_infinitetalk.server")
        if LORA_RANK != "64" or COMPUTE_MODE != "gguf" or PROFILE_TRANSFORMER:
            infer = importlib.import_module("ml_modules.comfyui_experiments_v10.infinitetalk_infer")
            infer.import_custom_nodes()

        if VAE_OPT_MODE != "none":
            base_vae_loader = infer.NODE_CLASS_MAPPINGS["WanVideoVAELoader"]

            class CompiledVAELoader(base_vae_loader):
                def loadmodel(self, model_name, precision, compile_args=None):
                    if compile_args is not None:
                        raise ValueError("VAE already has compile settings")
                    compile_settings = {
                        "backend": "inductor",
                        "fullgraph": False,
                        "dynamic": False,
                        "mode": "default",
                    }
                    if VAE_OPT_MODE == "compile_decoder":
                        result = super().loadmodel(model_name, precision, compile_args=compile_settings)
                    elif VAE_OPT_MODE == "compile_encoder":
                        result = super().loadmodel(model_name, precision)
                        result[0].model.encoder = torch.compile(
                            result[0].model.encoder, **compile_settings
                        )
                    elif VAE_OPT_MODE == "direct_gpu_encode":
                        result = super().loadmodel(model_name, precision)
                        vae = result[0]
                        original_encode = vae.encode

                        def direct_gpu_encode(videos, device, tiled=False, end_=False,
                                              tile_size=None, tile_stride=None,
                                              pbar=True, sample=False):
                            if tiled or end_ or not torch.is_tensor(videos) or not videos.is_cuda:
                                return original_encode(
                                    videos, device, tiled=tiled, end_=end_,
                                    tile_size=tile_size, tile_stride=tile_stride,
                                    pbar=pbar, sample=sample,
                                )
                            vae.model.clear_cache()
                            hidden_states = [
                                vae.single_encode(video.unsqueeze(0), device,
                                                  pbar=pbar, sample=sample).squeeze(0)
                                for video in videos
                            ]
                            return torch.stack(hidden_states)

                        vae.encode = direct_gpu_encode
                    else:
                        result = super().loadmodel(model_name, precision)
                        vae = result[0]
                        original_encode = vae.encode
                        cached_input = None
                        cached_latent = None
                        report["vae_encode_cache_hits"] = 0
                        report["vae_encode_cache_misses"] = 0
                        if VAE_CACHE_AUDIT:
                            report["vae_encode_cache_latent_diffs"] = []

                        def cached_encode(videos, device, tiled=False, end_=False,
                                          tile_size=None, tile_stride=None,
                                          pbar=True, sample=False):
                            nonlocal cached_input, cached_latent
                            eligible = (
                                not tiled and not end_ and not sample and
                                torch.is_tensor(videos) and videos.ndim == 5 and
                                videos.shape[2] == FRAME_WINDOW_SIZE
                            )
                            if eligible and cached_input is not None and torch.equal(videos, cached_input):
                                report["vae_encode_cache_hits"] += 1
                                if VAE_CACHE_AUDIT:
                                    fresh = original_encode(
                                        videos, device, tiled=tiled, end_=end_,
                                        tile_size=tile_size, tile_stride=tile_stride,
                                        pbar=pbar, sample=sample,
                                    )
                                    difference = (fresh - cached_latent).abs()
                                    report["vae_encode_cache_latent_diffs"].append({
                                        "max_abs": float(difference.max().item()),
                                        "mean_abs": float(difference.mean().item()),
                                        "exact": bool(torch.equal(fresh, cached_latent)),
                                    })
                                    return fresh
                                vae.model.clear_cache()
                                return cached_latent.clone()
                            result_latent = original_encode(
                                videos, device, tiled=tiled, end_=end_,
                                tile_size=tile_size, tile_stride=tile_stride,
                                pbar=pbar, sample=sample,
                            )
                            if eligible:
                                cached_input = videos.detach().clone()
                                cached_latent = result_latent.detach().clone()
                                report["vae_encode_cache_misses"] += 1
                            return result_latent

                        vae.encode = cached_encode
                    print(f"[vae variant] {VAE_OPT_MODE}", flush=True)
                    return result

            infer.NODE_CLASS_MAPPINGS["WanVideoVAELoader"] = CompiledVAELoader

        if FRAME_WINDOW_SIZE != 81:
            base_image_to_video = infer.NODE_CLASS_MAPPINGS["WanVideoImageToVideoMultiTalk"]

            class WindowSizeImageToVideo(base_image_to_video):
                def process(self, *args, **kwargs):
                    if kwargs.get("frame_window_size") != 81:
                        raise ValueError(f"Unexpected window size: {kwargs.get('frame_window_size')}")
                    kwargs["frame_window_size"] = FRAME_WINDOW_SIZE
                    print(f"[window variant] frame_window_size={FRAME_WINDOW_SIZE}", flush=True)
                    return super().process(*args, **kwargs)

            infer.NODE_CLASS_MAPPINGS["WanVideoImageToVideoMultiTalk"] = WindowSizeImageToVideo

        if LORA_RANK != "64" or COMPUTE_MODE != "gguf":
            base_selector = infer.NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]
            baseline_lora = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
            variant_lora = baseline_lora.replace("rank64", f"rank{LORA_RANK}")

            class VariantLoraSelect(base_selector):
                def getlorapath(self, *args, **kwargs):
                    if kwargs.get("lora") != baseline_lora:
                        raise ValueError(f"Unexpected LoRA selection: {kwargs.get('lora')}")
                    kwargs["lora"] = variant_lora
                    if COMPUTE_MODE in {"fp8", "int8"}:
                        kwargs["merge_loras"] = True
                    print(f"[lora variant] selected={variant_lora}", flush=True)
                    return super().getlorapath(*args, **kwargs)

            infer.NODE_CLASS_MAPPINGS["WanVideoLoraSelect"] = VariantLoraSelect
            lora_path = SOURCE / "ml_modules/comfyui_experiments_v10/models/loras" / variant_lora
            report["lora_file"] = str(lora_path)
            report["lora_sha256"] = sha256(lora_path)
        if COMPUTE_MODE in {"fp8", "int8"}:
            base_multitalk = infer.NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]
            base_wan = infer.NODE_CLASS_MAPPINGS["WanVideoModelLoader"]
            fp8_wan = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
            fp16_multitalk = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"
            load_count = 0

            class FP8MultiTalkLoader(base_multitalk):
                def loadmodel(self, *args, **kwargs):
                    if kwargs.get("model") != "Wan2_1-InfiniteTalk_Single_Q8.gguf":
                        raise ValueError(f"Unexpected InfiniteTalk model: {kwargs.get('model')}")
                    kwargs["model"] = fp16_multitalk
                    print(f"[fp8 variant] InfiniteTalk adapter={fp16_multitalk}", flush=True)
                    return super().loadmodel(*args, **kwargs)

            class QuantWanLoader(base_wan):
                def loadmodel(self, *args, **kwargs):
                    nonlocal load_count
                    load_count += 1
                    if kwargs.get("model") != "wan2.1-i2v-14b-480p-Q8_0.gguf":
                        raise ValueError(f"Unexpected Wan model: {kwargs.get('model')}")
                    kwargs["model"] = fp8_wan
                    kwargs["quantization"] = "fp8_e4m3fn_fast" if COMPUTE_MODE == "fp8" else "disabled"
                    if COMPILE_MODE == "off":
                        kwargs["compile_args"] = None
                    print(f"[{COMPUTE_MODE} variant] Wan source={fp8_wan} quantization={kwargs['quantization']}", flush=True)
                    result = super().loadmodel(*args, **kwargs)
                    if COMPUTE_MODE == "int8" and load_count > 1:
                        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
                        from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor

                        excluded = (
                            "norm", "bias", "time_in", "patch_embedding", "time_", "img_emb",
                            "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj",
                        )
                        model = result[0].model.diffusion_model
                        converted = 0
                        for parameter in model.parameters():
                            if parameter.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                parameter.data = parameter.data.to(torch.float16)
                                converted += 1
                        for buffer in model.buffers():
                            if buffer.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                buffer.data = buffer.data.to(torch.float16)
                                converted += 1
                        print(f"[int8 variant] converted {converted} FP8 tensors to FP16 before quantization", flush=True)
                        eligible = [name for name, module in model.named_modules()
                                    if isinstance(module, torch.nn.Linear)
                                    and not any(word in name for word in excluded)]
                        print(f"[int8 variant] quantizing {len(eligible)} linear layers with TorchAO W8A8", flush=True)
                        quantize_(
                            model,
                            Int8DynamicActivationInt8WeightConfig(),
                            filter_fn=lambda module, name: isinstance(module, torch.nn.Linear)
                            and not any(word in name for word in excluded),
                        )
                        # This PyTorch nightly asks tensor subclasses to dispatch this
                        # internal compatibility check during Module.to(cuda). TorchAO's
                        # quantized tensor does not implement it; replacing the module
                        # Parameter is the correct transfer behavior for this subclass.
                        original_compatible = torch._has_compatible_shallow_copy_type

                        def quantized_compatible(left, right):
                            if isinstance(left, LinearActivationQuantizedTensor) or isinstance(right, LinearActivationQuantizedTensor):
                                return False
                            return original_compatible(left, right)

                        torch._has_compatible_shallow_copy_type = quantized_compatible
                        sampler_class = infer.NODE_CLASS_MAPPINGS["WanVideoSampler"]
                        sampler_globals = sampler_class.__mro__[1].process.__globals__
                        original_offload = sampler_globals["offload_transformer"]

                        def offload_quantized_transformer(transformer):
                            # torch.compile retains weakrefs to the original INT8
                            # Parameters. Replace them before the wrapper's normal
                            # CPU offload, which otherwise tries to swap them in place.
                            for layer in transformer.modules():
                                if isinstance(layer, torch.nn.Linear) and isinstance(layer.weight, LinearActivationQuantizedTensor):
                                    layer._parameters["weight"] = torch.nn.Parameter(
                                        layer.weight.detach().to("cpu"), requires_grad=False
                                    )
                            return original_offload(transformer)

                        sampler_globals["offload_transformer"] = offload_quantized_transformer
                        report["int8_quantized_linear_count"] = len(eligible)
                        report["int8_source_fp8_tensors_converted"] = converted
                        report["int8_compute_backend"] = "TorchAO dynamic activation INT8 / weight INT8; aten::_int_mm"
                    return result

            infer.NODE_CLASS_MAPPINGS["MultiTalkModelLoader"] = FP8MultiTalkLoader
            infer.NODE_CLASS_MAPPINGS["WanVideoModelLoader"] = QuantWanLoader
            weights_root = SOURCE / "ml_modules/comfyui_experiments_v10/models/diffusion_models"
            report["wan_file"] = str(weights_root / fp8_wan)
            report["wan_sha256"] = sha256(weights_root / fp8_wan)
            report["infinitetalk_file"] = str(weights_root / fp16_multitalk)
            report["infinitetalk_sha256"] = sha256(weights_root / fp16_multitalk)

        if REUSE_INITIAL_WAN_MODEL:
            base_wan_loader = infer.NODE_CLASS_MAPPINGS["WanVideoModelLoader"]

            class ReuseInitialWanLoader(base_wan_loader):
                cached_call = None
                cached_result = None

                def loadmodel(self, *args, **kwargs):
                    normalized = dict(kwargs)
                    normalized.setdefault("block_swap_args", None)
                    previous = type(self).cached_call
                    if previous is None:
                        result = super().loadmodel(*args, **kwargs)
                        type(self).cached_call = (args, normalized)
                        type(self).cached_result = result
                        print("[wan reuse] cached initialization model", flush=True)
                        return result

                    old_args, old_kwargs = previous
                    if len(args) != len(old_args) or set(normalized) != set(old_kwargs):
                        raise ValueError("Wan reload arguments differ from initialization")
                    for old, new in zip(old_args, args):
                        if old is not new and not (isinstance(old, (str, int, float, bool, type(None))) and old == new):
                            raise ValueError("Wan reload positional argument differs from initialization")
                    for key, new in normalized.items():
                        old = old_kwargs[key]
                        if old is not new and not (isinstance(old, (str, int, float, bool, type(None))) and old == new):
                            raise ValueError(f"Wan reload argument {key} differs from initialization")
                    report["wan_model_reload_reused"] = True
                    print("[wan reuse] reused initialization model", flush=True)
                    return type(self).cached_result

            infer.NODE_CLASS_MAPPINGS["WanVideoModelLoader"] = ReuseInitialWanLoader

        if TEACACHE_THRESHOLD:
            tea_base_sampler = infer.NODE_CLASS_MAPPINGS["WanVideoSampler"]
            tea_cache_node = infer.NODE_CLASS_MAPPINGS["WanVideoTeaCache"]()
            cache_args = tea_cache_node.process(
                rel_l1_thresh=TEACACHE_THRESHOLD,
                start_step=1,
                end_step=TEACACHE_END_STEP,
                cache_device="main_device",
                use_coefficients=True,
                mode="e",
            )[0]

            class TeaCacheSampler(tea_base_sampler):
                def process(self, *args, **kwargs):
                    if "cache_args" in kwargs:
                        raise ValueError("Sampler already has cache_args")
                    kwargs["cache_args"] = cache_args
                    sampler_globals = tea_base_sampler.process.__globals__
                    original_offload = sampler_globals["offload_transformer"]
                    skipped = []

                    def capture_offload(transformer):
                        states = transformer.teacache_state.states
                        skipped.extend(step for state in states.values() for step in state["skipped_steps"])
                        return original_offload(transformer)

                    sampler_globals["offload_transformer"] = capture_offload
                    try:
                        return super().process(*args, **kwargs)
                    finally:
                        sampler_globals["offload_transformer"] = original_offload
                        report["teacache_skipped_steps"] = len(skipped)
                        report["teacache_skipped_step_indices"] = skipped
                        print(f"[teacache] skipped={len(skipped)} steps={skipped}", flush=True)

            infer.NODE_CLASS_MAPPINGS["WanVideoSampler"] = TeaCacheSampler

        if PROFILE_TRANSFORMER:
            base_sampler = infer.NODE_CLASS_MAPPINGS["WanVideoSampler"]

            class TimedSampler(base_sampler):
                def process(self, model, *args, **kwargs):
                    transformer = model.model.diffusion_model
                    original_forward = transformer.forward
                    events = []

                    def timed_forward(*forward_args, **forward_kwargs):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        result = original_forward(*forward_args, **forward_kwargs)
                        end.record()
                        events.append((start, end))
                        return result

                    transformer.forward = timed_forward
                    started = time.perf_counter()
                    try:
                        return super().process(model, *args, **kwargs)
                    finally:
                        transformer.forward = original_forward
                        torch.cuda.synchronize()
                        report["sampler_wall_seconds"] = round(time.perf_counter() - started, 3)
                        report["transformer_forward_calls"] = len(events)
                        report["transformer_cuda_seconds"] = round(
                            sum(start.elapsed_time(end) for start, end in events) / 1000, 3
                        )
                        print(
                            f"[transformer] calls={report['transformer_forward_calls']} "
                            f"cuda_seconds={report['transformer_cuda_seconds']}",
                            flush=True,
                        )

            infer.NODE_CLASS_MAPPINGS["WanVideoSampler"] = TimedSampler
        if PROFILE_STAGES:
            current_phase = {"name": "initialization"}
            def timed_call(label, method):
                def wrapped(*args, **kwargs):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        torch.cuda.synchronize()
                        report["stage_events"].append({
                            "phase": current_phase["name"],
                            "stage": label,
                            "wall_seconds": round(time.perf_counter() - started, 3),
                        })
                return wrapped

            node_methods = {
                "MultiTalkModelLoader": "loadmodel",
                "WanVideoVAELoader": "loadmodel",
                "DownloadAndLoadWav2VecModel": "loadmodel",
                "Wav2VecModelLoader": "loadmodel",
                "MelBandRoFormerModelLoader": "loadmodel",
                "WanVideoModelLoader": "loadmodel",
                "CLIPVisionLoader": "load_clip",
                "WanVideoTextEncodeCached": "process",
                "LoadAudio": "load",
                "LoadImage": "load_image",
                "ImageResizeKJv2": "resize",
                "WanVideoClipVisionEncode": "process",
                "WanVideoImageToVideoMultiTalk": "process",
                "MelBandRoFormerSampler": "process",
                "MultiTalkWav2VecEmbeds": "process",
                "WanVideoSampler": "process",
                "WanVideoPassImagesFromSamples": "decode",
                "VHS_VideoSave": "save_video_with_audio",
            }
            for node_name, method_name in node_methods.items():
                node_class = infer.NODE_CLASS_MAPPINGS[node_name]
                original_method = getattr(node_class, method_name)
                setattr(node_class, method_name, timed_call(node_name, original_method))

            worker.get_face_scale = timed_call("face_scale_gate", worker.get_face_scale)
            worker.sanitize_audio = timed_call("sanitize_audio", worker.sanitize_audio)
            worker.SimpleFaceSwapEnhancer.__init__ = timed_call(
                "face_enhancer_initialization", worker.SimpleFaceSwapEnhancer.__init__
            )

            original_subprocess_run = subprocess.run

            def timed_subprocess_run(*args, **kwargs):
                command = args[0] if args else kwargs.get("args", [])
                executable = Path(command[0]).name if isinstance(command, (list, tuple)) and command else str(command).split()[0]
                started = time.perf_counter()
                try:
                    return original_subprocess_run(*args, **kwargs)
                finally:
                    report["subprocess_events"].append({
                        "phase": current_phase["name"],
                        "executable": executable,
                        "wall_seconds": round(time.perf_counter() - started, 3),
                    })

            subprocess.run = timed_subprocess_run
            if PROFILE_DETAIL:
                sampler_base = (
                    tea_base_sampler if TEACACHE_THRESHOLD
                    else infer.NODE_CLASS_MAPPINGS["WanVideoSampler"].__mro__[1]
                )
                sampler_module = sys.modules[sampler_base.__module__]
                for method_name in ("load_weights", "compile_model", "offload_transformer"):
                    setattr(
                        sampler_module,
                        method_name,
                        timed_call(f"sampler_{method_name}", getattr(sampler_module, method_name)),
                    )
        report["gpu"] = torch.cuda.get_device_name(0)
        report["python_version"] = sys.version.split()[0]
        report["torch_version"] = torch.__version__
        report["script2video_commit"] = (ROOT / "source_commit.txt").read_text().strip()

        sample = messages[0]
        if SAMPLE_ID:
            sample = json.loads((INPUTS / "validation_samples.json").read_text())[SAMPLE_ID]
            image = INPUTS / sample["image"]
            audio = INPUTS / sample["audio"]
        else:
            image = INPUTS / "infinitetalk_test.png"
            audio = INPUTS / ("PaulVoiceShort_81frames.wav" if SEGMENT_ONLY else "PaulVoiceShort.wav")
        uid = f"local_canonical_{RUN_NAME}"
        now = datetime.now(timezone.utc).isoformat()
        message = {
            "uid": uid,
            "image_path": str(image),
            "reference_audio": str(audio),
            "video_length": 12,
            "prompt": sample["prompt"],
            "negative_prompt": schema_image2video_infinitetalk.model_fields["negative_prompt"].default,
            "seed": sample.get("seed", 8818),
            "width": 1920,
            "height": 1080,
            "fps": 25,
            "render_width": 832,
            "render_height": 480,
            "face_enhance": FACE_ENHANCE,
            "super_resolution": False,
            "audio_replace_path": "",
            "request_time": now,
            "server_start_time": now,
        }
        report["settings"] = {k: v for k, v in message.items() if k not in ("uid", "request_time", "server_start_time")}
        report["input_image_sha256"] = sha256(image)
        report["input_audio_sha256"] = sha256(audio)

        server = worker.InfiniteTalkServer.__new__(worker.InfiniteTalkServer)
        server.message = message
        server.google_cloud = LocalInputStorage()
        server.boto_cloud = LocalOutputStorage()
        server.current_time = lambda: datetime.now(timezone.utc).isoformat()
        server.download_all = lambda: str(SOURCE / "cache/RealESRGAN_x2plus.pth")

        started = time.perf_counter()
        server.initialize_model()
        report["worker_initialization_seconds"] = round(time.perf_counter() - started, 3)
        if PROFILE_STAGES:
            server.face_swap_enhancer.process_video = timed_call(
                "face_enhance_video", server.face_swap_enhancer.process_video
            )
            if PROFILE_DETAIL:
                vae = server.model_dict["wanvideovaeloader_129"][0]
                for method_name in ("encode", "decode", "to"):
                    setattr(vae, method_name, timed_call(f"vae_{method_name}", getattr(vae, method_name)))
        print(f"[canonical benchmark] initialization={report['worker_initialization_seconds']:.3f}s", flush=True)

        started = time.perf_counter()
        if PROFILE_STAGES:
            current_phase["name"] = "run_message"
            torch.cuda.reset_peak_memory_stats()
        response = server.run_message()
        report["run_message_seconds"] = round(time.perf_counter() - started, 3)
        if PROFILE_STAGES:
            report["run_message_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            )
            report["run_message_peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            )
            totals = {}
            for event in report["stage_events"]:
                key = f"{event['phase']}/{event['stage']}"
                totals[key] = round(
                    totals.get(key, 0) + event["wall_seconds"], 3
                )
            report["stage_wall_totals_seconds"] = totals
        report["face_enhance_enabled_after_gate"] = message["face_enhance"]
        report["resolved_seed"] = response["seed"]
        report["final_video"] = response["out_path"]
        report["final_video_sha256"] = sha256(response["out_path"])
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        (RUN_DIR / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(run())
