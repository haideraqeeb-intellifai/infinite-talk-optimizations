import os
import random
import sys
from turtle import mode
from typing import Sequence, Mapping, Any, Union
# from ComfyUI.comfy.model_management import load_model_gpu
import torch
current_dir = os.path.dirname(__file__)
# Opt in per worker process; the original Q8 path remains available.
OPTIMIZED = os.environ.get("INFINITETALK_OPTIMIZED", "0") == "1"
FRAME_WINDOW_SIZE = 49 if OPTIMIZED else 81
WAN_MODEL = (
    "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" if OPTIMIZED
    else "wan2.1-i2v-14b-480p-Q8_0.gguf"
)
MULTITALK_MODEL = (
    "Wan2_1-InfiniTetalk-Single_fp16.safetensors" if OPTIMIZED
    else "Wan2_1-InfiniteTalk_Single_Q8.gguf"
)
WAN_QUANTIZATION = "fp8_e4m3fn_fast" if OPTIMIZED else "disabled"


def enable_request_vae_cache(model_dict):
    """Reset the one-entry padded encode cache for each request."""
    vae = get_value_at_index(model_dict["wanvideovaeloader_129"], 0)
    original_encode = model_dict.setdefault("_original_vae_encode", vae.encode)
    cached_input = None
    cached_latent = None

    def cached_encode(videos, device, tiled=False, end_=False, tile_size=None,
                      tile_stride=None, pbar=True, sample=False):
        nonlocal cached_input, cached_latent
        eligible = (
            not tiled and not end_ and not sample and torch.is_tensor(videos)
            and videos.ndim == 5 and videos.shape[2] == FRAME_WINDOW_SIZE
        )
        if eligible and cached_input is not None and torch.equal(videos, cached_input):
            # The original encode clears the VAE's internal cache on every call.
            vae.model.clear_cache()
            return cached_latent.clone()
        latent = original_encode(
            videos, device, tiled=tiled, end_=end_, tile_size=tile_size,
            tile_stride=tile_stride, pbar=pbar, sample=sample,
        )
        if eligible:
            cached_input = videos.detach().clone()
            cached_latent = latent.detach().clone()
        return latent

    vae.encode = cached_encode

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    utils_dir = os.path.join(current_dir, "utils")
    sys.path.append(utils_dir)
    from extra_config import load_extra_path_config
    

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

add_comfyui_directory_to_sys_path()
add_extra_model_paths()


_NODES_IMPORTED = False
def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    global _NODES_IMPORTED
    if _NODES_IMPORTED: return
    import asyncio
    import execution
    from nodes import init_extra_nodes

    # sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())
    _NODES_IMPORTED = True


from nodes import NODE_CLASS_MAPPINGS

def initialise_model():
    import_custom_nodes()
    model_dict = {}
    with torch.inference_mode():
        multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        model_dict["multitalkmodelloader_120"] = multitalkmodelloader.loadmodel(
            model=MULTITALK_MODEL
        )

        # loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        # loadaudio_125 = loadaudio.load(audio="1472-a5e4-4265-9cb9-47e59d608fbe.mp3")

        wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        model_dict["wanvideovaeloader_129"] = wanvideovaeloader.loadmodel(
            model_name="Wan2_1_VAE_bf16.safetensors", precision="bf16"
        )

        wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        model_dict["wanvideoblockswap_134"] = wanvideoblockswap.setargs(
            blocks_to_swap=20,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=1,
            block_swap_debug=False,
        )

        downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadWav2VecModel"
        ]()
        model_dict["downloadandloadwav2vecmodel_137"] = downloadandloadwav2vecmodel.loadmodel(
            model="TencentGameMate/chinese-wav2vec2-base",
            base_precision="fp16",
            load_device="main_device",
        )

        wanvideoloraselect = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
        model_dict["wanvideoloraselect_138"] = wanvideoloraselect.getlorapath(
            lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            strength=1,
            low_mem_load=False,
            merge_loras=OPTIMIZED,
            unique_id=16103358818665079374,
        )

        wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS[
            "WanVideoTorchCompileSettings"
        ]()
        model_dict["wanvideotorchcompilesettings_177"] = wanvideotorchcompilesettings.set_args(
            backend="inductor",
            fullgraph=False,
            mode="default",
            dynamic=False,
            dynamo_cache_size_limit=64,
            compile_transformer_blocks_only=True,
            dynamo_recompile_limit=128,
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        model_dict["clipvisionloader_238"] = clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        # wanvideotextencodecached = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
        # model_dict["wanvideotextencodecached_241"] = wanvideotextencodecached.process(
        #     model_name="umt5-xxl-enc-bf16.safetensors",
        #     precision="bf16",
        #     positive_prompt="a woman is singing",
        #     negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        #     quantization="disabled",
        #     use_disk_cache=False,
        #     device="gpu",
        # )
        
        wav2vecmodelloader = NODE_CLASS_MAPPINGS["Wav2VecModelLoader"]()
        model_dict["wav2vecmodelloader_300"] = wav2vecmodelloader.loadmodel(
            model="wav2vec2-chinese-base_fp16.safetensors",
            base_precision="fp16",
            load_device="main_device",
        )

        melbandroformermodelloader = NODE_CLASS_MAPPINGS["MelBandRoFormerModelLoader"]()
        model_dict["melbandroformermodelloader_301"] = melbandroformermodelloader.loadmodel(
            model_name="MelBandRoformer_fp16.safetensors"
        )
        print("reloading model...once")
        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        wan_loader_args = dict(
            model=WAN_MODEL,
            base_precision="fp16_fast",
            quantization=WAN_QUANTIZATION,
            load_device="offload_device",
            attention_mode="sageattn",
            compile_args=get_value_at_index(model_dict["wanvideotorchcompilesettings_177"], 0),
            lora=get_value_at_index(model_dict["wanvideoloraselect_138"], 0),
            multitalk_model=get_value_at_index(model_dict["multitalkmodelloader_120"], 0),
            block_swap_args=None,
        )
        model_dict["wanvideomodelloader_122"] = wanvideomodelloader.loadmodel(**wan_loader_args)
        model_dict["_wan_loader_args"] = wan_loader_args
            
        return model_dict
    
prompt = "a woman is singing"
negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def main(model_dict, audio_path, image_path, save_path = "output.mp4", prompt =  prompt, negative_prompt = negative_prompt, width = 832, height = 480, num_frames = 2000, fps = 25, seed = -1, upsampler = None):
    import_custom_nodes()
    if OPTIMIZED:
        model_dict["_optimized_request_count"] = model_dict.get("_optimized_request_count", 0) + 1
        enable_request_vae_cache(model_dict)
    with torch.inference_mode():
        # multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        # multitalkmodelloader_120 = multitalkmodelloader.loadmodel(
        #     model="Wan2_1-InfiniteTalk_Single_Q8.gguf"
        # )

        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        loadaudio_125 = loadaudio.load(audio=audio_path)

        # wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        # wanvideovaeloader_129 = wanvideovaeloader.loadmodel(
        #     model_name="Wan2_1_VAE_bf16.safetensors", precision="bf16"
        # )

        # wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        # wanvideoblockswap_134 = wanvideoblockswap.setargs(
        #     blocks_to_swap=20,
        #     offload_img_emb=False,
        #     offload_txt_emb=False,
        #     use_non_blocking=True,
        #     vace_blocks_to_swap=0,
        #     prefetch_blocks=1,
        #     block_swap_debug=False,
        # )

        # downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS[
        #     "DownloadAndLoadWav2VecModel"
        # ]()
        # downloadandloadwav2vecmodel_137 = downloadandloadwav2vecmodel.loadmodel(
        #     model="TencentGameMate/chinese-wav2vec2-base",
        #     base_precision="fp16",
        #     load_device="main_device",
        # )

        # wanvideoloraselect = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
        # wanvideoloraselect_138 = wanvideoloraselect.getlorapath(
        #     lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        #     strength=1,
        #     low_mem_load=False,
        #     merge_loras=False,
        #     unique_id=16103358818665079374,
        # )

        # wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS[
        #     "WanVideoTorchCompileSettings"
        # ]()
        # wanvideotorchcompilesettings_177 = wanvideotorchcompilesettings.set_args(
        #     backend="inductor",
        #     fullgraph=False,
        #     mode="default",
        #     dynamic=False,
        #     dynamo_cache_size_limit=64,
        #     compile_transformer_blocks_only=True,
        #     dynamo_recompile_limit=128,
        # )

        # clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        # clipvisionloader_238 = clipvisionloader.load_clip(
        #     clip_name="clip_vision_h.safetensors"
        # )

        wanvideotextencodecached = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
        wanvideotextencodecached_241 = wanvideotextencodecached.process(
            model_name="umt5-xxl-enc-bf16.safetensors",
            precision="bf16",
            positive_prompt= prompt,
            negative_prompt= negative_prompt, 
            quantization="disabled",
            use_disk_cache=False,
            device="gpu",
        )

        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        # intconstant_245 = intconstant.get_value(value=832)
        intconstant_245 = intconstant.get_value(value=width)

        # intconstant_246 = intconstant.get_value(value=480)
        intconstant_246 = intconstant.get_value(value=height)

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_284 = loadimage.load_image(
            image=image_path
        )

        # wav2vecmodelloader = NODE_CLASS_MAPPINGS["Wav2VecModelLoader"]()
        # wav2vecmodelloader_300 = wav2vecmodelloader.loadmodel(
        #     model="wav2vec2-chinese-base_fp16.safetensors",
        #     base_precision="fp16",
        #     load_device="main_device",
        # )

        # melbandroformermodelloader = NODE_CLASS_MAPPINGS["MelBandRoFormerModelLoader"]()
        # melbandroformermodelloader_301 = melbandroformermodelloader.loadmodel(
        #     model_name="MelBandRoformer_fp16.safetensors"
        # )

        # wanvideocontextoptions = NODE_CLASS_MAPPINGS["WanVideoContextOptions"]()
        # wanvideocontextoptions_311 = wanvideocontextoptions.process(
        #     context_schedule="uniform_standard",
        #     context_frames=81,
        #     context_stride=4,
        #     context_overlap=16,
        #     freenoise=True,
        #     verbose=False,
        #     fuse_method="linear",
        # )

        intconstant_312 = intconstant.get_value(value=num_frames)

        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
        wanvideoclipvisionencode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
        wanvideoimagetovideomultitalk = NODE_CLASS_MAPPINGS[
            "WanVideoImageToVideoMultiTalk"
        ]()
        melbandroformersampler = NODE_CLASS_MAPPINGS["MelBandRoFormerSampler"]()
        multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideopassimagesfromsamples = NODE_CLASS_MAPPINGS[
            "WanVideoPassImagesFromSamples"
        ]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        previewany = NODE_CLASS_MAPPINGS["PreviewAny"]()
        
        # Only enable block swap if resolution exceeds 832x480 or 480x832
        if (height > 832 and width > 480) or (width > 832 and height > 480):
            print("Block swap enabled, reloading...")
            block_swap_args = get_value_at_index(model_dict["wanvideoblockswap_134"], 0)
        else:
            print("Block swap disabled, reloading...")
            block_swap_args = None
            
        for q in range(1):
            print("reloading model...twice")
            requested_wan_args = dict(
                model_dict["_wan_loader_args"],
                compile_args=get_value_at_index(model_dict["wanvideotorchcompilesettings_177"], 0),
                lora=get_value_at_index(model_dict["wanvideoloraselect_138"], 0),
                multitalk_model=get_value_at_index(model_dict["multitalkmodelloader_120"], 0),
                block_swap_args=block_swap_args,
            )
            initialized_wan_args = model_dict["_wan_loader_args"]
            same_args = all(
                old is new or (
                    isinstance(old, (str, int, float, bool, type(None))) and old == new
                )
                for key, old in initialized_wan_args.items()
                for new in (requested_wan_args[key],)
            )
            # Subsequent requests have not been validated with the reused model.
            if not (OPTIMIZED and same_args and model_dict["_optimized_request_count"] == 1):
                model_dict["wanvideomodelloader_122"] = wanvideomodelloader.loadmodel(
                    **requested_wan_args
                )

            imageresizekjv2_281 = imageresizekjv2.resize(
                width=get_value_at_index(intconstant_245, 0),
                height=get_value_at_index(intconstant_246, 0),
                upscale_method="lanczos",
                keep_proportion="crop",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=16,
                device="cpu",
                per_batch=0,
                image=get_value_at_index(loadimage_284, 0),
                unique_id=16490626496028923479,
            )

            getimagesizeandcount_291 = getimagesizeandcount.getsize(
                image=get_value_at_index(imageresizekjv2_281, 0)
            )

            wanvideoclipvisionencode_237 = wanvideoclipvisionencode.process(
                strength_1=1,
                strength_2=1,
                crop="center",
                combine_embeds="average",
                force_offload=True,
                tiles=0,
                ratio=0.5,
                clip_vision=get_value_at_index(model_dict["clipvisionloader_238"], 0),
                image_1=get_value_at_index(getimagesizeandcount_291, 0),
            )

            wanvideoimagetovideomultitalk_192 = wanvideoimagetovideomultitalk.process(
                width=get_value_at_index(getimagesizeandcount_291, 1),
                height=get_value_at_index(getimagesizeandcount_291, 2),
                frame_window_size=FRAME_WINDOW_SIZE,
                motion_frame=9,
                force_offload=False,
                colormatch="disabled",
                tiled_vae=False,
                mode="infinitetalk",
                output_path="",
                vae=get_value_at_index(model_dict["wanvideovaeloader_129"], 0),
                start_image=get_value_at_index(getimagesizeandcount_291, 0),
                clip_embeds=get_value_at_index(wanvideoclipvisionencode_237, 0),
            )

            melbandroformersampler_302 = melbandroformersampler.process(
                model=get_value_at_index(model_dict["melbandroformermodelloader_301"], 0),
                audio=get_value_at_index(loadaudio_125, 0),
            )

            multitalkwav2vecembeds_194 = multitalkwav2vecembeds.process(
                normalize_loudness=True,
                num_frames=get_value_at_index(intconstant_312, 0),
                fps=fps,
                audio_scale=1,
                audio_cfg_scale=1,
                multi_audio_type="para",
                wav2vec_model=get_value_at_index(model_dict["downloadandloadwav2vecmodel_137"], 0),
                audio_1=get_value_at_index(melbandroformersampler_302, 0),
            )
            
            if seed == -1:
                seed = random.randint(1, 10000)
            teacache_args = None
            if OPTIMIZED:
                teacache_args = NODE_CLASS_MAPPINGS["WanVideoTeaCache"]().process(
                    rel_l1_thresh=0.26,
                    start_step=1,
                    end_step=2,
                    cache_device="main_device",
                    use_coefficients=True,
                    mode="e",
                )[0]
            wanvideosampler_128 = wanvideosampler.process(
                steps=4,
                cfg=1.0000000000000002,
                shift=11.000000000000002,
                seed=seed,
                force_offload=True,
                scheduler="dpm++_sde",
                riflex_freq_index=0,
                denoise_strength=1,
                batched_cfg=False,
                rope_function="comfy",
                start_step=0,
                end_step=-1,
                add_noise_to_samples=True,
                cache_args=teacache_args,
                model=get_value_at_index(model_dict["wanvideomodelloader_122"], 0),
                image_embeds=get_value_at_index(wanvideoimagetovideomultitalk_192, 0),
                text_embeds=get_value_at_index(wanvideotextencodecached_241, 0),
                multitalk_embeds=get_value_at_index(multitalkwav2vecembeds_194, 0),
            )

            wanvideopassimagesfromsamples_309 = wanvideopassimagesfromsamples.decode(
                samples=get_value_at_index(wanvideosampler_128, 0)
            )

            vhs_video_save = NODE_CLASS_MAPPINGS["VHS_VideoSave"]()
            vhs_videocombine_30 = vhs_video_save.save_video_with_audio(
                frame_rate=fps,
                save_path = save_path,
                images=get_value_at_index(wanvideopassimagesfromsamples_309, 0),
                audio=get_value_at_index(loadaudio_125, 0),
                upsampler = upsampler
            )


            # vhs_videocombine_131 = vhs_videocombine.combine_video(
            #     frame_rate=fps,
            #     loop_count=0,
            #     filename_prefix="WanVideo2_1_InfiniteTalk",
            #     format="video/h264-mp4",
            #     pix_fmt="yuv420p",
            #     crf=19,
            #     save_metadata=True,
            #     trim_to_audio=False,
            #     pingpong=False,
            #     save_output=True,
            #     save_path="/home/yash/ComfyUI/output.mp4",
            #     images=get_value_at_index(wanvideopassimagesfromsamples_309, 0),
            #     audio=get_value_at_index(loadaudio_125, 0),
            #     unique_id=4728003073032921359,
            # )

            # previewany_293 = previewany.main(
            #     source=get_value_at_index(multitalkwav2vecembeds_194, 2)
            # )
            
            return seed
