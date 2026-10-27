#!/bin/bash

# uv port of env_1_main_5090.sh. Same install order and pins; conda's only
# role in the original (providing python 3.10.16) is replaced by `uv venv`.

setup_env_1_uv() {
    local ENV_DIR="$1"
    local UVPIP="uv pip install --python ${ENV_DIR}/bin/python"

    echo "Creating uv environment at '${ENV_DIR}'..."
    uv venv --clear --python 3.10.16 "${ENV_DIR}"

    mkdir -p cache
    [ -f cache/torch-2.10.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ] || wget -O cache/torch-2.10.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl https://s3.renderplatform.com/public-renderplatform/wheels/torch-2.10.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
    [ -f cache/torchvision-0.25.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ] || wget -O cache/torchvision-0.25.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl https://s3.renderplatform.com/public-renderplatform/wheels/torchvision-0.25.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
    [ -f cache/torchaudio-2.8.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ] || wget -O cache/torchaudio-2.8.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl https://s3.renderplatform.com/public-renderplatform/wheels/torchaudio-2.8.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
    [ -f cache/pytorch_triton-3.5.0%2Bgitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ] || wget -O cache/pytorch_triton-3.5.0%2Bgitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl https://s3.renderplatform.com/public-renderplatform/wheels/pytorch_triton-3.5.0%2Bgitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

    # uv rejects the URL-encoded %2B in the filename; give it the canonical + name
    cp -f cache/pytorch_triton-3.5.0%2Bgitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl "cache/pytorch_triton-3.5.0+gitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    ${UVPIP} "cache/pytorch_triton-3.5.0+gitbbb06c03-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    ${UVPIP} --no-deps cache/torch-2.10.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl cache/torchvision-0.25.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl cache/torchaudio-2.8.0.dev20250923+cu128-cp310-cp310-manylinux_2_28_x86_64.whl

    ${UVPIP} -e real_esrgan
    ${UVPIP} -r requirements_basic.txt

    ${UVPIP} einops==0.8.1
    ${UVPIP} ninja==1.11.1.4 wheel==0.45.1 transformers==4.53.1 accelerate==1.6.0 sentencepiece==0.2.0 protobuf==3.20.2 huggingface_hub==0.30.2
    [ -f cache/nunchaku-1.0.0.dev20250820+torch2.9-py3-none-any.whl ] || wget -O cache/nunchaku-1.0.0.dev20250820+torch2.9-py3-none-any.whl https://s3.renderplatform.com/public-renderplatform/wheels/nunchaku-1.0.0.dev20250820+torch2.9-py3-none-any.whl
    ${UVPIP} --no-deps cache/nunchaku-1.0.0.dev20250820+torch2.9-py3-none-any.whl

    ${UVPIP} diffusers==0.35.1 --no-deps
    ${UVPIP} insightface==0.7.3
    ${UVPIP} onnxruntime==1.22.0
    ${UVPIP} onnxruntime-gpu==1.22.0 --no-deps
    ${UVPIP} timm==1.0.15
    ${UVPIP} filetype==1.2.0
    ${UVPIP} av==14.2.0
    ${UVPIP} --no-deps peft==0.17.0
    ${UVPIP} --no-deps torchao==0.11.0
    ${UVPIP} mmgp==3.5.10 --no-deps
    ${UVPIP} ftfy==6.3.1 --no-deps
    ${UVPIP} decord==0.6.0 --no-deps
    ${UVPIP} loguru==0.7.3 --no-deps
    ${UVPIP} librosa==0.11.0 --no-deps
    # Build against this environment's PyTorch/CUDA stack, as in the benchmark.
    # SageAttention 2.2 requires CUDA >= 12.8 on Blackwell.
    ${UVPIP} sageattention==2.2.0 --no-build-isolation --no-deps
    ${UVPIP} rembg[gpu]==2.0.65 --no-deps
    ${UVPIP} optimum_quanto==0.2.7 --no-deps
    ${UVPIP} pymatting==1.1.14 --no-deps
    ${UVPIP} pooch==1.8.2 --no-deps
    ${UVPIP} torchsde==0.2.6 trampoline==0.1.2 orderly_set==5.4.1 spandrel==0.4.1 blend_modes==2.2.0 kornia==0.8.1 comfyui-frontend-package==1.22.2 soundfile==0.13.1 deepdiff==8.5.0 segment_anything==1.0 piexif==1.1.3 py-cpuinfo==9.0.0 kornia-rs==0.1.9 wget==3.2 color-matcher==0.6.0 --no-deps
    ${UVPIP} tyro==0.9.22 --no-deps
    ${UVPIP} pykalman==0.10.1 --no-deps
    ${UVPIP} onnx==1.18.0 --no-deps
    ${UVPIP} docstring_parser==0.16.0 --no-deps
    ${UVPIP} shtab==1.7.2 --no-deps
    ${UVPIP} skbase==0.4.6 --no-deps
    ${UVPIP} omegaconf==2.3.0 --no-deps
    ${UVPIP} hydra-core==1.3.2 --no-deps
    ${UVPIP} antlr4-python3-runtime==4.9.3 --no-deps

    ${UVPIP} colorlog==6.9.0 torchdiffeq==0.2.5 open_clip_torch==3.1.0 msgpack==1.1.1 torchcodec==0.6.0 --no-deps

    ${UVPIP} openai==1.106.1 distro==1.9.0 --no-deps

    ${UVPIP} -e ml_modules/MagicQuill/MagicQuill/LLaVA/ --no-deps
    ${UVPIP} webcolors==24.11.1
    ${UVPIP} jpeg4py==0.1.4 --no-deps
    ${UVPIP} controlnet-aux==0.0.10 --no-deps
    ${UVPIP} monotonic-align==1.0.0 --no-deps
    ${UVPIP} nsfw-image-detector==0.1.2 --no-deps
    ${UVPIP} toml==0.10.2 --no-deps
    ${UVPIP} pyloudnorm==0.1.1 --no-deps
    ${UVPIP} rotary-embedding-torch==0.8.9 --no-deps
    ${UVPIP} gguf==0.17.1 --no-deps
    ${UVPIP} audioread==3.1.0 --no-deps
    ${UVPIP} soxr==1.0.0 --no-deps sniffio==1.3.1
    ${UVPIP} typeguard==4.4.4 --no-deps
    ${UVPIP} GitPython==3.1.50 pytorch-lightning==2.6.5

    # Wan-Animate-2 requires these two newer ComfyUI runtime packages. Keep them
    # exact-pinned and --no-deps so no existing main_5090 package can move.
    # comfy-kitchen supplies the INT8 ConvRot kernels the A1 configuration uses.
    ${UVPIP} comfy-kitchen==0.2.31 --no-deps
    ${UVPIP} comfy-aimdo==0.4.13 --no-deps

    # Several legacy servers resolve shared model assets relative to ml_modules.
    # Keep those paths pointed at the canonical repository-level cache.
    mkdir -p ml_modules/cache
    [ -e ml_modules/.assets ] || ln -s ../.assets ml_modules/.assets
    [ -e ml_modules/cache/liveportrait ] || ln -s ../../cache/liveportrait ml_modules/cache/liveportrait
    [ -e ml_modules/cache/index_tts_2 ] || ln -s ../../cache/index_tts_2 ml_modules/cache/index_tts_2

    # The legacy ComfyUI loaders discover $HOME/ComfyUI first. Link the model
    # files used by the v7/v9 servers into that canonical model tree.
    mkdir -p "${HOME}/ComfyUI/models/"{diffusion_models,text_encoders,vae,loras,controlnet,pulid}
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v9/models/unet/qwen_image_edit_2509_fp8_e4m3fn.safetensors" "${HOME}/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v9/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors" "${HOME}/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v9/models/vae/qwen_image_vae.safetensors" "${HOME}/ComfyUI/models/vae/qwen_image_vae.safetensors"
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v9/models/loras/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors" "${HOME}/ComfyUI/models/loras/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    for model in t5xxl_fp16.safetensors clip_l.safetensors EVA02_CLIP_L_336_psz14_s6B.pt; do
        ln -sf "${PWD}/ml_modules/comfyui_experiments_v7/models/clip/${model}" "${HOME}/ComfyUI/models/text_encoders/${model}"
    done
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v7/models/vae/ae.safetensors" "${HOME}/ComfyUI/models/vae/ae.safetensors"
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v7/models/controlnet/controlnet-union-pro-2.0.safetensors" "${HOME}/ComfyUI/models/controlnet/controlnet-union-pro-2.0.safetensors"
    ln -sf "${PWD}/ml_modules/comfyui_experiments_v7/models/pulid/pulid_flux_v0.9.1.safetensors" "${HOME}/ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors"

    local SITE_PACKAGES="${ENV_DIR}/lib/python3.10/site-packages"
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' ${SITE_PACKAGES}/basicsr/data/degradations.py
    MODEL_CONFIGS_DIR=${SITE_PACKAGES}/nunchaku/models/pulid/eva_clip/model_configs
    mkdir -p ${MODEL_CONFIGS_DIR}
    sed -i -E 's/^(\s*)qint4quantization =  isinstance\(target_module, QModuleMixin\) and  target_module\.weight!= None and  target_module\.weight\.qtype == qint4/\1try:\n\1    qint4quantization =  isinstance(target_module, QModuleMixin) and  target_module.weight!= None and  target_module.weight.qtype == qint4\n\1except:\n\1    qint4quantization = False/' ${SITE_PACKAGES}/mmgp/offload.py

    current_dir=$(pwd)
    if [ -f "${current_dir}/pulid/EVA02-CLIP-L-14-336.json" ]; then
        cp "${current_dir}/pulid/EVA02-CLIP-L-14-336.json" ${MODEL_CONFIGS_DIR}/
        echo "EVA02-CLIP-L-14-336.json file copied to ${MODEL_CONFIGS_DIR}"
    else
        echo "Warning: EVA02-CLIP-L-14-336.json file not found in the current directory ${current_dir}"
    fi
}
