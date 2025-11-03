# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Qwen3-VL repository, a multimodal vision-language model series that supports image and video understanding. The repository contains:
- Inference examples and cookbooks demonstrating model capabilities
- Fine-tuning framework for training custom Qwen VL models
- Utilities for processing vision inputs (images/videos)
- Web demo for interactive model testing
- Evaluation code for benchmarking

## Repository Structure

```
.
├── cookbooks/              # Jupyter notebooks demonstrating capabilities
├── qwen-vl-finetune/      # Training framework for fine-tuning
├── qwen-vl-utils/         # Vision processing utilities
├── evaluation/            # Evaluation scripts and utilities
├── docker/                # Docker configurations
├── web_demo_mm.py         # Gradio-based web interface
└── requirements_web_demo.txt
```

## Model Architecture

Qwen3-VL uses a multimodal architecture with:
- **Vision Encoder**: Processes images/videos with DeepStack multi-level ViT features
- **MLP Merger**: Projects visual features to language model space
- **Language Model**: Decoder-only transformer for text generation
- **Interleaved-MRoPE**: Positional embeddings for time, width, and height dimensions
- **Text-Timestamp Alignment**: Precise temporal grounding for video understanding

Available model variants:
- Dense models: 2B, 4B, 8B, 32B, 235B
- MoE models: 30B-A3B, 235B-A22B
- Editions: Instruct (standard) and Thinking (reasoning-enhanced)

## Common Commands

### Running Web Demo

```bash
# Install dependencies
pip install -r requirements_web_demo.txt

# Launch with Hugging Face backend (default)
python web_demo_mm.py -c Qwen/Qwen3-VL-235B-A22B-Instruct

# Launch with vLLM backend (requires vllm and qwen-vl-utils)
python web_demo_mm.py -c Qwen/Qwen3-VL-235B-A22B-Instruct --backend vllm

# Customize server settings
python web_demo_mm.py -c /path/to/checkpoint --server-port 8080 --share
```

### Fine-tuning

Fine-tuning uses the `qwen-vl-finetune/` directory:

```bash
cd qwen-vl-finetune

# Single-node training with example script
bash scripts/sft.sh

# Multi-node training
MASTER_ADDR=<master_ip> NPROC_PER_NODE=<gpus_per_node> bash scripts/sft.sh

# Example for specific model sizes
bash scripts/sft_qwen3_4b.sh     # 4B model
bash scripts/sft_7b.sh           # 7B model
bash scripts/sft_32b.sh          # 32B model (requires 8x 80GB GPUs)
bash scripts/sft_30a3b_lora.sh   # MoE with LoRA
```

Key training entry point: `qwenvl/train/train_qwen.py`

### Using DeepSpeed

The training scripts use DeepSpeed for distributed training. Configuration files:
- `scripts/zero2.json` - ZeRO Stage 2 optimization
- `scripts/zero3.json` - ZeRO Stage 3 optimization
- `scripts/zero3_offload.json` - ZeRO Stage 3 with CPU offload

Note: Qwen3VL MoE models do not support DeepSpeed ZeRO-3.

### Inference with Transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# For Qwen3-VL, image_patch_size is 16 (vs 14 for Qwen2.5-VL)
```

### Inference with vLLM

```bash
# Install vLLM (requires >= 0.11.0 for Qwen3-VL)
uv pip install -U vllm
pip install qwen-vl-utils==0.0.14

# Start vLLM server
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 22002
```

### Docker Usage

```bash
# Pull and run official Docker image
docker run --gpus all --ipc=host --network=host --rm --name qwen3vl -it \
  qwenllm/qwenvl:qwen3vl-cu128 bash

# Build custom Docker image
cd docker
bash docker_web_demo.sh -c /path/to/checkpoint --port 8881
```

## Fine-tuning Framework Architecture

### Dataset Configuration

Datasets are configured in `qwen-vl-finetune/qwenvl/data/__init__.py`:

```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/images/",  # Can be empty if paths are absolute
}

data_dict = {
    "your_dataset_name": DATASET_NAME,
}
```

Dataset sampling rates can be specified with `%` notation:
- `"dataset_name%50"` samples 50% of the data
- Multiple datasets: `--dataset_use dataset1%100,dataset2%30`

### Data Format

JSON/JSONL format with `image`/`video` field and `conversations` array:

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat's in this image?"},
        {"from": "gpt", "value": "A red apple on a table"}
    ]
}
```

For multi-image: `"image": ["img1.jpg", "img2.jpg"]` with multiple `<image>` tags.
For video: `"video": "video.mp4"` with `<video>` tag.

**Important**: One `<image>` tag must correspond to exactly one image. Same for `<video>` tags.

### Training Components

- `qwenvl/train/train_qwen.py`: Main training script
- `qwenvl/train/trainer.py`: Custom Trainer with attention class replacement
- `qwenvl/train/argument.py`: ModelArguments, DataArguments, TrainingArguments
- `qwenvl/data/data_processor.py`: Processes data into model inputs
- `qwenvl/data/rope2d.py`: RoPE position encoding implementation

### Key Training Arguments

**Component Control**:
- `--tune_mm_vision`: Train vision encoder (should be False for mixed image+video)
- `--tune_mm_mlp`: Train visual-to-language projection
- `--tune_mm_llm`: Train language model

**Data Processing**:
- `--data_flatten`: Concatenate batch sequences into one sequence
- `--data_packing`: Use pre-packed data (requires `tools/pack_data.py`)
- `--model_max_length`: Maximum sequence length (default: 4096)

**Image Resolution**:
- `--max_pixels`: Maximum pixels per image (e.g., `576*28*28`)
- `--min_pixels`: Minimum pixels per image (e.g., `16*28*28`)

**Video Processing**:
- `--video_fps`: Frame sampling rate (default: 2)
- `--video_max_frames`: Max frames per video
- `--video_min_frames`: Min frames per video
- `--video_max_pixels`: Max total pixels for video (e.g., `1664*28*28`)
- `--video_min_pixels`: Min total pixels for video

**LoRA**:
- `--lora_enable`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout rate

**Learning Rates**:
- `--learning_rate`: Base LR (recommended: 1e-6 to 2e-7)
- `--mm_projector_lr`: Separate LR for MLP projector
- `--vision_tower_lr`: Separate LR for vision encoder

### Tools

- `tools/check_image.py`: Validate dataset images exist and are readable
- `tools/pack_data.py`: Pre-pack data into even-length buckets for efficiency

## Vision Processing (qwen-vl-utils)

The `qwen-vl-utils` package provides `process_vision_info()` for handling images/videos:

**Key differences between model versions**:
- Qwen2.5-VL: `image_patch_size=14`, no video metadata
- Qwen3-VL: `image_patch_size=16`, `return_video_metadata=True`

```python
from qwen_vl_utils import process_vision_info

# For Qwen3-VL
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True
)

# Split videos and metadata
if videos is not None:
    videos, video_metadatas = zip(*videos)
```

Supports:
- Local files: `"file:///path/to/image.jpg"`
- URLs: `"http://example.com/image.jpg"`
- Base64: `"data:image;base64,..."`
- PIL Images
- Video files or frame lists

## Important Implementation Details

### Flash Attention

To enable Flash Attention 2:
1. Install: `pip install -U flash-attn --no-build-isolation`
2. Add to `config.json`: `"_attn_implementation": "flash_attention_2"`
3. Or specify in model loading: `attn_implementation="flash_attention_2"`

Requires GPU compatible with Flash Attention and `torch.float16` or `torch.bfloat16`.

### Long Context (YaRN)

To extend beyond 256K tokens (up to 1M):

Modify `config.json`:
```json
{
    "max_position_embeddings": 1000000,
    "rope_scaling": {
        "rope_type": "yarn",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": true,
        "factor": 3.0,
        "original_max_position_embeddings": 262144
    }
}
```

For vLLM: Use `--rope-scaling` and `--max-model-len` arguments.

**Note**: Use smaller scaling factors (2-3, not 4) because Interleaved-MRoPE grows position IDs more slowly than vanilla RoPE.

### Model Versions

The codebase supports:
- `Qwen2VLForConditionalGeneration` (Qwen2-VL)
- `Qwen2_5_VLForConditionalGeneration` (Qwen2.5-VL)
- `Qwen3VLForConditionalGeneration` (Qwen3-VL)
- `Qwen3VLMoeForConditionalGeneration` (Qwen3-VL MoE)

Import from `transformers` (requires `transformers >= 4.57.0`).

### Pixel Budget Control

**Official Processor**:
```python
# Image: size['longest_edge'] = max_pixels, size['shortest_edge'] = min_pixels
processor.image_processor.size = {
    "longest_edge": 1280*32*32,  # 256-1280 visual tokens
    "shortest_edge": 256*32*32
}

# Video: total pixels across all frames T×H×W
processor.video_processor.size = {
    "longest_edge": 16384*32*32,  # 256-16384 visual tokens
    "shortest_edge": 256*32*32
}
```

**qwen-vl-utils**:
Per-input control via `min_pixels`, `max_pixels`, `total_pixels`, `resized_height`, `resized_width`.

## Evaluation

Evaluation code is in `evaluation/mmmu/`:
- Uses VLMEvalKit or lmms-eval frameworks
- vLLM as inference runtime
- See README for specific hyperparameters (different for Instruct vs Thinking models)

## Known Issues and Limitations

1. **Qwen3VL MoE**: Does not support DeepSpeed ZeRO-3
2. **Training with mixed data**: Set `tune_mm_vision=False` when training with both image and video data
3. **Video decoding**: Prefer `torchcodec` backend over `decord` (unmaintained, has hanging issues)
4. **HTTPS video URLs**: Requires `torchvision >= 0.19.0` or `torchcodec`
5. **Transformers version**: Qwen3-VL requires `transformers >= 4.57.0`

## Cookbooks

The `cookbooks/` directory contains Jupyter notebooks demonstrating:
- Omni recognition (objects, people, landmarks, etc.)
- Document parsing with layout and Qwen HTML format
- 2D/3D object grounding
- OCR and key information extraction
- Video understanding and grounding
- Mobile and computer-use agents
- Multimodal coding (generating HTML/CSS/JS/Draw.io)
- Long document understanding
- Spatial understanding

These are executable examples showcasing model capabilities.
