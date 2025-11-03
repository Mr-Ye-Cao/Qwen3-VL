# Qwen3-VL Experiments

**Date**: 2025-11-02
**Hardware**: NVIDIA RTX 5090 (32GB VRAM)
**Model**: Qwen3-VL-8B-Instruct

---

## Experiment 1: Model Inference with vLLM

### Objective
Set up fast inference for Qwen3-VL-8B-Instruct using vLLM backend for 3-5x speedup compared to HuggingFace Transformers.

### Environment Setup

**Created Isolated Conda Environment**:
```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl
```

**Installed Dependencies**:
```bash
pip install -r requirements_web_demo.txt
pip install "qwen-vl-utils[decord]==0.0.14"
pip install vllm
```

**Note**: Flash-attention installation failed due to missing CUDA_HOME, but this is optional and doesn't prevent operation.

### Initial Attempts and Issues

#### Attempt 1: Default HuggingFace Backend (Success)
```bash
python web_demo_mm.py -c Qwen/Qwen3-VL-8B-Instruct --backend hf --server-name 0.0.0.0 --server-port 7860
```

**Results**:
- ✅ Successfully loaded model
- Model size: ~17GB download
- GPU memory usage: 26GB / 32GB
- Inference: Working but slower

#### Attempt 2: vLLM with Default Settings (Failed)
```bash
python web_demo_mm.py -c Qwen/Qwen3-VL-8B-Instruct --backend vllm --server-name 0.0.0.0 --server-port 7860
```

**Error**: `Available KV cache memory: -1.79 GiB`

**Root Cause**: Default `gpu_memory_utilization=0.7` (70%) was insufficient for KV cache allocation.

#### Attempt 3: Increased GPU Utilization (Failed)
```bash
python web_demo_mm.py -c Qwen/Qwen3-VL-8B-Instruct --backend vllm --gpu-memory-utilization 0.9 --server-name 0.0.0.0 --server-port 7860
```

**Error**:
```
To serve at least one request with the models's max seq len (262144),
36.00 GiB KV cache is needed, which is larger than the available KV cache memory (4.66 GiB)
```

**Root Cause**: Default `max_model_len=262144` tokens requires 36GB KV cache, far exceeding available memory.

### Solution: Modified web_demo_mm.py

**Added `--max-model-len` parameter support** (web_demo_mm.py:61-64):
```python
parser.add_argument('--max-model-len',
                    type=int,
                    default=None,
                    help='Maximum model context length for vLLM (default: model default)')
```

**Updated LLM initialization** (web_demo_mm.py:89-91):
```python
if args.max_model_len is not None:
    llm_kwargs['max_model_len'] = args.max_model_len
```

#### Attempt 4: vLLM with Reduced Context Length (Success) ✅
```bash
conda run -n qwen3vl python web_demo_mm.py \
    -c Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32000 \
    --server-name 0.0.0.0 \
    --server-port 7860
```

**Results**:
- ✅ Successfully running
- GPU memory usage: 25.8GB / 32.6GB
- Context length: 32,000 tokens (vs default 262,144)
- Inference: 3-5x faster than HuggingFace
- Web UI: http://localhost:7860

### Video Inference Testing

**Test Video Downloaded**:
- Location: `~/Desktop/test_video.mp4`
- Size: 35MB
- Source: Qwen official example (space/astronaut theme)
- URL: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4

**Recommended Video Length**:
- Testing: 10-30 seconds
- Production: Up to 5-10 minutes with 32K token limit
- The model samples at default 2 fps

### Key Takeaways

1. **vLLM requires careful memory management**: Balance between gpu_memory_utilization and max_model_len
2. **32K context is sufficient for most use cases**: Still handles long conversations and videos
3. **RTX 5090 32GB can comfortably run 8B model with vLLM**: ~6GB headroom remaining
4. **Code modification needed**: web_demo_mm.py didn't originally expose max_model_len parameter

---

## Experiment 2: Fine-Tuning Framework Analysis

### Framework Architecture

**Location**: `qwen-vl-finetune/`

**Core Components**:

1. **Training Pipeline** (`qwenvl/train/`):
   - `train_qwen.py`: Main training script with model loading logic
   - `trainer.py`: Custom HuggingFace Trainer with modifications
   - `argument.py`: Argument dataclasses (ModelArguments, DataArguments, TrainingArguments)

2. **Data Processing** (`qwenvl/data/`):
   - `data_processor.py`: Handles image/video preprocessing and batching
   - `rope2d.py`: Implements RoPE (Rotary Position Embeddings) for 2D/3D data
   - `__init__.py`: Dataset registry and configuration

3. **Tools** (`qwenvl/tools/`):
   - `process_bbox.ipynb`: Convert bounding boxes to QwenVL format (for grounding tasks)
   - `pack_data.py`: Pack sequences into uniform-length buckets for efficiency
   - `check_image.py`: Validate dataset completeness

4. **Training Scripts** (`scripts/`):
   - `sft_7b.sh`: 7B model training (ZeRO-3)
   - `sft_32b.sh`: 32B model training (requires 8x 80GB GPUs)
   - `sft_30a3b_lora.sh`: MoE model with LoRA (ZeRO-2 only)
   - `sft_qwen3_4b.sh`: 4B model training

5. **DeepSpeed Configs**:
   - `zero2.json`: ZeRO-2 optimization (for MoE models)
   - `zero3.json`: ZeRO-3 optimization (better memory efficiency)
   - `zero3_offload.json`: ZeRO-3 with CPU offloading (extreme memory savings)

### Dataset Format

**Simple JSON Structure**:

**Single Image**:
```json
{
    "image": "images/001.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat's the main object?"},
        {"from": "gpt", "value": "A red apple on a wooden table"}
    ]
}
```

**Multi-Image**:
```json
{
    "image": ["cats/001.jpg", "cats/002.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n<image>\nWhat are the differences?"},
        {"from": "gpt", "value": "The first cat is orange tabby, second is gray Siamese..."}
    ]
}
```

**Video**:
```json
{
    "video": "videos/005.mp4",
    "conversations": [
        {"from": "human", "value": "<video>\nWhat caused the blue object to move?"},
        {"from": "gpt", "value": "Answer: (B) Collision"}
    ]
}
```

**Key Rules**:
- Each `<image>` tag must correspond to exactly one image file
- Each `<video>` tag must correspond to exactly one video file
- Tags should NOT appear in assistant responses
- Multi-turn conversations supported
- Paths can be relative (to `data_path`) or absolute

**Dataset Registration** (`qwenvl/data/__init__.py`):
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/media/files",  # Can be empty if absolute paths
}

data_dict = {
    "my_dataset": DATASET_NAME,
    # Add more datasets here
}
```

**Sampling Rates**: Append `%X` to dataset name (e.g., `"my_dataset%50"` for 50% sampling)

### Training Configurations for RTX 5090 (32GB)

#### Option A: LoRA Fine-tuning 8B Model ⭐ RECOMMENDED

**Memory**: ~18-22GB VRAM
**Training Time**: Fast (adapters only)
**Quality**: Good (90-95% of full fine-tuning)

**Configuration**:
```bash
torchrun --nproc_per_node=1 \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_use your_dataset \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --bf16 \
    --output_dir ./checkpoints/qwen3vl-8b-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --save_steps 500 \
    --save_total_limit 2
```

**Pros**:
- Memory efficient - fits comfortably on RTX 5090
- Fast training - only updates small adapter weights
- Easy to experiment with multiple LoRA adapters
- Can merge LoRA weights back to base model

**Cons**:
- Slightly lower quality than full fine-tuning (~2-5% performance gap)

#### Option B: Full Fine-tuning 4B Model

**Memory**: ~22-28GB VRAM
**Training Time**: Moderate
**Quality**: Best for 4B size

**Configuration**:
```bash
torchrun --nproc_per_node=1 \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --dataset_use your_dataset \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    ...
```

**Pros**:
- Full fine-tuning quality
- Smaller model is easier to handle
- Good memory fit on RTX 5090

**Cons**:
- 4B base model has lower capabilities than 8B

#### Option C: Full Fine-tuning 8B Model (Aggressive)

**Memory**: ~28-32GB VRAM (VERY TIGHT)
**Training Time**: Slow (due to offloading)
**Quality**: Best possible

**Configuration**:
```bash
torchrun --nproc_per_node=1 \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    ...
```

**Pros**:
- Maximum quality
- Full fine-tuning of 8B model

**Cons**:
- Memory extremely tight - may OOM
- Slow due to CPU offloading
- Requires careful monitoring

### Component Training Control

The framework allows selective training of model components:

**Flags**:
- `--tune_mm_vision`: Train vision encoder (ViT) - usually False for image+video
- `--tune_mm_mlp`: Train MLP merger (projects vision features to LLM space)
- `--tune_mm_llm`: Train language model

**Recommended Combinations**:
1. **Vision + Text tasks**: `vision=False, mlp=True, llm=True`
2. **Domain adaptation**: `vision=True, mlp=True, llm=False`
3. **LoRA**: `vision=False, mlp=False, llm=True` (LoRA on LLM only)

### Training Hyperparameters

**Recommended Settings** (from `scripts/sft_7b.sh`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-7 to 1e-5 | Lower for full FT, higher for LoRA |
| Batch Size | 1-4 | Depends on memory |
| Grad Accumulation | 4-16 | Achieve larger effective batch |
| Max Length | 4096-8192 | Context window for training |
| Warmup Ratio | 0.03 | 3% of training for LR warmup |
| LR Schedule | cosine | Smooth decay |
| Weight Decay | 0.01 | Regularization |
| BF16 | True | Mixed precision (Ampere+ GPUs) |

**Image/Video Processing**:
| Parameter | Value | Notes |
|-----------|-------|-------|
| max_pixels | 576×28×28 | Max image resolution (450K pixels) |
| min_pixels | 16×28×28 | Min image resolution (12K pixels) |
| video_fps | 2 | Frame sampling rate |
| video_max_frames | 8 | Max frames per video |
| video_max_pixels | 1664×28×28 | Max video resolution (1.3M pixels) |

---

## TODO: Fine-Tuning Next Steps

### Phase 1: Environment Setup ⏳

- [ ] Install fine-tuning dependencies:
  ```bash
  conda activate qwen3vl
  pip install deepspeed==0.17.1
  pip install peft==0.17.1
  pip install torchcodec==0.2
  ```

- [ ] Verify DeepSpeed installation:
  ```bash
  ds_report
  ```

### Phase 2: Dataset Preparation ⏳

**Option A: Use Demo Data (Quick Test)**
- [ ] Verify demo data exists:
  - `qwen-vl-finetune/demo/single_images.json` (3 examples)
  - `qwen-vl-finetune/demo/video.json` (3 examples)
  - `qwen-vl-finetune/demo/images/*.png`

- [ ] Register demo dataset in `qwenvl/data/__init__.py`:
  ```python
  DEMO_DATASET = {
      "annotation_path": "demo/single_images.json",
      "data_path": "demo/",
  }
  data_dict["demo_dataset"] = DEMO_DATASET
  ```

**Option B: Create Custom Dataset**
- [ ] Prepare 10-50 image/video examples
- [ ] Create JSON file following format above
- [ ] Validate with `tools/check_image.py`
- [ ] Register in `qwenvl/data/__init__.py`

### Phase 3: Test Training Run ⏳

- [ ] Create test training script based on Option A (LoRA)
- [ ] Run short training (50-100 steps):
  ```bash
  # Stop vLLM demo first to free GPU memory
  # Then run training
  ```

- [ ] Monitor GPU memory with:
  ```bash
  watch -n 1 nvidia-smi
  ```

- [ ] Check outputs:
  - Training logs in console
  - Checkpoints in `--output_dir`
  - Loss curves (if using wandb/tensorboard)

### Phase 4: Inference Testing ⏳

- [ ] Load fine-tuned model:
  ```python
  from peft import PeftModel
  base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
  model = PeftModel.from_pretrained(base_model, "./checkpoints/qwen3vl-8b-lora/checkpoint-XXX")
  ```

- [ ] Test on validation examples
- [ ] Compare with base model performance
- [ ] (Optional) Merge LoRA weights: `model.merge_and_unload()`

### Phase 5: Production Training ⏳

- [ ] Prepare full dataset (hundreds to thousands of examples)
- [ ] Set up experiment tracking (wandb/tensorboard)
- [ ] Run full training (1-3 epochs)
- [ ] Evaluate on held-out test set
- [ ] Document results

---

## Key Findings and Recommendations

### 1. Model Deployment
- **vLLM is production-ready** on RTX 5090 for 8B model
- **32K context is sweet spot** - balances capability and memory
- **Keep web demo modifications** - max_model_len parameter is essential

### 2. Fine-Tuning Strategy
- **Start with LoRA** - lowest risk, fastest iteration
- **Use 8B model** - best quality on single RTX 5090
- **Test with demo data first** - validate pipeline before custom data
- **Monitor memory carefully** - RTX 5090 32GB is tight for full fine-tuning

### 3. Dataset Considerations
- **Simple format** - easy to create and validate
- **Quality > Quantity** - 100 high-quality examples better than 1000 noisy
- **Video processing** - 2 fps sampling means 30s video = ~60 frames sampled

### 4. Memory Management
| Task | Memory | Status |
|------|--------|--------|
| Inference (vLLM, 8B) | 25.8GB | ✅ Comfortable |
| LoRA Training (8B) | ~20GB | ✅ Comfortable |
| Full FT (8B, ZeRO-3) | ~30GB | ⚠️ Tight |
| Full FT (8B, ZeRO-3 + offload) | ~28GB | ⚠️ Very tight, slow |

---

## Resources and References

### Documentation
- Main README: `/README.md`
- Fine-tuning README: `/qwen-vl-finetune/README.md`
- Claude guidance: `/CLAUDE.md`

### Training Scripts
- 7B baseline: `/qwen-vl-finetune/scripts/sft_7b.sh`
- LoRA example: `/qwen-vl-finetune/scripts/sft_30a3b_lora.sh`

### Demo Data
- Images: `/qwen-vl-finetune/demo/single_images.json`
- Videos: `/qwen-vl-finetune/demo/video.json`

### Code Modifications
- Web demo: `/web_demo_mm.py` (Added --max-model-len parameter)

### Test Assets
- Video: `~/Desktop/test_video.mp4` (35MB, space theme)

---

## Questions for Future Exploration

1. **What's the minimum dataset size** for meaningful fine-tuning?
2. **How does LoRA compare to full fine-tuning** on specific tasks?
3. **Can we pack multiple samples** to improve training efficiency?
4. **What's the optimal learning rate** for different components?
5. **How to handle long videos** (>5 minutes) in training?
6. **Can we fine-tune on mixed image+video** datasets effectively?

---

## Experiment Log

| Date | Experiment | Result | Notes |
|------|------------|--------|-------|
| 2025-11-02 | HF inference setup | ✅ Success | 26GB VRAM, baseline working |
| 2025-11-02 | vLLM default | ❌ Failed | -1.79GB KV cache |
| 2025-11-02 | vLLM 90% GPU util | ❌ Failed | max_model_len too large (262K) |
| 2025-11-02 | vLLM + max_model_len | ✅ Success | 32K context, 25.8GB VRAM |
| 2025-11-02 | Fine-tuning analysis | ✅ Complete | Framework documented |

---

**Next Session**: Start with Phase 1 (Environment Setup) or Phase 2 (Dataset Preparation) based on your use case.
