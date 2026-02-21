# Detection Pipeline

This folder runs the detection stage of the dataconcept annotation pipeline. It uses [RAM++](https://github.com/xinyu1205/recognize-anything) for open-set tagging and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for grounded bounding box detection on tar-based image datasets.

## Overview

The pipeline has two main scripts:

- **`run_rampp_only.py`** runs RAM++ tagging only, saving per-image tags and class probabilities as pickle files. Use this when you only need tags (e.g. for filtering or captioning prompts).

- **`ensemble_boxes.py`** runs the full pipeline: RAM++ tagging, GroundingDINO detection at multiple image scales, per-scale NMS, and weighted box fusion across scales. It outputs one JSON per image containing tags, bounding boxes, confidence scores, and class labels.

## Setup

### GroundingDINO

Clone GroundingDINO into this directory and install it:

```bash
cd dataconcept/detection
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

If the GroundingDINO build fails, try the following:

1. Make sure GCC <= 11 is available. If your system has a newer version, you can install GCC 11 locally:
   ```bash
   wget http://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
   tar xzf gcc-11.4.0.tar.gz && cd gcc-11.4.0
   ./contrib/download_prerequisites
   mkdir build && cd build
   ../configure --prefix=$HOME/.local/gcc-11.4.0 --disable-multilib
   make -j$(nproc) && make install
   export PATH=$HOME/.local/gcc-11.4.0/bin:$PATH
   export LD_LIBRARY_PATH=$HOME/.local/gcc-11.4.0/lib64:$LD_LIBRARY_PATH
   ```

2. CUDA 11.8 toolkit must be on your path. With conda:
   ```bash
   conda install nvidia/label/cuda-11.8.0::cuda-toolkit -c nvidia/label/cuda-11.8.0
   ```
   Or install manually:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   chmod +x cuda_11.8.0_520.61.05_linux.run
   ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=$HOME/.local/cuda-11.8 --override
   export PATH=$HOME/.local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=$HOME/.local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

3. Pin numpy and install remaining dependencies:
   ```bash
   pip install numpy==1.26.4
   pip install fairscale ensemble-boxes
   pip install ftfy regex tqdm
   pip install git+https://github.com/openai/CLIP.git
   ```

4. Match your torch/torchvision to your CUDA version:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

Check the [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) for the latest installation guidance.

### Model checkpoints

Download these model weights before running:

- **RAM++ (Swin-L)**: `ram_plus_swin_large_14m.pth` from the [RAM++ repo](https://github.com/xinyu1205/recognize-anything)
- **GroundingDINO (Swin-B)**: `groundingdino_swinb.pth` from the [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO)

## Usage

### RAM++ tagging only

```bash
python run_rampp_only.py \
    --load_path /path/to/tar/shards \
    --chunk_start 000000 \
    --chunk_end 000010 \
    --class_jsons /path/to/class_descriptions.json \
    --ram_checkpoint /path/to/ram_plus_swin_large_14m.pth \
    --features_dir /path/to/features \
    --results_dir /path/to/results
```

### Full detection with multi-scale ensemble

```bash
python ensemble_boxes.py \
    --load_path /path/to/tar/shards \
    --chunk_start 00000 \
    --chunk_end 00010 \
    --class_jsons /path/to/class_descriptions.json \
    --ram_checkpoint /path/to/ram_plus_swin_large_14m.pth \
    --grounded_checkpoint /path/to/groundingdino_swinb.pth \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
    --features_dir /path/to/features \
    --results_dir /path/to/results \
    --detection_sizes 384,512,800,1000
```

Run `python ensemble_boxes.py --help` or `python run_rampp_only.py --help` for a full list of arguments and defaults.

## Files

| File | Description |
|------|-------------|
| `ensemble_boxes.py` | Full pipeline: RAM++ tagging + GroundingDINO detection + multi-scale box ensemble |
| `run_rampp_only.py` | RAM++ open-set tagging standalone (no GroundingDINO dependency) |
| `wbf.py` | Weighted box fusion and IoU-based overlap filtering |
| `tardataset.py` | PyTorch dataset for reading images and metadata from tar archives |
| `utils.py` | Shared helpers: collation, JSON I/O, tar file utilities |
| `ram/` | RAM++ model code (Swin-L backbone, open-set inference) |
| `GroundingDINO/` | GroundingDINO model code (cloned separately, see setup above) |

## Output format

Each output JSON file contains:

```json
{
    "caption": "original alt-text from the dataset",
    "bounding_boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, 0.87, ...],
    "classes": ["dog", "bicycle", ...],
    "tags": "dog | bicycle | grass | park | ...",
    "tag_probs": [0.98, 0.95, 0.87, 0.82, ...]
}
```
