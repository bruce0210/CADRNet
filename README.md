
# CADRNet

**CADRNet: Geometric Figure Recognition in Architectural Drawings via Dynamic Feature Pyramid and Cross-Architecture Adapter**


---

## 🔍 Introduction

CADRNet is a hybrid deep learning architecture designed to recognize geometric symbols in architectural CAD drawings. It tackles challenges such as:

- Large scale variations among symbols  
- Complex topological dependencies  
- Noisy or blurred drawing content  

CADRNet achieves this through:

- **DFPN (Dynamic Feature Pyramid Network)**: Attention-guided, multi-scale feature extraction  
- **CTAdapter (Cross-Architecture Adapter)**: Bidirectional fusion between CNN and Transformer branches  

---

## 🧠 Core Contributions

1. **CNN-Transformer Hybrid Backbone**  
   Combines local texture perception (CNN) with global semantic modeling (Transformer)

2. **DFPN**  
   Adaptive fusion of shallow and deep feature maps for scale-robustness

3. **CTAdapter**  
   Cross-architecture, context-aware fusion with axial pooling and residual gating

4. **Training Strategy**  
   Joint supervision with focal loss and BCE loss, tailored for class imbalance and boundary sharpness

---

## 📂 Dataset: FloorPlanCAD

We use the **FloorPlanCAD** dataset introduced by [Fan et al., ICCV 2021](https://doi.org/10.1109/ICCV48922.2021.00997), a large-scale benchmark for panoptic symbol recognition in architectural CAD drawings.

### 🔧 Key Features:
- **15,000+ floor plans**: residential and commercial buildings
- **Rich annotation**: walls, windows, HVAC, pipelines, etc.
- **Symbol scale**: from **<32 px to >512 px**
- **Topology-aware**: nested rooms, pipe chains, etc.
- **Challenges**: long-tail distribution, noise, fuzzy boundaries

### 📁 Preprocessed Directory Structure

```bash
datasets/
└── floorplancad_v2/
    ├── images/         # Rasterized PNGs
    ├── labels/         # Ground truth masks
    ├── svg/            # Original vector drawings
    ├── npy/            # Numpy-processed data
```

### ⚙️ Preprocessing Commands

```bash
# Convert SVG to PNG
python preprocess/svg2png.py --train_00 ./datasets/svg_raw/train --svg_dir ./datasets/svg_processed/svg --png_dir ./datasets/svg_processed/png --scale 7 --cvt_color

# Generate .npy from vectorized CAD
python preprocess/preprocess_svg.py -i ./datasets/svg_processed/svg/train -o ./datasets/svg_processed/npy/train --thread_num 64

# Create symbolic link
mkdir -p ./data
ln -s /your/path/to/datasets/svg_processed ./data/floorplancad_v2
```

---

## 🛠️ Environment Setup

```bash
# Install Anaconda (Linux CentOS 8.0)
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
./Anaconda3-2024.10-1-Linux-x86_64.sh

# Create conda environment
conda create -n CADRNet python=3.7.7 -y
conda activate CADRNet

# Install dependencies
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install scikit-learn=1.0.1 pillow=8.3.1 matplotlib scipy tqdm git -y
pip install opencv-python gdown svgpathtools timm==0.4.12
```

---

## 🚀 Training & Testing

### 🔧 Multi-GPU Distributed Training (DDP)

```bash
# 2 GPU (e.g. NVIDIA A10x2x24GB)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cad_ddp.py --data_root ./data/floorplancad_v2 --pretrained_model ./pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
```

### 📈 Evaluation on Test Set

```bash
# Load best model and run test
python -m torch.distributed.launch --nproc_per_node=2 train_cad_ddp.py --data_root ./data/floorplancad_v2 --pretrained_model ./pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --test_only --load_ckpt ./logs/best_model.pth
```

---

## 📊 Evaluation Metrics

We adopt the **panoptic segmentation evaluation protocol** proposed by [Kirillov et al., 2019], which includes:

| Metric | Description |
|--------|-------------|
| **PQ (Panoptic Quality)** = RQ × SQ | Overall metric for joint segmentation and recognition |
| **SQ (Segmentation Quality)** | Measures IoU overlap between predicted and ground-truth masks |
| **RQ (Recognition Quality)** | Measures detection/recognition precision and recall |

```bash
# Compute PQ, SQ, and RQ
python scripts/evaluate_pq.py --raw_pred_dir ./data/output/raw_predictions/ --svg_pred_dir ./data/svg_predictions/ --svg_gt_dir ./data/svg_ground_truth/ --thread_num 64
```

---

## 📁 Project Structure

```
CADRNet/
├── model/              # CADRNet architecture (DFPN, CTAdapter, etc.)
├── dataset/            # Data loaders and preprocessing
├── train_cad_ddp.py    # Main training script with DDP
├── preprocess/         # Data conversion (SVG to PNG/Numpy)
├── scripts/            # Evaluation utilities
├── config.py           # Hyperparameter settings
└── README.md
```

---

## 📚 Citation

**CADRNet: Geometric Figure Recognition in Architectural Drawings via Dynamic Feature Pyramid and Cross-Architecture Adapter**

---

## 📬 Contact

For any questions or collaborations:  
📧 hi-bruce.yin@connect.polyu.hk
