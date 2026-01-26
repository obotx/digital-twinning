# Digital-Twin Pipeline Comparasion: From Images to Mesh (SAM3 + COLMAP vs SAM3 + 3DGS + SuGaR)

> Comparison of 3D reconstruction of a single object from multiple images using SAM3-based segmentation, between the COLMAP pipeline (sparse + dense via PyCOLMAP) and the 3D Gaussian Splatting (3DGS) and SuGaR methods.

<img width="935" height="600" alt="image" src="https://github.com/user-attachments/assets/8ef060ff-1fc1-447a-847e-bee8ab0f0c21" />


This README documents the **exact pipeline used**, updated to reflect a **fully working dense COLMAP stage on Colab using PyCOLMAP CUDA**, and designed to be **replicable end-to-end**.
---

## Overview (Big Picture)

The pipeline follows a simple, modular idea:

1. **Segment the object** in all images using **SAM3**
2. **Recover camera geometry** with **COLMAP + masks (sparse)**
3. **Run dense multi-view stereo** using **PyCOLMAP (CUDA)**
4. **Generate a dense mesh** via **Poisson Meshing (COLMAP)**
5. **Train a vanilla 3D Gaussian Splatting (3DGS)** scene
6. **Convert Gaussians → Mesh** using **SuGaR**
7. (Optional) **Refine mesh + texture** for higher quality

This is conceptually similar to *SAM3D*, but:

* Uses **multiple views**
* Produces **true geometry (PLY / OBJ)**
* Produces **textured meshes**
* Works for **any object-centric dataset**

---

## Dataset

Example dataset: **Bear**

* [https://huggingface.co/mqye/Gaussian-Grouping/blob/main/data/bear.zip](https://huggingface.co/mqye/Gaussian-Grouping/blob/main/data/bear.zip)

https://github.com/user-attachments/assets/843c38ac-fe03-416f-849c-9467ac1376ad

**Notes:**

* The pipeline is **object-agnostic**
* Any multi-view object dataset works
* All paths below are **relative and replaceable**

---

## Environment

* **Platform:** Google Colab
* **OS:** Linux
* **GPU:** Required for dense COLMAP + training

---

## Step 1 — Object Segmentation with SAM3

We generate:

* Binary masks (for COLMAP)
* Black-background images (to avoid background features)

### 1.1 Install SAM3

```bash
git clone https://github.com/RizwanMunawar/sam3-inference
cd sam3-inference
pip install -e .
pip install decord
```

### 1.2 Download SAM3 Model

```python
MODEL_PATH = "<PATH_TO_SAM3_CHECKPOINT>"
```

Download from:

* [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

### 1.3 Generate Masks + Black Background Images

```python
from PIL import Image
import numpy as np
import cv2
import os
from sam3.sam3_processor import Sam3Processor
from sam3.build_sam3 import build_sam3_image_model

processor = Sam3Processor(
    build_sam3_image_model(checkpoint_path=MODEL_PATH)
)

OBJECT_PROMPTS = ["<OBJECT_NAME>"]  # e.g. "bear"
DATASET_ROOT = "./dataset"
IMAGES_DIR = f"{DATASET_ROOT}/images"
OUTPUT_IMAGES_BLACK = "./colmap/images_black"
OUTPUT_MASKS = "./colmap/masks"

os.makedirs(OUTPUT_IMAGES_BLACK, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)

for img_name in sorted(os.listdir(IMAGES_DIR)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    image_pil = Image.open(img_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    state = processor.set_image(image_pil)
    final_mask = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

    for prompt in OBJECT_PROMPTS:
        results = processor.set_text_prompt(state=state, prompt=prompt)
        for mask in results["masks"]:
            binary = mask.cpu().numpy().astype(np.uint8).squeeze()
            final_mask[binary == 1] = 255

    cv2.imwrite(f"{OUTPUT_MASKS}/{img_name}.png", final_mask)

    segmented = np.zeros_like(image_cv)
    for c in range(3):
        segmented[:, :, c] = np.where(final_mask == 255, image_cv[:, :, c], 0)

    cv2.imwrite(f"{OUTPUT_IMAGES_BLACK}/{img_name}", segmented)
```

---

## Step 2 — Sparse COLMAP with Masks

```bash
apt-get update
apt-get install -y colmap libgl1-mesa-glx libglib2.0-0
```

```bash
colmap feature_extractor \
  --database_path ./colmap/database.db \
  --image_path ./colmap/images_black \
  --ImageReader.mask_path ./colmap/masks \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \
  --database_path ./colmap/database.db \
  --SiftMatching.use_gpu 0

mkdir -p ./colmap/sparse

colmap mapper \
  --database_path ./colmap/database.db \
  --image_path ./colmap/images_black \
  --output_path ./colmap/sparse \
  --Mapper.multiple_models 0

colmap image_undistorter \
  --image_path ./colmap/images_black \
  --input_path ./colmap/sparse/0 \
  --output_path ./colmap/distorted \
  --output_type COLMAP
```

---

## Step 3 — Convert Background to White (Required for 3DGS)

```python
import cv2
import numpy as np
from glob import glob

IMG_DIR = "./colmap/distorted/images"
images = glob(f"{IMG_DIR}/*.jpg") + glob(f"{IMG_DIR}/*.png")

for img_path in images:
    img = cv2.imread(img_path)
    mask = cv2.inRange(img, (0, 0, 0), (30, 30, 30))
    img[mask > 0] = [255, 255, 255]
    cv2.imwrite(img_path, img)
```

Re-run mapper:

```bash
rm -rf ./colmap/sparse && mkdir -p ./colmap/sparse

colmap mapper \
  --database_path ./colmap/database.db \
  --image_path ./colmap/distorted/images \
  --output_path ./colmap/sparse \
  --Mapper.multiple_models 0
```

---

## Step 4 — Dense COLMAP with PyCOLMAP (CUDA)

### 4.1 Install PyCOLMAP

```bash
pip install pycolmap-cuda12==3.13.0
```

### 4.2 Patch Match Stereo + Fusion

```python
import pycolmap

mvs_path = "./colmap/distorted"

print("Running Patch Match Stereo (CUDA)...")
pycolmap.patch_match_stereo(mvs_path)

print("Running Stereo Fusion...")
pycolmap.stereo_fusion(f"{mvs_path}/fused.ply", mvs_path)
```

**Output:**

* `fused.ply` → dense point cloud

---

## Step 5 — Dense Mesh via Poisson Meshing (COLMAP)

```python
import subprocess

def generate_poisson_mesh(input_ply, output_mesh):
    cmd = [
        "colmap", "poisson_mesher",
        "--input_path", input_ply,
        "--output_path", output_mesh,
        "--PoissonMeshing.trim", "5",
        "--PoissonMeshing.depth", "10"
    ]
    subprocess.run(cmd, check=True)

generate_poisson_mesh(
    "./colmap/distorted/fused.ply",
    "./outputs/mesh_colmap.ply"
)
```



https://github.com/user-attachments/assets/fbc80e6f-c74f-4139-8532-198d64b87699




---

## Step 6 — Train Vanilla 3D Gaussian Splatting (3DGS)

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
git submodule update --init --recursive
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
```

```bash
python train.py \
  -s ./colmap \
  --iterations 5000 \
  --white_background
```

---

## Step 7 — SuGaR: Gaussians → Mesh

```bash
git clone --recursive https://github.com/Anttwo/SuGaR.git
cd SuGaR
pip install plyfile==0.8.1 trimesh open3d nvdiffrast
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

```bash
python train_full_pipeline.py \
  -s ./colmap \
  -r dn_consistency \
  --high_poly True \
  --export_obj True \
  --gs_output_dir ./gaussian-splatting/output \
  --white_background True
```


https://github.com/user-attachments/assets/957c488f-5fa1-4846-8fdd-da5bf2ec6bef

https://github.com/user-attachments/assets/3a689d73-552c-4277-97bf-74d81d0b670c



---

## Step 8 — PyTorch 2.6+ Compatibility (Important)

SuGaR was developed with older PyTorch versions. From **PyTorch 2.6**, all `torch.load` calls must explicitly set:

```python
state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)
```

This is:

* Required for PyTorch ≥ 2.6
* Safe and correct for SuGaR

---


## Final Result

✔ Dense COLMAP mesh (reference geometry)

✔ Clean SuGaR geometry (OBJ)

✔ High-quality textured mesh (PLY)

✔ Fully reproducible pipeline on Colab

## ⭐ New: Placing Objects in the MuJoCo
https://github.com/user-attachments/assets/0d22a238-63cb-4d1e-b0f3-81334a2adc0a

View the complete repository [here](https://github.com/obotx/3D-scene-reconstruction-and-segmentation-to-universal-scene-description-usd). 


### ✍️ Written by Gabriel Cicotoste
🔗 GitHub: [Ga0512](https://github.com/Ga0512)  

🔗 LinkedIn: [Gabriel Cicotoste](https://www.linkedin.com/in/gabrielcicotoste/)

