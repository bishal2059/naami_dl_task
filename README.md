# 3D Knee CT Feature Extraction & Comparison Pipeline

This repository provides a complete pipeline for extracting and comparing region-specific features from a 3D knee CT scan using an inflated 3D DenseNet-121. Results (cosine similarities) are saved in CSV form.

---

## 1. Environment Setup

### 1.1. Clone & Navigate

```bash
git clone git@github.com:bishal2059/naami_dl_task.git
cd naami_dl_task
```

### 1.2. Create & Activate Conda Environment

```bash
conda create -n knee3d python=3.10 -y
conda activate knee3d
```

### 1.3. Install Dependencies

```bash
pip install torch torchvision nibabel scipy numpy
```


## 2. Pipeline Workflow Overview

1. **Data Loading & Cropping**

   * Read the input CT volume (`.nii.gz`) and corresponding mask.
   * For each region label (1 = tibia, 2 = femur, 0 = background), crop to its bounding box ±5 voxels and normalize.

2. **2D→3D Model Inflation**

   * Take a pretrained 2D DenseNet-121.
   * Recursively replace every `Conv2d` → `Conv3d`, `BatchNorm2d` → `BatchNorm3d`, and 2D pooling → 3D pooling.
   * Collapse the first `conv0` layer from 3‑channel to 1‑channel to handle CT intensity.

3. **Feature Extraction**

   * Forward each cropped region through the 3D network.
   * Capture **all** `Conv3d` activations as feature maps.
   * For each region, global-average-pool the **last**, **3rd-last**, and **5th-last** conv outputs to obtain three fixed-length feature vectors (`layer_1`, `layer_2`, `layer_3`).

4. **Feature Comparison**

   * Compute **cosine similarity** between feature vectors of region pairs:

     * `tibia ↔ femur`
     * `tibia ↔ background`
     * `femur ↔ background`

5. **Results**

   * Save a CSV (`results.csv`) with columns:

     ```csv
     pair,layer_1,layer_2,layer_3
     ```

## 3. Data Organization

* Place your CT volume and mask in a root folder, e.g.:

  ```
    ├── 3702_left_knee.nii.gz
    └── left_knee_mask.nii.gz
  ```

## 4. Running the Pipeline

### 4.1. Local Python Script

```bash
python pipeline.py \
  --volume 3702_left_knee.nii.gz \
  --mask   left_knee_mask.nii.gz \
  --output results.csv \
  --device cpu
```

### 4.2. Google Colab (Recommended for TPU/GPU)

1. Open `notebook/naami_dl_task_pipeline.ipynb` in Colab.
2. Change runtime to **TPU** (v2-8) or **GPU**.
3. Upload your volume and mask or mount from Drive.
4. Run cells sequentially to see logs and results.

## 5. Interpreting Results

* Each row corresponds to one region-pair.

* **`layer_1`**, **`layer_2`**, **`layer_3`** represent feature similarity at different depths in the network:

  * `layer_1`: similarity of the **last** convolutional layer’s global-pooled feature vector.
  * `layer_2`: similarity of the **3rd-last** conv layer.
  * `layer_3`: similarity of the **5th-last** conv layer.

* **Negative** cosine values (e.g., `-0.81`) indicate opposite feature orientations, while **positive** values (e.g., `0.34`) indicate similarity.

