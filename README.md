# Effort: Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection ([Paper](https://arxiv.org/abs/2411.15633); [Checkpoints](https://drive.google.com/drive/folders/19kQwGDjF18uk78EnnypxxOLaG4Aa4v1h?usp=sharing))

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-1.0-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

> 🎉🎉🎉 **Our paper has been accepted by ICML 2025 Spotlight ⭐!**

Welcome to our work **Effort**, for detecting AI-generated images (AIGIs). \

**In this work, we propose: (1) a **very very easy and effective method** for generalization AIGI detection😀; and (2) a **novel analysis tool** for quantifying the "degree of model's overfitting"😊.**


The figure below provides a brief introduction to our method: our method can be **plug-and-play inserted** into *any* vit-based large models such as CLIP.

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figs/effort_pipeline.png" style="max-width:60%;">
</div>


---


The following two tables display the **part results** of our method on **both the (face) deepfake detection benchmark and the (natural) AIGI detection benchmark**. Please refer to our paper for more results.

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figs/deepfake_tab1.png" style="max-width:50%;">
</div>



<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figs/genimage_tab.png" style="max-width:50%;">
</div>

---


## ⏳ Quick Start (if you just want to do the *inference*)
<a href="#top">[Back to top]</a>


### 1. Installation
Please run the following script to install the required libraries:

```
sh install.sh
```

### 2. Download checkpoints
If you are a deepfake player, more interested in face deepfake detection:
- The checkpoint of "CLIP-L14 + our Effort" **training on FaceForensics++** are released at [Google Drive](https://drive.google.com/file/d/16QdtCScfOf4ZLGEX-VdlLcJQkj2e7zu2/view?usp=drive_link).

If you are interested in detecting general AI-generated images, we provide two checkpoints that are trained on GenImage and Chameleon datasets, respectively:
- The checkpoint of "CLIP-L14 + our Effort" **training on GenImage (sdv1.4)** are released at [Google Drive](https://drive.google.com/file/d/1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d/view?usp=sharing).
- The checkpoint of "CLIP-L14 + our Effort" **training on Chameleon (sdv1.4)** are released at [Google Drive](https://drive.google.com/file/d/1GlJ1y4xmTdqV0FfIcyBwNNU6cQird9DR/view?usp=sharing).


### 3. Run demo
You can then infer **one image *or* one folder with several images** using the pretrained weights. 

Specifically, run the following line:

```
cd DeepfakeBench/

python3 training/demo.py --detector_config training/config/detector/effort.yaml --weights ./training/weights/{NAME_OF_THE_CKPT}.pth --image {IMAGE_PATH or IMAGE_FOLDER}
```

After running the above line, you can obtain the prediction results (fake probabilities) for each image. 


