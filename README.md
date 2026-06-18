# CoFiVLA: Synergistic Coarse-Fine Vision-Language Alignment for Image Aesthetic Assessment

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202025-blue)](https://doi.org/10.1145/3746027.3755089)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **"CoFiVLA: Synergistic Coarse-Fine Vision-Language Alignment for Image Aesthetic Assessment"**.

CoFiVLA is designed for image aesthetic assessment. It learns aesthetic-aware vision-language representations from image-comment pairs during training, while requiring only images during inference.

---

## 📰 News

- 🎉 We release the source code of **CoFiVLA**.
- 🎉 We release the **aesthetic summary dataset** constructed from user comments.
- 🎉 CoFiVLA was accepted by **ACM MM 2025**.

---

## 💡 Overview

Image aesthetic assessment is challenging because aesthetic perception is subjective and depends on multiple factors such as composition, color, lighting, semantics, and depth of field. Existing cross-modal IAA methods often use paired image-comment data, but user comments are usually unavailable during inference. In addition, user comments contain both highly aesthetic-related information and weakly aesthetic-related information.

To address these issues, **CoFiVLA** proposes a synergistic coarse-fine vision-language alignment framework. The framework contains two stages:

1. **CoFiVLA Pretraining Network**
   - Learns aesthetic-aware vision-language representations from images, complete user comments, and aesthetic summaries.
   - Uses a coarse-grained branch to align images with high-aesthetic-related summaries.
   - Uses a fine-grained branch to align images with complete user comments.

2. **CoFiVLA Prediction Network**
   - Uses the pretrained CoFiVLA model to initialize the image and text encoders.
   - Introduces learnable aesthetic quality prompts for different quality categories.
   - Uses image features and learnable prompt features for final aesthetic prediction.
   - Requires only images during inference.

---

## 📁 Repository Structure

```text
CoFiVLA/
├── PretrainingNet/
│   └── train/
│   │   ├── train.py
│   │   └── ...
│   └── ...
├── PredictionNet/
│   ├── main.py
│   └── ...
├── README.md
├── aesthetic summary dataset.txt
└── LICENSE
```

- `PretrainingNet/` contains the code for CoFiVLA pretraining.
- `PredictionNet/` contains the code for aesthetic score prediction.

---

## 🚀 What's Released

### ✅ Source Code

- CoFiVLA pretraining network.
- CoFiVLA prediction network.
- Training scripts for both stages.

### ✅ Aesthetic Summary Dataset

We release the aesthetic summary dataset used by the coarse-grained alignment branch. The summaries are generated from user comments and focus on high-aesthetic-related descriptions, such as composition, color, lighting, focus, depth of field, and overall aesthetic impression.

---

## 📦 Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/lifusheng-chn/CoFiVLA.git
cd CoFiVLA

# Create and activate conda environment
conda create -n cofivla python=3.8 -y
conda activate cofivla
```

---

## 📂 Data Preparation

### Datasets for Training

- The **pretraining network** is trained using AVA images, complete user comments, and the released aesthetic summaries.
- The **prediction network** can be trained and evaluated on IAA datasets such as AVA, PARA, and AADB.
- During inference, only images are required.

---

## 🏋️ Training

CoFiVLA contains two training stages. Please first train the pretraining network, then train the prediction network.

### Stage 1. Train the CoFiVLA Pretraining Network

Go to the training folder of `PretrainingNet`:

```bash
cd PretrainingNet/train
```

Run the pretraining script with `torchrun`:

```bash
torchrun --nproc_per_node=1 --master_port='29501' train.py
```

For multi-GPU training, you can modify `--nproc_per_node` according to the number of GPUs:

```bash
torchrun --nproc_per_node=4 --master_port='29501' train.py
```

After training, please save the pretrained checkpoint and set its path correctly before training the prediction network.

### Stage 2. Train the CoFiVLA Prediction Network

Go to the prediction network folder:

```bash
cd ../../PredictionNet
```

Run:

```bash
python main.py
```

---

## 🙏 Acknowledgement

This project is built upon several excellent open-source projects and datasets. We sincerely thank the authors of:

- [CLIP](https://github.com/openai/CLIP)
- [LLaMA](https://github.com/meta-llama/llama)

---

## 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{niu2025cofivla,
  title={CoFiVLA: Synergistic Coarse-Fine Vision-Language Alignment for Image Aesthetic Assessment},
  author={Niu, Yuzhen and Chen, Siling and Chen, Yuzhong and Li, Fusheng and Xu, Rui and Da, Hui},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025},
  doi={10.1145/3746027.3755089}
}
```
