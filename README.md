# 🐶🐱 Cats vs Dogs Classifier (Group Project)

This repository contains a modular PyTorch pipeline to classify dog vs cat images.
Each teammate can add a new model easily (plugin style) and train it without touching others’ code.

---

## Repo Layout
```
cats-vs-dogs/
├─ src/
│ ├─ data.py # unified data loading + augmentation
│ ├─ registry.py # auto-discovers models in src/models/*
│ ├─ train.py # generic trainer (warm-up + fine-tune)
│ └─ models/
│ ├─ model1/
│ │ └─ model.py
│ ├─ model2/
│ │ └─ model.py
│ └─ model3/
│ └─ model.py
├─ configs/ # optional per-model config YAMLs
│ ├─ base.yaml
│ ├─ resnet18.yaml
│ └─ my_sequential.yaml
├─ runs/ # checkpoints, logs, submission.csv (gitignored)
├─ README.md
└─ requirements.txt


datasets/
├─ train/
│ ├─ cat/
│ └─ dog/
├─ val/
│ ├─ cat/
│ └─ dog/
└─ test/ # 500 unlabeled images
```


---

## ⚙️ Installation

```bash
git clone https://github.com/<your-org>/cats-vs-dogs.git
cd cats-vs-dogs
pip install -r requirements.txt
```

---
🛠️ Adding Your Model
Create a folder: src/models/<your_model_name>/

Inside it, add model.py:

```bash

import torch.nn as nn
NAME = "my_model"

def build(num_classes=2, pretrained=False, freeze="none"):
    # return a torch.nn.Module
    return nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32*112*112, num_classes)
    )
```
Done! The registry will discover it automatically.
---

---
🚀 Training
Example: ResNet-18 transfer learning

```bash
Copy code
python -m src.train --model resnet18 --pretrained \
    --epochs 12 --warmup_epochs 2 --data_root datasets --out_dir runs
Custom Sequential CNN from scratch:
```

```bash
Copy code
python -m src.train --model my_sequential \
    --epochs 30 --warmup_epochs 0 --data_root datasets --out_dir runs
```
Key flags:

--img_size (default 224)

--batch_size (default 64)

--aug_strength : none | standard | plus

standard = flip + rotation + zoom + brightness

plus = adds contrast/saturation jitter

--train_per_class / --val_per_class : optional subsampling for faster experiments



📊 Producing Predictions
---
Checkpoints & logs go to runs/<model_name>/best.pt.
```bash
python -m src.infer --model resnet18 --ckpt runs/resnet18/best.pt \
    --data_root datasets --out runs/resnet18/submission.csv
(Or add a simple infer.py — same transforms as val/test; outputs submission.csv with columns id,label.)
```
---