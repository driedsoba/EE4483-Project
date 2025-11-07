# IE — README

This repository contains two key Jupyter notebooks used for **dataset exploration** and **model performance comparison**. Follow the instructions below to set up your environment and run the notebooks.

---

## ✅ 1. Project Structure

```
.
├── .github/
├── datasets_cifar/
├── datasets_subset/
├── models/
├── outputs/
├── runs/
├── tests/
├── eda.ipynb
├── model_comparison_analysis.ipynb
├── requirements.txt
├── your_model.py
├── LICENSE
└── README.md
```

---

## ✅ 2. Environment Setup

### **Option A — Conda (Recommended)**

```bash
conda create -n model-analysis python=3.10 -y
conda activate model-analysis
pip install -r requirements.txt
```

### **Option B — pip + Virtualenv**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## ✅ 3. Dataset Preparation

Ensure the project includes the following dataset directories:

```
datasets_cifar/
datasets_subset/
```

If your dataset is stored elsewhere, update the paths inside each notebook.

---

## ✅ 4. Running the Notebooks

### **4.1 Run `eda.ipynb`**

This notebook performs:

- Exploratory Data Analysis  
- Dataset visualization  
- Checking class distribution  
- Sample image inspection  
- Basic preprocessing checks  

Run it using:

```bash
jupyter notebook eda.ipynb
```

---

### **4.2 Run `model_comparison_analysis.ipynb`**

This notebook covers:

- Loading models from `models/`  
- Running predictions  
- Performance comparison  
- Accuracy, loss, and confusion matrix generation  
- Saving results to `outputs/` 

Run using:

```bash
jupyter notebook model_comparison_analysis.ipynb
```

Make sure your trained models are inside:

```
models/
```

---

## ✅ 5. Expected Outputs

### From `eda.ipynb`
- Class distribution plots  
- Dataset preview images  
- Preprocessing summary  

### From `model_comparison_analysis.ipynb`
- Accuracy & loss plots  
- Confusion matrices  
- Model comparison results  
- Saved outputs in:

```
outputs/
runs/
```

---

