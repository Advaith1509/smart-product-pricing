# ***Smart Product Pricing Challenge - A Multimodal ML Solution***  


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-3.3%2B-006400?logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/SentenceTransformers-1.2+-orange?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.5%2B-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-1.24%2B-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

### **Amazon ML Challenge 2025**  
> **Theme:** Smart Product Pricing  
> **Objective:** Predict product prices using multimodal (text + image) data  
> **Team:** *Woozie* - Advaith Moholker · Luv Valecha · Dheeraj Kumar · Meet Tiala  
> **Final Validation SMAPE:** **48.94**  

---

## ***1. Project Overview*** 

This repository presents our **end-to-end multimodal solution** for the *Amazon ML Challenge 2025 - Smart Product Pricing*.  
The goal was to build a model that predicts e-commerce product prices using both **textual descriptions** and **product images**, without any external reference data.

Our final approach fuses **deep image embeddings**, **semantic text representations**, and **tree-based modeling** with **SMAPE-aware ensembling**, achieving a **final cross-validated SMAPE of 48.94**.

---

## ***2. Technical Approach***  

### 2.1 Problem Framing  
- Formulated as a **multimodal regression** problem.  
- Observed **right-skewed target distribution** → applied `log1p(price)` transformation.  
- SMAPE penalizes **under-predictions more than over-predictions**, motivating bias correction during ensembling.  

---

## ***3. Methodology*** 

### 3.1 Strategic Framework  

#### **Metric-Aware Target Transformation**
- Applied `np.log1p(price)` to stabilize variance and align regression loss with SMAPE’s relative-error structure.  
- Predictions inverted with `exp(pred) - 1` before evaluation.  

#### **Robust Cross-Validation**
- **5-Fold CV** with stratified random sampling based on price percentiles.  
- All model and feature choices validated strictly using out-of-fold (OOF) SMAPE.  

---

### 3.2 Multimodal Feature Engineering  

#### **Visual Feature Pipeline - Dual CNN Ensemble**

| Model | Architecture | Pretrained On | Output Dim | Purpose |
|--------|---------------|----------------|--------------|-----------|
| **EfficientNet-B2** | CNN | ImageNet | 1408 | Efficient & accurate visual baseline |
| **ConvNeXt-Tiny** | Modern CNN | ImageNet | 768 | Transformer-inspired CNN adding diversity |

**Process Summary:**  
1. Images resized (224–260 px) and normalized.  
2. Final classification layers replaced with `Identity()` to extract penultimate embeddings.  
3. Concatenated both embeddings → **2176-dimensional feature vector per image**.  
4. Stored embeddings as `.parquet` files for reproducibility.

---

#### **Textual Feature Pipeline - Hybrid Representation**

| Component | Technique | Dimensionality | Purpose |
|------------|------------|----------------|-----------|
| **Item Pack Quantity (IPQ)** | Regex-based extraction | 1 | Captures multiplicative pricing effect |
| **TF-IDF Vectorization** | `TfidfVectorizer(ngram_range=(1,2))` | ~20k | Sparse keyword representation |
| **Sentence Embeddings** | `SentenceTransformer('all-MiniLM-L6-v2')` | 384 | Captures semantic context |

The concatenation of structured, sparse, and dense text features provided a **rich, hybrid representation** of each product’s description.

---

### 3.3 Modeling and Ensembling  

#### **Base Models**
- **LightGBM**: Primary regressor trained on combined text + image embeddings.  
- **Transformer Regressor (PyTorch)**: Auxiliary model for nonlinear residual patterns.  

#### **Weighted Ensemble Optimization**
- Predictions blended via `scipy.optimize.minimize`, optimizing weights to minimize OOF SMAPE.  
- Ensemble stabilized model variance and improved leaderboard robustness.  

#### **Bias Correction**
- Applied multiplicative bias calibration (factor ∈ [1.0, 1.5]) to compensate SMAPE under-prediction penalty.

---

## ***4. Repository Structure***

```
├── data/               			# Raw CSV data files (train.csv, test.csv)
├── notebooks/          			# Jupyter notebooks for EDA, advanced analysis, and visualization
├── src/                			# All production-ready Python source code for the pipeline
│   ├── config.py       			# Centralized configuration parameters and file paths
│   ├── download_data.py  			# Script for parallel image data ingestion
│   ├── feature_engineering.py 		# Script for multimodal (image & text) feature generation
│   ├── train.py        			# Script for LightGBM model training with CV, OOF generation, and model persistence
│   ├── predict.py      			# Script for loading trained models and generating raw predictions
│   └── blend_models.py 			# Script for ensembling multiple model predictions with optimized weights
├──.gitignore          				# Specifies files and folders to be ignored by Git
├── README.md           			# This comprehensive project documentation
└── requirements.txt    			# Python package dependencies
```

---

## ***5. Reproducibility***

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/smart-product-pricing.git
cd smart-product-pricing
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Data and Models
Pretrained models and embeddings are hosted under Releases:
	•	image_model_files.zip - CNN checkpoints
	•	embeddings.zip - Precomputed image & text embeddings
  
```bash
unzip image_model_files.zip -d .
unzip embeddings.zip -d .
```

### Step 4: Run the Pipeline
Option A - Full Training (from scratch)
```bash
python src/download_data.py
python src/feature_engineering.py
python src/train.py
python src/predict.py
```

Option B - Quick Submission (pretrained models)
```bash
python src/predict.py
```
---

## ***6. Key Insights***

* Multimodal fusion (text + vision) enhances model robustness.
* Metric-aware transformations yield more consistent leaderboard performance.
* Combining **architectural diversity (CNN + Transformer)** reduces overfitting.
* Rigorous CV discipline ensures generalization beyond leaderboard noise.

---

## ***7. License***
This project is released under the MIT License. You are free to use, modify, and distribute this work with proper attribution.

---
