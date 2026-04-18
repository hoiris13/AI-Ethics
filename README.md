# # ⚖️ AI Fairness & Bias — Adult Census Dataset
<div align="center">

**Exploring how demographic imbalances in training data produce discriminatory predictions — and one technique to mitigate it.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Pipeline Overview

```
┌─────────┐     ┌─────────────┐     ┌────────────────┐     ┌────────────┐     ┌────────────┐
│   EDA   │ ──▸ │ Train Model │ ──▸ │ Fairness Check │ ──▸ │ Mitigation │ ──▸ │ Discussion │
└─────────┘     └─────────────┘     └────────────────┘     └────────────┘     └────────────┘
 Protected       LR & Decision       Per-group P/R/F1       Reweight under-    Accuracy vs
 attributes      Tree, accuracy       by gender, race,       represented        Fairness
 & underrep.     only — no            age group              groups, retrain    trade-offs
 analysis        fairness yet                                & compare
```

---

## 📂 About the Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) |
| **Records** | 32,561 (after dropping missing values) |
| **Features** | 14 (age, workclass, education, occupation, race, sex, etc.) |
| **Target** | Binary income: `<=50K` (0) vs `>50K` (1) |
| **Class Split** | ~75% ≤50K / ~25% >50K |

---

## 1️⃣ EDA — Protected Attributes & Underrepresentation

Records are counted across three protected attributes, and each group's share is compared against the **balanced threshold** (`100% ÷ number of groups`) to flag underrepresentation.

### Gender

| Group | Count | Share | Status |
|-------|------:|------:|--------|
| Male | 21,790 | 66.92% | ✅ OK |
| Female | 10,771 | 33.08% | 🔴 Underrepresented |

### Race

| Group | Count | Share | Status |
|-------|------:|------:|--------|
| White | 27,816 | 85.43% | ✅ OK |
| Black | 3,124 | 9.59% | 🔴 Underrepresented |
| Asian-Pac-Islander | 1,039 | 3.19% | 🔴 Underrepresented |
| Amer-Indian-Eskimo | 311 | 0.96% | 🔴 Underrepresented |
| Other | 271 | 0.83% | 🔴 Underrepresented |

### Age Group

| Group | Count | Share | Status |
|-------|------:|------:|--------|
| 26–35 | 8,514 | 26.15% | ✅ OK |
| 36–45 | 8,009 | 24.60% | ✅ OK |
| ≤25 | 6,411 | 19.69% | 🔴 Underrepresented |
| 46–55 | 5,538 | 17.01% | 🔴 Underrepresented |
| 56+ | 4,089 | 12.56% | 🔴 Underrepresented |

> **Key Finding:** Female representation is only 33% despite being ~50% of the population. Every non-White racial group falls below the balanced 20% share — White alone accounts for 85% of the dataset.

---

## 2️⃣ Baseline Model Training

Two classifiers trained on a **70/30 stratified split** with no fairness constraints — purely optimizing for accuracy.

| Model | Accuracy | Notes |
|-------|:--------:|-------|
| **Logistic Regression** | **82.63%** | Scaled via `StandardScaler`, `max_iter=1000` |
| **Decision Tree** | **85.60%** | `max_depth=8`, unscaled features |

---

## 3️⃣ Fairness Audit

Precision, recall, and F1 for the **positive class (>50K)** are computed separately per demographic subgroup.

### Fairness by Gender (Logistic Regression)

| Group | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| Female | 3,199 | 0.509 | **0.232** | 0.319 |
| Male | 6,570 | 0.695 | **0.401** | 0.508 |

> ⚠️ **Disparity:** The model's recall for Females (23.2%) is almost half that of Males (40.1%) — it systematically under-identifies high-income women.

Fairness metrics are also computed by **race** and **age group** with grouped bar charts for visual comparison.

---

## 4️⃣ Mitigation — Sample Reweighting

Training samples are assigned weights **inversely proportional to gender group frequency**, so that underrepresented groups exert more influence during training.

```
weight = total_samples / (num_groups × group_count)

Female → 1.505×     Male → 0.749×
```

### Results After Reweighting

| Metric | Baseline | Reweighted | Change |
|--------|:--------:|:----------:|:------:|
| Overall Accuracy | 82.63% | 82.42% | −0.21% |

The model is retrained and per-group metrics are recomputed. A side-by-side bar chart visualizes the before/after fairness comparison.

---

## 5️⃣ Discussion — Accuracy vs Fairness

Optimizing for raw accuracy alone perpetuates historical biases, as the model learns from data that already reflects societal inequalities. Reweighting narrows the fairness gap at a minimal accuracy cost (−0.21%).

The recommended approach is to treat **fairness as a constraint**: maximize accuracy subject to acceptable fairness thresholds (e.g., equalized odds, demographic parity). Key considerations include:

- **Domain context** — healthcare vs. advertising carry different stakes
- **Legal requirements** — anti-discrimination laws may apply
- **Stakeholder impact** — who is harmed by prediction errors?
- **Transparency** — per-group metric reporting enables informed trade-off decisions

---

## 🚀 Quick Start

**1. Clone & enter the project**
```bash
git clone https://github.com/your-username/ai-fairness-adult.git
cd ai-fairness-adult
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib scikit-learn
```

**3. Get the dataset**

Download `adult.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/2/adult) or extract the provided `adult.zip` into the project folder.

**4. Run**
```bash
python fairness_handon.py
```
Or paste sections into Jupyter notebook cells.

---

## 📊 Generated Outputs

Running the script produces seven PNG visualizations:

| File | Description |
|------|-------------|
| `eda_gender.png` | Gender distribution & income rate |
| `eda_race.png` | Race distribution & income rate |
| `eda_age.png` | Age group distribution & income rate |
| `underrepresentation.png` | Share vs balanced threshold (red/green) |
| `fairness_gender.png` | Precision / Recall / F1 by gender |
| `fairness_race.png` | Precision / Recall / F1 by race |
| `mitigation_comparison.png` | Before vs after reweighting |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations & weight computation |
| `matplotlib` | All visualizations (no seaborn) |
| `scikit-learn` | Logistic Regression, Decision Tree, metrics, scaling |

---

<div align="center">

Built for the Machine Learning course at **UNYT** · Dataset from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)

</div>
