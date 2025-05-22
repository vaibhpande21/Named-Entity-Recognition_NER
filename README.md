# Named Entity Recognition (NER) using BiLSTM and Transformer Models

This repository contains a complete pipeline for training and evaluating Named Entity Recognition (NER) models using BiLSTM and Transformer-based architectures. The pipeline includes data management, exploratory analysis, model training, and evaluation on a benchmark dataset.

---

## 📁 1. Data Management

### 1.1 Experimental Protocol

The dataset was split into training, validation, and test sets using a 70/15/15 ratio:
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

A fixed random seed (`42`) was used to ensure reproducibility across runs.

---

### 1.2 Exploratory Data Analysis

Before modeling, the dataset was explored to understand structure, imbalance, and content.

#### 🏷️ Entity Tag Distribution
- Significant class imbalance observed.
- Most tokens are labeled as non-entity (`O`).

📊 *![image](https://github.com/user-attachments/assets/c2c50a00-c1ad-4adc-9862-6f15cd04a996)*

#### 📏 Sentence Length Distribution
- Most sequences are within a manageable length range.

☁️ *![image](https://github.com/user-attachments/assets/e6c416c0-fc3c-4eb9-a322-a27b3d99835c)*

#### 🔤 Token Frequency
- Token frequency analysis was performed to build the vocabulary.
- Special tokens like `<PAD>` and `<UNK>` were included.

### 1.3 Data Preprocessing & Dataloader

- Implemented a custom PyTorch Dataset class: `NERDataset`
- Tokenization and integer ID conversion using a dataset-specific vocabulary.
- Special Tokens: `<PAD>` for padding, `<UNK>` for unknowns.
- Tags were indexed using a tag-to-index mapping.
- Used `pad_sequence` for batch-level sequence padding.
- Attention masks were created to ignore padded tokens during training/evaluation.

---

## 🧠 2. Neural Networks

### 2.1 Model Architectures

Two models were implemented and compared:

#### 🔁 BiLSTM Model
- Embedding Layer
- Bidirectional LSTM Encoder
- Dropout
- Linear Output Layer

#### 🔀 Transformer Model
- Custom Positional Encoding
- Transformer Encoder Stack
- Linear Output Layer

---

### 2.2 Training Setup

- **Loss Function**: Cross-entropy (suitable for sequence classification).
- **Optimizer**: Adam (adaptive learning rate for faster convergence).
- **Scheduler (Transformer only)**: Warm-up + decay to stabilize early training.

#### 🧪 Evaluation Metric: Weighted F1-Score
- Balances precision and recall across classes.
- Adjusts for class imbalance using class-wise support.

📈 *![image](https://github.com/user-attachments/assets/a9e2d59d-3a99-4e51-82f8-f45c3cb7f325)*  
📈 *![image](https://github.com/user-attachments/assets/a16c86a2-07df-490b-bde4-c066ebbe5fca)*

#### Observations:

- **LSTM**:  
  - Rapid convergence.
  - Early plateauing of validation loss.  
  - High and stable F1 (>0.75) after few epochs.  
  - Minimal gap between train/val curves → Low overfitting.

- **Transformer**:  
  - Slower, steady learning.  
  - Higher training F1 (~1.0) but validation F1 plateaued (~0.85).  
  - Signs of overfitting due to model capacity.

---

### 2.3 Model Performance Summary

| Model           | Best Validation F1 |
|----------------|--------------------|
| BiLSTMNER       | ≈ 0.92             |
| TransformerNER  | ≈ 0.94             |

✅ **Best Model**: `TransformerNER` (based on validation F1).

---

## 📊 3. Model Evaluation

The `TransformerNER` model was evaluated on the **test set** using:
- Precision
- Recall
- F1-Score (Macro, Micro, and Weighted)

### 📋 Classification Report

| Label   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| I-LOC   | 0.85      | 0.91   | 0.88     | 229     |
| I-MISC  | 0.89      | 0.78   | 0.83     | 232     |
| I-ORG   | 0.83      | 0.72   | 0.77     | 279     |
| I-PER   | 0.93      | 0.78   | 0.85     | 269     |
| O       | 0.97      | 0.99   | 0.98     | 5026    |

| Metric Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Macro Avg   | 0.90      | 0.84   | 0.86     | 6035    |
| Weighted Avg| 0.96      | 0.96   | 0.96     | 6035    |

### 🎯 Final Metrics

- **Macro F1**: 0.863  
- **Micro F1**: 0.957  
- **Weighted F1**: 0.956  
- **Weighted Precision**: 0.956  
- **Weighted Recall**: 0.957  
