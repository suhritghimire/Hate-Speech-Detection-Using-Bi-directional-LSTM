# 🛡️ Hate Speech Detection: LSTM → BiLSTM → BERT

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **comparative study** of deep learning architectures for **hate speech detection** in social media text. Benchmarks LSTM, Bidirectional LSTM, and BERT on the same dataset, providing a clear progression from classical RNNs to transformer-based models.

---

## 📊 Models Compared

| Model | Architecture | Test F1 |
|-------|-------------|---------|
| LSTM | Unidirectional | — |
| Bi-LSTM | Bidirectional | — |
| BERT (fine-tuned) | Transformer | — |

---

## 📁 Structure

```
hate-speech-detection/
├── Hate_Speech_Detection_Using_LSTM.ipynb       # Baseline LSTM
├── Hate_Speech_Detection_Using_LSTM-2.ipynb     # BiLSTM
├── labeled_data.csv                             # Dataset
└── README.md
```

---

## 🗂️ Dataset

Davidson et al. (2017) Hate Speech Dataset — 24,000 tweets labeled as:
- **Hate speech**
- **Offensive language**
- **Neither**

---

## 🚀 Quickstart

Open any notebook in [Google Colab](https://colab.research.google.com/) or Jupyter and follow the cells sequentially.

---

## 📚 Reference

> Davidson, T., et al. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language*. ICWSM. [arXiv:1703.04009](https://arxiv.org/abs/1703.04009)
