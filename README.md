# Autoencoder-Based Representation Learning (Fashion-MNIST)

## Overview
This component of the project explores how different autoencoder architectures learn latent representations of Fashion-MNIST images and evaluates how useful these representations are for downstream classification using a K-Nearest Neighbours (KNN, k=5) classifier.

The goal is to compare how architectural constraints affect the quality of learned embeddings.

---

## Models Implemented

### 1. Baseline Autoencoder
A standard fully-connected encoder–decoder network:
- Encoder: Dense(32) → Dense(16)  
- Decoder: Dense(32) → Dense(784)  
- Activation: ReLU (hidden layers), Sigmoid (output layer)  
- Loss: Binary cross-entropy  

---

### 2. Denoising Autoencoder
Same architecture as the baseline, but trained with corrupted inputs:
- Gaussian noise added (noise factor = 0.3)
- Objective: reconstruct original clean images from noisy inputs
- Encourages robustness in learned features

---

### 3. Sparse Autoencoder
Same architecture as the baseline, with sparsity constraint:
- L1 activity regularisation (lambda = 1e-5)
- Encourages sparse activations in the latent space
- Promotes compressed representations

---

## Evaluation Method
- Latent representations extracted from the encoder
- K-Nearest Neighbours classifier (k = 5) applied to embeddings
- Evaluated on Fashion-MNIST test set

---

## Results

| Model | KNN Accuracy |
|------|-------------|
| Baseline Autoencoder | 84.4% |
| Denoising Autoencoder | 82.2% |
| Sparse Autoencoder | 74.8% |

---

## Key Insight
- The baseline autoencoder produces the most discriminative latent space for KNN classification.
- Denoising improves robustness but slightly reduces class separability.
- Sparsity constraints significantly reduce representational capacity, leading to weaker classification performance.

---

## Dependencies
- TensorFlow / Keras
- NumPy
- scikit-learn
