# Install dependencies first (numpy, pandas, scikit-learn)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pickle

# Load data

def load_fashion_mnist_csv(train_path="fashion-mnist_train.csv",
                            test_path="fashion-mnist_test.csv"):
    """
    Load Fashion-MNIST from CSV files.
    Column 0 = label, columns 1 to 784 = pixel values.
    """
    print("[1] Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_raw = train_df.iloc[:, 1:].values.astype(np.float32) # (60000, 784)
    y_train = train_df.iloc[:, 0].values.astype(np.int64) # (60000,)

    X_test_raw = test_df.iloc[:, 1:].values.astype(np.float32) # (10000, 784)
    y_test = test_df.iloc[:, 0].values.astype(np.int64) # (10000,)

    print(f"  Training set : {X_train_raw.shape}, labels {y_train.shape}")
    print(f"  Test set     : {X_test_raw.shape},  labels {y_test.shape}")
    return X_train_raw, y_train, X_test_raw, y_test

# Validation Split

def validation_split(X_train_raw, y_train, val_size=0.1, random_state=42):
    """
    Stratified 90/10 train/validation split to formally gurantee class split.
    """
    print("[2] Creating stratified validation split (90/10)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state
    )
    print(f"  Train : {X_tr.shape}  |  Val : {X_val.shape}")
    return X_tr, y_tr, X_val, y_val

# Classical Machine Learning pipeline

def preprocess_classical_ml(X_tr_norm, y_tr, X_val_norm, y_val,
                             X_test_norm, y_test,
                             apply_pca=True, pca_variance=0.95):
    """
    For Logistic Regression, SVM and Random Forest:
      a) StandardScaler: for zero mean, unit variance. Fitted on train only.
         Critical for LR and SVM + applied to RF for fair comparison.
      b) PCA (optional): reduces 784 -> ~330 dimensions retaining 95% variance
    """
    print("\n[3] Classical ML pipeline...")

    # StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_norm)
    X_val_scaled = scaler.transform(X_val_norm)
    X_test_scaled = scaler.transform(X_test_norm)
    print(f"  After scaling — Train: {X_tr_scaled.shape}")

    # PCA
    if apply_pca:
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"  PCA: {X_tr_scaled.shape[1]} → {X_tr_pca.shape[1]} dims "
              f"({pca_variance*100:.0f}% variance retained)")
        return {
            "X_train": X_tr_pca,   "y_train": y_tr,
            "X_val":   X_val_pca,  "y_val":   y_val,
            "X_test":  X_test_pca, "y_test":  y_test,
            "scaler":  scaler,     "pca":      pca
        }
    else:
        return {
            "X_train": X_tr_scaled,   "y_train": y_tr,
            "X_val":   X_val_scaled,  "y_val":   y_val,
            "X_test":  X_test_scaled, "y_test":  y_test,
            "scaler":  scaler,         "pca":    None
        }

# Reshape and Standardize

def reshape_and_standardize(X_tr_norm, X_val_norm, X_test_norm):
    """
    For CNN and ViT pipelines:
    a) Reshape (N, 784) -> (N, 1, 28, 28)
    b) Compute per-channel mean and std from training split
    c) Standardise all splits
    """
    X_tr = X_tr_norm.reshape(-1, 1, 28, 28)
    X_val = X_val_norm.reshape(-1, 1, 28, 28)
    X_te = X_test_norm.reshape(-1, 1, 28, 28)

    mean = X_tr.mean(axis=(0, 2, 3), keepdims=True) # (1, 1, 1, 1)
    std = X_tr.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std == 0, 1.0, std)

    print(f"  Channel mean={mean.squeeze():.4f}, std={std.squeeze():.4f}")
 
    X_tr = (X_tr  - mean) / std
    X_val = (X_val - mean) / std
    X_te = (X_te  - mean) / std
 
    return X_tr, X_val, X_te, mean, std

# CNN pipeline


def preprocess_cnn(X_tr_norm, y_tr, X_val_norm, y_val,
                   X_test_norm, y_test):
    """
    For CNN:
      a) Reshape and standardize via reshape_and_standardize()
    """
    print("\n[4] CNN pipeline...")
    X_tr, X_val, X_te, mean, std = reshape_and_standardize(
        X_tr_norm, X_val_norm, X_test_norm
    )
    print(f"  Output shape — Train: {X_tr.shape}")
    return {
        "X_train": X_tr,  "y_train": y_tr,
        "X_val":   X_val, "y_val":   y_val,
        "X_test":  X_te,  "y_test":  y_test,
        "mean":    mean,  "std":     std
    }


# ViT pipeline


def preprocess_vit(X_tr_norm, y_tr, X_val_norm, y_val,
                   X_test_norm, y_test,
                   patch_size=7, target_size=28):
    """
    For ViT:
      a) Reshape and standardize via reshape_and_standardize()
      b) Patch configuration with patch_size=7 to yield 16 non-overlapping
         patches per image.
    """
    print("\n[5] ViT pipeline...")
    assert target_size % patch_size == 0, \
    f"target_size ({target_size}) must be divisible by patch_size ({patch_size})"
    X_tr, X_val, X_te, mean, std = reshape_and_standardize(
        X_tr_norm, X_val_norm, X_test_norm
    )
    num_patches = (target_size // patch_size) ** 2
    print(f"  ViT config : image={target_size}x{target_size}, "
          f"patch={patch_size}x{patch_size}, num_patches={num_patches}")
    print(f"  Output shape — Train: {X_tr.shape}")
 
    return {
        "X_train":     X_tr,        "y_train":     y_tr,
        "X_val":       X_val,       "y_val":       y_val,
        "X_test":      X_te,        "y_test":      y_test,
        "mean":        mean,        "std":         std,
        "patch_size":  patch_size,  "num_patches": num_patches
    }

# Save preprocessed data

def save_data(data, filename):
    os.makedirs("preprocessed", exist_ok=True)
    path = os.path.join("preprocessed", filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved → {path}")

# Main

if __name__ == "__main__":
    # Load
    X_train_raw, y_train, X_test_raw, y_test = load_fashion_mnist_csv()

    # Validation split
    X_tr, y_tr, X_val, y_val = validation_split(X_train_raw, y_train)

    # Normalize
    X_tr_norm = X_tr / 255.0
    X_val_norm = X_val / 255.0
    X_test_norm = X_test_raw / 255.0

    # Classical ML
    classical_data = preprocess_classical_ml(
        X_tr_norm, y_tr, X_val_norm, y_val, X_test_norm, y_test,
        apply_pca=True, pca_variance=0.95
    )
    save_data(classical_data, "classical_ml.pkl")

    # CNN
    cnn_data = preprocess_cnn(
        X_tr_norm, y_tr, X_val_norm, y_val, X_test_norm, y_test
    )
    save_data(cnn_data, "cnn.pkl")

    # ViT
    vit_data = preprocess_vit(
        X_tr_norm, y_tr, X_val_norm, y_val, X_test_norm, y_test,
        patch_size=7, target_size=28
    )
    save_data(vit_data, "vit.pkl")

    print("\n Preprocessing complete. Files saved to ./preprocessed/")
