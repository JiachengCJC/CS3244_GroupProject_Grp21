
import argparse
import csv
import json
import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    top_k_accuracy_score
)

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 0
N_SPLITS     = 5
PCA_VAR      = 0.95

CLASS_NAMES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# ─── Data ─────────────────────────────────────────────────────────────────────
def _subsample(X, y, max_samples, seed_offset: int = 0):
    """Optional stratified subsample for faster smoke tests."""
    if max_samples is None or max_samples >= len(y):
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=max_samples,
        stratify=y,
        random_state=RANDOM_STATE + seed_offset
    )
    return X_sub, y_sub


def load_data(max_train_samples=None, max_test_samples=None):
    """Download Fashion-MNIST from OpenML; split into train/val/test."""
    print("Loading Fashion-MNIST from OpenML …")
    ds = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    X = ds.data.astype(np.float32) / 255.0   # Min-Max normalisation
    y = ds.target.astype(int)

    # Standard 60k/10k split (as shipped by Zalando)
    X_tr_full, X_te = X[:60000], X[60000:]
    y_tr_full, y_te = y[:60000], y[60000:]

    # 90/10 stratified train/val
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full,
        test_size=0.10, stratify=y_tr_full,
        random_state=RANDOM_STATE
    )

    X_tr, y_tr = _subsample(X_tr, y_tr, max_train_samples, seed_offset=1)
    X_te, y_te = _subsample(X_te, y_te, max_test_samples, seed_offset=2)

    print(f"  Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")
    return X_tr, y_tr, X_te, y_te


# ─── Pipeline factory ─────────────────────────────────────────────────────────
def make_pipeline(clf, use_pca: bool) -> Pipeline:
    """
    Build a leak-free Pipeline.
    StandardScaler and PCA are fitted *inside* each CV fold,
    preventing validation-set statistics from contaminating the scaler/PCA fit.
    """
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)))
    steps.append(('clf', clf))
    return Pipeline(steps)


# ─── Cross-validation ─────────────────────────────────────────────────────────
def run_cv(pipeline, X, y, label: str):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    t0 = time.time()
    scores = cross_val_score(pipeline, X, y, cv=skf,
                             scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - t0
    print(f"    [{label}] CV={scores.mean():.4f}±{scores.std():.4f}  "
          f"folds={np.round(scores, 4)}  ({elapsed:.1f}s total)")
    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'folds': [float(v) for v in scores],
        'time_sec': float(elapsed),
    }


# ─── Full train + test evaluation ────────────────────────────────────────────
def evaluate(pipeline, X_tr, y_tr, X_te, y_te, name: str):
    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = pipeline.predict(X_te)
    t_infer = time.time() - t0

    acc = accuracy_score(y_te, y_pred)

    # Top-k accuracy
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        scores_mat = pipeline.predict_proba(X_te)
    else:                                       # SVM: use decision_function
        scores_mat = pipeline.decision_function(X_te)
    top3 = top_k_accuracy_score(y_te, scores_mat, k=3)
    top5 = top_k_accuracy_score(y_te, scores_mat, k=5)

    cls_text = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES, digits=4
    )
    cls_dict = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
    )

    print(f"    Test acc={acc:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}  "
          f"Train={t_train:.1f}s  Infer={t_infer:.3f}s")
    print(cls_text)
    return {
        'acc': acc, 'top3': top3, 'top5': top5,
        't_train': t_train, 't_infer': t_infer,
        'macro_precision': float(cls_dict['macro avg']['precision']),
        'macro_recall': float(cls_dict['macro avg']['recall']),
        'macro_f1': float(cls_dict['macro avg']['f1-score']),
        'y_pred': y_pred,
        'class_report': cls_dict,
    }


# ─── Complexity stats ─────────────────────────────────────────────────────────
def complexity_stats(pipeline, name: str) -> dict:
    stats = {}
    clf = pipeline.named_steps['clf']

    if 'pca' in pipeline.named_steps:
        stats['pca_dims'] = int(pipeline.named_steps['pca'].n_components_)

    if isinstance(clf, LogisticRegression):
        stats['n_params']   = int(clf.coef_.size + clf.intercept_.size)
        stats['coef_shape'] = list(clf.coef_.shape)   # (10, n_pca)
        stats['n_iter']     = int(clf.n_iter_[0])

    elif isinstance(clf, SVC):
        stats['n_support_vectors'] = int(clf.n_support_.sum())
        stats['sv_per_class']      = clf.n_support_.tolist()

    elif isinstance(clf, RandomForestClassifier):
        depths = [e.get_depth() for e in clf.estimators_]
        nodes  = [e.tree_.node_count for e in clf.estimators_]
        stats['n_trees']      = int(clf.n_estimators)
        stats['mean_depth']   = float(np.mean(depths))
        stats['max_depth']    = int(np.max(depths))
        stats['total_nodes']  = int(np.sum(nodes))
        stats['mean_nodes']   = float(np.mean(nodes))

    print(f"    Complexity: {stats}")
    return stats


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, name: str, save_path: str):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_misclassified(X_te, y_true, y_pred,
                       class_a: int, class_b: int,
                       save_path: str, n: int = 12):
    """
    Show n examples where true label=class_a but predicted=class_b.
    Helps diagnose whether errors stem from pixel-level ambiguity or label noise.
    """
    mask = (y_true == class_a) & (y_pred == class_b)
    idx  = np.where(mask)[0][:n]
    if len(idx) == 0:
        print(f"    No {CLASS_NAMES[class_a]}→{CLASS_NAMES[class_b]} errors found.")
        return

    cols = min(6, len(idx))
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(idx):
            ax.imshow(X_te[idx[i]].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'True: {CLASS_NAMES[class_a]}\nPred: {CLASS_NAMES[class_b]}',
                         fontsize=7)
        ax.axis('off')
    plt.suptitle(
        f'Misclassified samples: {CLASS_NAMES[class_a]} → {CLASS_NAMES[class_b]}'
        f'  (n={len(idx)} shown)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_rf_importance(pipeline, save_path: str):
    clf    = pipeline.named_steps['clf']
    imp    = clf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:30]
    plt.figure(figsize=(12, 4))
    plt.bar(range(30), imp[top_idx], color='steelblue')
    plt.xlabel('PCA Component Rank')
    plt.ylabel('Gini Importance')
    plt.title('Random Forest — Top 30 PCA Component Importances')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def top_confusion_pairs(y_true, y_pred, top_n: int = 3):
    """Return the top off-diagonal confusion pairs for microscopic analysis."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm.ravel())[::-1]

    pairs = []
    for flat_idx in order:
        i, j = np.unravel_index(flat_idx, cm.shape)
        count = int(cm[i, j])
        if count == 0:
            break
        pairs.append({
            'true_idx': int(i),
            'pred_idx': int(j),
            'true_class': CLASS_NAMES[i],
            'pred_class': CLASS_NAMES[j],
            'count': count,
        })
        if len(pairs) == top_n:
            break
    return pairs


def _fmt_float(v):
    return "" if v is None else f"{float(v):.6f}"


def save_summary_files(summary: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'classical_summary.json')
    csv_path = os.path.join(output_dir, 'classical_summary.csv')
    complexity_csv = os.path.join(output_dir, 'classical_complexity.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_fields = [
        'model',
        'cv_pca_mean', 'cv_pca_std', 'cv_pca_time_sec',
        'cv_nopca_mean', 'cv_nopca_std', 'cv_nopca_time_sec',
        'test_acc_pca', 'test_acc_nopca',
        'top3', 'top5',
        'macro_precision', 'macro_recall', 'macro_f1',
        'train_time_pca_sec', 'train_time_nopca_sec', 'infer_time_pca_sec',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for model_name, r in summary.items():
            writer.writerow({
                'model': model_name,
                'cv_pca_mean': _fmt_float(r['cv_pca_mean']),
                'cv_pca_std': _fmt_float(r['cv_pca_std']),
                'cv_pca_time_sec': _fmt_float(r.get('cv_pca_time_sec')),
                'cv_nopca_mean': _fmt_float(r.get('cv_nopca_mean')),
                'cv_nopca_std': _fmt_float(r.get('cv_nopca_std')),
                'cv_nopca_time_sec': _fmt_float(r.get('cv_nopca_time_sec')),
                'test_acc_pca': _fmt_float(r.get('test_acc')),
                'test_acc_nopca': _fmt_float(r.get('nopca_test_acc')),
                'top3': _fmt_float(r.get('top3')),
                'top5': _fmt_float(r.get('top5')),
                'macro_precision': _fmt_float(r.get('macro_precision')),
                'macro_recall': _fmt_float(r.get('macro_recall')),
                'macro_f1': _fmt_float(r.get('macro_f1')),
                'train_time_pca_sec': _fmt_float(r.get('t_train')),
                'train_time_nopca_sec': _fmt_float(r.get('t_nopca_train')),
                'infer_time_pca_sec': _fmt_float(r.get('t_infer')),
            })

    with open(complexity_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'metric', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, r in summary.items():
            comp = r.get('complexity', {})
            for metric, value in comp.items():
                writer.writerow({
                    'model': model_name,
                    'metric': metric,
                    'value': value,
                })

    print(f"\nSaved summary JSON: {json_path}")
    print(f"Saved summary CSV:  {csv_path}")
    print(f"Saved complexity:   {complexity_csv}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(no_plots: bool = False, output_dir: str = 'artifacts/classical_ml',
         max_train_samples=None, max_test_samples=None):
    X_tr, y_tr, X_te, y_te = load_data(
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples
    )
    os.makedirs(output_dir, exist_ok=True)

    # Model definitions (hyperparameters chosen after manual sweep)
    model_defs = {
        'Logistic Regression': LogisticRegression(
            # C grid tried: [0.01, 0.1, 1.0, 10.0] → C=1.0 best CV
            solver='saga', C=1.0, max_iter=1000,
            random_state=RANDOM_STATE, n_jobs=-1),

        'SVM (RBF)': SVC(
            # C grid tried: [1, 10, 100]; gamma: ['scale','auto',0.001]
            # C=10, gamma='scale' gave best CV across 3 candidates
            kernel='rbf', C=10.0, gamma='scale',
            decision_function_shape='ovr',
            random_state=RANDOM_STATE),

        'Random Forest': RandomForestClassifier(
            # n_estimators: [100,200,300]; max_features: ['sqrt','log2']
            # 300 trees / sqrt features gave best CV with diminishing returns >300
            n_estimators=300, max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    summary = {}

    for name, clf_def in model_defs.items():
        print(f"\n{'='*65}\n  {name}\n{'='*65}")

        # ── 1. CV with Pipeline (leak-free) ──────────────────────────────
        print("  Cross-validation (PCA setting, leak-free Pipeline):")
        pipe_cv_pca = make_pipeline(clone(clf_def), use_pca=True)
        cv_pca = run_cv(pipe_cv_pca, X_tr, y_tr, 'PCA')

        # ── 2. Ablation: No-PCA ──────────────────────────────────────────
        print("  Cross-validation (No-PCA ablation):")
        if name == 'SVM (RBF)':
            # RBF-SVM on 784 features × 54k samples is O(n²·d) — infeasible.
            # Estimated from a 5k-sample pilot: ~4h on this hardware.
            cv_nopca_mean, cv_nopca_std = None, None
            cv_nopca_time_sec = None
            nopca_test_acc, t_nopca_train = None, None
            print("    [No-PCA] Skipped: O(n²·d) cost with d=784 intractable "
                  "(estimated >4h from 5k pilot)")
        else:
            pipe_cv_nopca = make_pipeline(clone(clf_def), use_pca=False)
            cv_nopca = run_cv(pipe_cv_nopca, X_tr, y_tr, 'No-PCA')
            cv_nopca_mean = cv_nopca['mean']
            cv_nopca_std = cv_nopca['std']
            cv_nopca_time_sec = cv_nopca['time_sec']
            # Full train + test for no-PCA
            pipe_nopca = make_pipeline(clone(clf_def), use_pca=False)
            t0 = time.time()
            pipe_nopca.fit(X_tr, y_tr)
            t_nopca_train = time.time() - t0
            nopca_test_acc = accuracy_score(y_te, pipe_nopca.predict(X_te))
            print(f"    [No-PCA] Test acc={nopca_test_acc:.4f}  Train={t_nopca_train:.1f}s")

        # ── 3. Full train + test evaluation (PCA setting) ───────────────
        print("  Test evaluation (PCA setting):")
        pipe_final = make_pipeline(clone(clf_def), use_pca=True)
        res = evaluate(pipe_final, X_tr, y_tr, X_te, y_te, name)

        # ── 4. Complexity ────────────────────────────────────────────────
        print("  Complexity statistics:")
        comp = complexity_stats(pipe_final, name)
        hardest_pairs = top_confusion_pairs(y_te, res['y_pred'], top_n=3)
        print(f"  Hardest confusion pairs: {hardest_pairs}")

        # ── 5. Plots ─────────────────────────────────────────────────────
        if not no_plots:
            safe = name.replace(' ', '_').replace('(', '').replace(')', '')
            plot_confusion(
                y_te, res['y_pred'], name,
                os.path.join(output_dir, f'cm_{safe}.png')
            )
            # Shirt(6)→T-shirt/Top(0), Shirt(6)→Pullover(2), Pullover(2)→Coat(4)
            for ca, cb in [(6, 0), (6, 2), (2, 4)]:
                plot_misclassified(X_te, y_te, res['y_pred'], ca, cb,
                                   os.path.join(
                                       output_dir,
                                       f'misc_{safe}_{CLASS_NAMES[ca]}_{CLASS_NAMES[cb]}.png'
                                       .replace('/', '-')
                                   ))
            if name == 'Random Forest':
                plot_rf_importance(
                    pipe_final,
                    os.path.join(output_dir, 'rf_feature_importance.png')
                )

        summary[name] = {
            'cv_pca_mean': cv_pca['mean'], 'cv_pca_std': cv_pca['std'],
            'cv_pca_folds': cv_pca['folds'], 'cv_pca_time_sec': cv_pca['time_sec'],
            'cv_nopca_mean': cv_nopca_mean, 'cv_nopca_std': cv_nopca_std,
            'cv_nopca_time_sec': cv_nopca_time_sec,
            'test_acc': res['acc'],
            'nopca_test_acc': nopca_test_acc,
            't_nopca_train': t_nopca_train,
            'top3': res['top3'], 'top5': res['top5'],
            'macro_precision': res['macro_precision'],
            'macro_recall': res['macro_recall'],
            'macro_f1': res['macro_f1'],
            't_train': res['t_train'], 't_infer': res['t_infer'],
            'hardest_pairs': hardest_pairs,
            'complexity': comp,
        }

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    fmt = f"{{:<25}} {{:>14}} {{:>14}} {{:>8}} {{:>8}} {{:>8}} {{:>9}} {{:>9}}"
    print(fmt.format('Model', 'CV(PCA)', 'CV(No-PCA)',
                     'TestPCA', 'TestNoPCA', 'Top-3', 'Top-5', 'Train(s)'))
    print("-" * 80)
    for name, r in summary.items():
        pca_str   = f"{r['cv_pca_mean']:.4f}±{r['cv_pca_std']:.4f}"
        nopca_str = (f"{r['cv_nopca_mean']:.4f}±{r['cv_nopca_std']:.4f}"
                     if r['cv_nopca_mean'] is not None else "N/A (SVM)")
        nptest    = (f"{r['nopca_test_acc']:.4f}"
                     if r['nopca_test_acc'] is not None else "N/A")
        print(fmt.format(name, pca_str, nopca_str,
                         f"{r['test_acc']:.4f}", nptest,
                         f"{r['top3']:.4f}", f"{r['top5']:.4f}",
                         f"{r['t_train']:.1f}"))
    print("=" * 80)
    save_summary_files(summary, output_dir)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion-MNIST classical ML benchmark')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip saving matplotlib figures')
    parser.add_argument(
        '--output-dir', default='artifacts/classical_ml',
        help='Directory to save figures and summary tables'
    )
    parser.add_argument(
        '--max-train-samples', type=int, default=None,
        help='Optional stratified cap on training samples (for quick smoke tests)'
    )
    parser.add_argument(
        '--max-test-samples', type=int, default=None,
        help='Optional stratified cap on test samples (for quick smoke tests)'
    )
    args = parser.parse_args()
    main(
        no_plots=args.no_plots,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
