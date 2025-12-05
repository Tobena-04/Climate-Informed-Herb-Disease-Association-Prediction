"""
Evaluate baseline vs climate-fused predictions on the filtered herb set.

Inputs (default locations):
- data/herb_kernel/climate_filtered/herb_target.txt                 (baseline herb kernel, filtered)
- data/herb_kernel/climate_filtered/herb_target_with_climate.txt    (climate-fused herb kernel)
- data/disease_kernel/disease_target.txt                            (disease kernel)
- data/herb_kernel/climate_filtered/disease_herb01.filtered.txt     (association matrix: diseases x herbs)
- data/herb_kernel/climate_filtered/herb_id.filtered.csv            (filtered herb ids, for info only)

Outputs:
- data/herb_kernel/climate_filtered/metrics/metrics.json
- ROC/PR curve images for baseline and climate (if matplotlib available)

This script does not hardcode herb count; it uses filtered shapes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

# NSP import path handling (like in hdapm_climate.py)
try:
    from consistency_projection import NSP
except ModuleNotFoundError:
    import sys
    CURRENT = Path(__file__).resolve()
    ROOT = CURRENT.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from consistency_projection import NSP

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_DIR = DATA_DIR / "herb_kernel"
FILTER_DIR = HERB_DIR / "climate_filtered"
DISEASE_DIR = DATA_DIR / "disease_kernel"
METRICS_DIR = FILTER_DIR / "metrics"

BASELINE_HERB = FILTER_DIR / "herb_target.txt"
CLIMATE_HERB = FILTER_DIR / "herb_target_with_climate.txt"
DISEASE_KERNEL = DISEASE_DIR / "disease_target.txt"
ASSOC = FILTER_DIR / "disease_herb01.filtered.txt"


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.loadtxt(path, dtype=float, delimiter=',')


def ensure_A_herb_by_disease(A: np.ndarray, disease_k: np.ndarray, herb_k: np.ndarray) -> np.ndarray:
    h = herb_k.shape[0]
    d = disease_k.shape[0]
    if A.shape == (d, h):
        return A.T
    if A.shape == (h, d):
        return A
    raise ValueError(f"Association matrix shape {A.shape} not compatible with herb {h} and disease {d}")


def run_nsp(herb_k: np.ndarray, disease_k: np.ndarray, A_hd: np.ndarray) -> np.ndarray:
    nsp = NSP(herb_k, disease_k, A_hd)
    return nsp.network_NSP()


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray):
    # y_true, y_score flattened 1D arrays
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_score)
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'average_precision': float(ap),
        'positives': int(np.sum(y_true == 1)),
        'negatives': int(np.sum(y_true == 0)),
        'n': int(y_true.size)
    }


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, title_prefix: str, out_dir: Path):
    if not HAVE_PLOT:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val = auc(recall, precision)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc_val:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f"{title_prefix} ROC")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_roc.png", dpi=200)
    plt.close()

    plt.figure()
    plt.step(recall, precision, where='post', label=f"PR AUC={pr_auc_val:.4f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f"{title_prefix} PR")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_pr.png", dpi=200)
    plt.close()


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    disease_k = load_matrix(DISEASE_KERNEL)
    A = load_matrix(ASSOC)

    results = {}

    # Baseline
    if BASELINE_HERB.exists():
        herb_base = load_matrix(BASELINE_HERB)
        A_hd = ensure_A_herb_by_disease(A, disease_k, herb_base)
        pred_base = run_nsp(herb_base, disease_k, A_hd)
        y_true = A_hd.flatten()
        y_score = pred_base.flatten()
        metrics_base = compute_metrics(y_true, y_score)
        results['baseline'] = metrics_base
        plot_curves(y_true, y_score, 'Baseline', METRICS_DIR)
        np.savetxt(FILTER_DIR / 'baseline_predictions.txt', pred_base, fmt='%f', delimiter=',')
    else:
        results['baseline'] = {'error': f"Missing {BASELINE_HERB}"}

    # Climate-fused
    if CLIMATE_HERB.exists():
        herb_clim = load_matrix(CLIMATE_HERB)
        A_hd = ensure_A_herb_by_disease(A, disease_k, herb_clim)
        pred_clim = run_nsp(herb_clim, disease_k, A_hd)
        y_true = A_hd.flatten()
        y_score = pred_clim.flatten()
        metrics_clim = compute_metrics(y_true, y_score)
        results['climate'] = metrics_clim
        plot_curves(y_true, y_score, 'Climate', METRICS_DIR)
        np.savetxt(FILTER_DIR / 'climate_predictions.txt', pred_clim, fmt='%f', delimiter=',')
    else:
        results['climate'] = {'error': f"Missing {CLIMATE_HERB}. Run run_climate_pipeline.py"}

    with open(METRICS_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
