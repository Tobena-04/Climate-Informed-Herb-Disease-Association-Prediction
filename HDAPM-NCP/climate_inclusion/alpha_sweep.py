"""Alpha sweep for climate fusion weight (alpha) to compare performance
against baseline herb kernel.

Outputs (written under data/herb_kernel/climate_filtered/metrics/):
  - alpha_metrics.csv: table of alpha, roc_auc, pr_auc, average_precision
  - alpha_trends.png: line plot of metrics vs alpha
  - curves/roc_alpha_{val}.png: ROC overlay (baseline vs alpha)
  - curves/pr_alpha_{val}.png: PR overlay (baseline vs alpha)

Prerequisites:
  Run climate_vectorize.py to produce filtered kernels & climate kernel.
  Run run_climate_pipeline.py once (optional, for climate-fused reference).

Alpha definition: fused = alpha * climate_kernel + (1-alpha) * baseline_kernel,
followed by min-max normalization (replicating fuse_kernels logic).
Baseline corresponds to alpha=0.0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, auc

try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

try:
    from consistency_projection import NSP
except ModuleNotFoundError:
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from consistency_projection import NSP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_DIR = DATA_DIR / "herb_kernel"
FILTER_DIR = HERB_DIR / "climate_filtered"
DISEASE_DIR = DATA_DIR / "disease_kernel"
METRICS_DIR = FILTER_DIR / "metrics"
CURVE_DIR = METRICS_DIR / "curves"

BASELINE_PATH = FILTER_DIR / "herb_target.txt"
CLIMATE_KERNEL_PATH = FILTER_DIR / "herb_climate_kernel.txt"  # built by climate_vectorize
DISEASE_KERNEL_PATH = DISEASE_DIR / "disease_target.txt"
ASSOC_PATH = FILTER_DIR / "disease_herb01.filtered.txt"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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
    raise ValueError(f"Association matrix shape {A.shape} incompatible with herb {h} and disease {d}")


def fuse_kernels(baseline: np.ndarray, climate: np.ndarray, alpha: float) -> np.ndarray:
    if baseline.shape != climate.shape:
        raise ValueError("Kernel shapes must match for fusion")
    fused = alpha * climate + (1 - alpha) * baseline
    mn, mx = fused.min(), fused.max()
    if mx - mn > 0:
        fused = (fused - mn) / (mx - mn)
    return fused


def run_nsp(herb_k: np.ndarray, disease_k: np.ndarray, A_hd: np.ndarray) -> np.ndarray:
    nsp = NSP(herb_k, disease_k, A_hd)
    return nsp.network_NSP()


def metric_bundle(y_true: np.ndarray, y_score: np.ndarray):
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc, ap, (precision, recall), roc_curve(y_true, y_score)


def plot_overlay(curve_dir: Path, alpha: float, base_curves, alpha_curves):
    if not HAVE_PLOT:
        return
    curve_dir.mkdir(parents=True, exist_ok=True)
    (base_fpr, base_tpr, _), (alpha_fpr, alpha_tpr, _) = base_curves[2], alpha_curves[2]
    plt.figure()
    plt.plot(base_fpr, base_tpr, label=f"Baseline ROC")
    plt.plot(alpha_fpr, alpha_tpr, label=f"Alpha {alpha:.1f} ROC")
    plt.plot([0,1],[0,1],'k--',linewidth=0.7)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Overlay α={alpha:.1f}')
    plt.legend(); plt.tight_layout()
    plt.savefig(curve_dir / f"roc_alpha_{alpha:.1f}.png", dpi=180)
    plt.close()
    (base_prec, base_rec), (alpha_prec, alpha_rec) = base_curves[1], alpha_curves[1]
    plt.figure()
    plt.step(base_rec, base_prec, where='post', label='Baseline PR')
    plt.step(alpha_rec, alpha_prec, where='post', label=f'Alpha {alpha:.1f} PR')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Overlay α={alpha:.1f}')
    plt.legend(); plt.tight_layout()
    plt.savefig(curve_dir / f"pr_alpha_{alpha:.1f}.png", dpi=180)
    plt.close()


def plot_trends(metrics_df: pd.DataFrame, out_path: Path):
    if not HAVE_PLOT:
        return
    plt.figure(figsize=(7.5,4.5))
    plt.plot(metrics_df['alpha'], metrics_df['roc_auc'], marker='o', linewidth=2.2, label='ROC AUC', color='#1f77b4')
    # Emphasize PR AUC line so it is visible even when close to AP
    plt.plot(metrics_df['alpha'], metrics_df['pr_auc'], marker='s', linewidth=2.6, label='PR AUC', color='#d62728')
    # Use dashed line for Avg Precision to distinguish from PR AUC
    plt.plot(metrics_df['alpha'], metrics_df['average_precision'], marker='^', linewidth=1.8, linestyle='--', label='Avg Precision', color='#2ca02c')
    plt.xlabel('Alpha (climate weight)')
    plt.ylabel('Metric value')
    plt.title('Metric trends vs alpha')
    plt.grid(alpha=0.3)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    disease_k = load_matrix(DISEASE_KERNEL_PATH)
    baseline_k = load_matrix(BASELINE_PATH)
    climate_k = load_matrix(CLIMATE_KERNEL_PATH)
    A = load_matrix(ASSOC_PATH)
    A_hd = ensure_A_herb_by_disease(A, disease_k, baseline_k)

    y_true = A_hd.flatten()

    records = []
    all_curves = []  # collect for combined plots
    # Precompute baseline curves (alpha=0)
    pred_base = run_nsp(baseline_k, disease_k, A_hd)
    roc_auc_b, pr_auc_b, ap_b, (prec_b, rec_b), roc_tuple_b = metric_bundle(y_true, pred_base.flatten())
    baseline_curves = (roc_auc_b, pr_auc_b, ap_b, (prec_b, rec_b), roc_tuple_b)
    records.append({'alpha': 0.0, 'roc_auc': roc_auc_b, 'pr_auc': pr_auc_b, 'average_precision': ap_b})
    base_fpr, base_tpr, _ = roc_tuple_b
    all_curves.append({
        'alpha': 0.0,
        'roc': (base_fpr, base_tpr),
        'pr': (prec_b, rec_b)
    })

    # Iterate alphas > 0
    for alpha in ALPHAS[1:]:
        fused = fuse_kernels(baseline_k, climate_k, alpha)
        pred = run_nsp(fused, disease_k, A_hd)
        roc_auc_a, pr_auc_a, ap_a, (prec_a, rec_a), roc_tuple_a = metric_bundle(y_true, pred.flatten())
        records.append({'alpha': alpha, 'roc_auc': roc_auc_a, 'pr_auc': pr_auc_a, 'average_precision': ap_a})
        plot_overlay(CURVE_DIR, alpha, (roc_auc_b, (prec_b, rec_b), roc_tuple_b), (roc_auc_a, (prec_a, rec_a), roc_tuple_a))
        a_fpr, a_tpr, _ = roc_tuple_a
        all_curves.append({
            'alpha': alpha,
            'roc': (a_fpr, a_tpr),
            'pr': (prec_a, rec_a)
        })

    df = pd.DataFrame(records)
    df.sort_values('alpha', inplace=True)
    df.to_csv(METRICS_DIR / 'alpha_metrics.csv', index=False)
    if HAVE_PLOT:
        plot_trends(df, METRICS_DIR / 'alpha_trends.png')
        # Combined plots of all ROC/PR curves
        CURVE_DIR.mkdir(parents=True, exist_ok=True)
        # Sort by alpha for consistent coloring
        all_curves_sorted = sorted(all_curves, key=lambda x: x['alpha'])
        # Build a color map over alphas
        import matplotlib.cm as cm
        import numpy as np
        alphas_vals = np.array([c['alpha'] for c in all_curves_sorted])
        norm = (alphas_vals - alphas_vals.min()) / (alphas_vals.max() - alphas_vals.min() + 1e-12)
        colors = [cm.viridis(v) for v in norm]

        # ROC combined
        plt.figure(figsize=(6.5,6))
        for c, col in zip(all_curves_sorted, colors):
            fpr, tpr = c['roc']
            label = f"alpha={c['alpha']:.1f}"
            lw = 2.0 if c['alpha'] == 0.0 else 1.6
            plt.plot(fpr, tpr, label=label, color=col, linewidth=lw)
        plt.plot([0,1],[0,1],'k--', linewidth=0.7)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves for All Alphas')
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(CURVE_DIR / 'all_roc_curves.png', dpi=200)
        plt.close()

        # PR combined
        plt.figure(figsize=(6.5,6))
        for c, col in zip(all_curves_sorted, colors):
            prec, rec = c['pr']
            label = f"alpha={c['alpha']:.1f}"
            lw = 2.0 if c['alpha'] == 0.0 else 1.6
            plt.step(rec, prec, where='post', label=label, color=col, linewidth=lw)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curves for All Alphas')
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(CURVE_DIR / 'all_pr_curves.png', dpi=200)
        plt.close()

    # Write summary JSON
    best_row = df.loc[df['roc_auc'].idxmax()].to_dict()
    summary = {
        'alphas_tested': ALPHAS,
        'best_by_roc_auc': best_row,
        'metrics_csv': str(METRICS_DIR / 'alpha_metrics.csv'),
        'trend_plot': str(METRICS_DIR / 'alpha_trends.png') if HAVE_PLOT else None
    }
    with open(METRICS_DIR / 'alpha_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
