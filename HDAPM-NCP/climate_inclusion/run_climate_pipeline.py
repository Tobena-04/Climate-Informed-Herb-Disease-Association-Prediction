from pathlib import Path
from hdapm_climate import build_and_predict_with_climate
import numpy as np
from consistency_projection import NSP

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_DIR = DATA_DIR / "herb_kernel"
DISEASE_DIR = DATA_DIR / "disease_kernel"
OUT_DIR = HERB_DIR / "climate_filtered"

if __name__ == "__main__":
    # Inputs (filtered herb kernels and ids)
    herb_id_path = OUT_DIR / "herb_id.filtered.csv"
    herb_kernel_path = OUT_DIR / "herb_target.txt"
    disease_kernel_path = DISEASE_DIR / "disease_target.txt"
    assoc_path = OUT_DIR / "disease_herb01.filtered.txt"  # diseases x filtered herbs

    # Outputs
    output_herb_kernel_path = OUT_DIR / "herb_target_with_climate.txt"
    predict_out_path = DATA_DIR / "predict_with_climate_filtered.txt"

    results = build_and_predict_with_climate(
        alpha=0.5,
        use_gbif=False,
        dry_run=True,
        herb_id_path=str(herb_id_path),
        herb_kernel_path=str(herb_kernel_path),
        disease_kernel_path=str(disease_kernel_path),
        assoc_path=str(assoc_path),
        output_herb_kernel_path=str(output_herb_kernel_path),
        predict_out_path=str(predict_out_path),
        ecoregion_mapping_csv=str(HERB_DIR / 'herb_ecoregion_mapping.csv'),
        ecoregion_weight=0.5,
    )

    print("\nFused herb kernel:", results['fused_kernel_path'])
    print("Predictions:", results['prediction_path'])

    # Save baseline predictions alongside climate for custom analysis
    # Load filtered baseline herb kernel, disease kernel, and filtered assoc
    herb_base = np.loadtxt(herb_kernel_path, dtype=float, delimiter=',')
    disease_k = np.loadtxt(disease_kernel_path, dtype=float, delimiter=',')
    A = np.loadtxt(assoc_path, dtype=float, delimiter=',')
    # Orient A to herbs x diseases if it's diseases x herbs
    if A.shape == (disease_k.shape[0], herb_base.shape[0]):
        A = A.T
    # Run NSP baseline
    nsp = NSP(herb_base, disease_k, A)
    pred_base = nsp.network_NSP()
    baseline_out = OUT_DIR / "baseline_predictions.txt"
    np.savetxt(baseline_out, pred_base, fmt="%f", delimiter=',')
    print("Saved baseline predictions:", baseline_out)

    # Also save climate predictions next to baseline for convenience
    pred_clim_path_global = Path(results['prediction_path'])
    try:
        pred_clim = np.loadtxt(pred_clim_path_global, dtype=float, delimiter=',')
        climate_out = OUT_DIR / "climate_predictions.txt"
        np.savetxt(climate_out, pred_clim, fmt="%f", delimiter=',')
        print("Saved climate predictions copy:", climate_out)
    except Exception as e:
        print("[WARN] Could not copy climate predictions:", e)
