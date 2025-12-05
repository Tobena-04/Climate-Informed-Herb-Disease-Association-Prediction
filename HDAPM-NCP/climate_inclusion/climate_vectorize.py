import csv
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HERB_DIR = os.path.join(DATA_DIR, "herb_kernel")
OUT_DIR = os.path.join(HERB_DIR, "climate_filtered")

# Inputs from repo
HERB_ID_CSV = os.path.join(HERB_DIR, "herb_id.csv")
HERB_CLIMATE_CSV = os.path.join(HERB_DIR, "herb_climate_data.csv")  # produced by create_kernel.py

# Herb kernel component files (as used by the paper code)
HERB_KERNEL_FILES = [
    "herb_target.txt",
    "herb_go_enrichment.txt",
    "herb_ingredient.txt",
    "herb_KEGG_enrichment.txt",
    "herb_Gene_Targets.txt",
]


def read_csv_rows(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def load_herb_order() -> Tuple[List[str], Dict[str, int]]:
    # herb_id.csv has two columns: index, HERBxxxxx
    order = []
    with open(HERB_ID_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # row[0] -> index (1-based), row[1] -> HERB id
            order.append(row[1].strip())
    index_map = {hid: i for i, hid in enumerate(order)}  # zero-based index for matrices
    return order, index_map


def derive_climate_vocab(rows: List[dict]) -> List[str]:
    # Build vocabulary of observed Köppen (and Ocean) codes from the CSV
    codes = []
    for r in rows:
        cz = r.get("climate_zone")
        if cz and cz.lower() != "unknown":
            codes.append(cz.strip())
    # stable order
    vocab = sorted(set(codes))
    return vocab


def one_hot_norm(values: List[str], categories: List[str]) -> np.ndarray:
    cnt = Counter(values)
    vec = np.array([cnt.get(cat, 0) for cat in categories], dtype=float)
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


def build_climate_vectors(rows: List[dict], herb_order: List[str], vocab: List[str]) -> Dict[str, np.ndarray]:
    by_herb: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        hid = r.get("HerbID") or r.get("herb_id")
        cz = r.get("climate_zone")
        if hid and cz and cz.lower() != "unknown":
            by_herb[hid.strip()].append(cz.strip())
    vectors = {}
    for hid in herb_order:
        vectors[hid] = one_hot_norm(by_herb.get(hid, []), vocab)
    return vectors


def cosine_kernel_from_vectors(vectors: Dict[str, np.ndarray], herb_order: List[str]) -> np.ndarray:
    X = np.stack([vectors[hid] for hid in herb_order], axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def drop_indices_square(M: np.ndarray, idxs: List[int]) -> np.ndarray:
    if not idxs:
        return M
    keep = [i for i in range(M.shape[0]) if i not in idxs]
    M2 = M[np.ix_(keep, keep)]
    return M2


def save_matrix(path: str, M: np.ndarray):
    np.savetxt(path, M, fmt="%f", delimiter=",")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load herb order and climate rows
    herb_order, index_map = load_herb_order()
    climate_rows = read_csv_rows(HERB_CLIMATE_CSV)

    # Raw set from climate CSV
    climate_ids_raw = sorted(set(r.get("HerbID") for r in climate_rows if r.get("HerbID")))
    # Only those that are part of canonical herb_id.csv (model scope)
    present_ids = [hid for hid in climate_ids_raw if hid in index_map]
    extras_out_of_scope = [hid for hid in climate_ids_raw if hid not in index_map]
    missing_ids = [hid for hid in herb_order if hid not in present_ids]
    missing_idxs = [index_map[hid] for hid in missing_ids]

    print(f"Herbs in herb_id.csv: {len(herb_order)}")
    print(f"Herbs with GBIF/climate rows (in-scope): {len(present_ids)}")
    if extras_out_of_scope:
        print(f"[INFO] {len(extras_out_of_scope)} climate HerbIDs not in herb_id.csv (excluded): {extras_out_of_scope}")
    print(f"Missing herbs (to drop): {missing_ids}")

    # 2) Filter the five herb kernel files by dropping rows/cols for missing herbs
    for fname in HERB_KERNEL_FILES:
        in_path = os.path.join(HERB_DIR, fname)
        if not os.path.exists(in_path):
            print(f"[WARN] Missing kernel file: {in_path}")
            continue
        M = np.loadtxt(in_path, dtype=float, delimiter=",")
        if M.shape[0] != M.shape[1] or M.shape[0] != len(herb_order):
            print(f"[WARN] Shape mismatch for {fname}: {M.shape}, expected ({len(herb_order)}, {len(herb_order)})")
        M2 = drop_indices_square(M, missing_idxs)
        out_path = os.path.join(OUT_DIR, fname)
        save_matrix(out_path, M2)
        print(f"Saved filtered kernel {fname} -> {out_path} with shape {M2.shape}")

    # Also emit a filtered herb_id file for convenience
    filtered_ids = [hid for hid in herb_order if hid not in missing_ids]
    with open(os.path.join(OUT_DIR, "herb_id.filtered.csv"), "w", encoding="utf-8") as f:
        for i, hid in enumerate(filtered_ids, start=1):
            f.write(f"{i},{hid}\n")
    print(f"Saved filtered herb_id list with {len(filtered_ids)} herbs")

    # 3) Build a climate-only kernel over the filtered set
    vocab = derive_climate_vocab(climate_rows)
    vectors_all = build_climate_vectors(climate_rows, herb_order, vocab)
    # restrict to filtered id order
    herb_order_filtered = filtered_ids
    K_clim_full = cosine_kernel_from_vectors(vectors_all, herb_order)
    # Drop missing herbs from climate kernel as well
    K_clim = drop_indices_square(K_clim_full, missing_idxs)
    clim_out = os.path.join(OUT_DIR, "herb_climate_kernel.txt")
    save_matrix(clim_out, K_clim)
    print(f"Saved climate kernel -> {clim_out} with shape {K_clim.shape}")

    # 4) Optional: Save the list of dropped herb indices (0-based, referring to original order)
    with open(os.path.join(OUT_DIR, "dropped_herb_indices.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(i) for i in sorted(missing_idxs)))
    print("Wrote dropped herb indices (0-based) for reproducibility")
    # Write audit of ID alignment
    with open(os.path.join(OUT_DIR, "id_alignment_audit.txt"), "w", encoding="utf-8") as f:
        f.write("Canonical herb_id.csv count: %d\n" % len(herb_order))
        f.write("Unique HerbIDs in climate CSV: %d\n" % len(climate_ids_raw))
        f.write("In-scope climate HerbIDs (kept): %d\n" % len(present_ids))
        f.write("Out-of-scope climate HerbIDs (excluded): %d\n" % len(extras_out_of_scope))
        if extras_out_of_scope:
            f.write("Excluded IDs: %s\n" % ", ".join(extras_out_of_scope))

    # 5) Also craft a filtered disease-herb association matrix (drop herb columns)
    assoc_path = os.path.join(DATA_DIR, "disease_herb", "disease_herb01.txt")
    if os.path.exists(assoc_path):
        A = np.loadtxt(assoc_path, dtype=float, delimiter=",")
        # Most repo scripts treat this as (diseases x herbs)
        if A.shape[1] != len(herb_order):
            print(f"[WARN] disease_herb01.txt columns={A.shape[1]} != herb count {len(herb_order)}")
        keep_cols = [i for i in range(A.shape[1]) if i not in missing_idxs]
        A2 = A[:, keep_cols]
        assoc_out = os.path.join(OUT_DIR, "disease_herb01.filtered.txt")
        np.savetxt(assoc_out, A2, fmt="%d", delimiter=",")
        print(f"Saved filtered association matrix -> {assoc_out} with shape {A2.shape}")
    else:
        print(f"[INFO] Association matrix not found at {assoc_path}; skipping.")


if __name__ == "__main__":
    main()