"""
hdapm_climate.py

Utilities to integrate GBIF occurrence-derived climate/geographic similarity
into the herb kernel and run the HDAPM-NCP network consistency projection.

This script provides functions but does not call external APIs automatically
unless explicitly requested by the user (to avoid unexpected network actions).

Main capabilities:
- load existing herb/disease kernels and association matrix
- compute a climate/geographic similarity kernel from occurrence coordinates
  (GBIF) or from a provided coordinates file
- fuse the climate kernel into the original herb kernel (weighted average)
- run the NSP projection and save prediction matrix
- optional: evaluate using existing sampling/ROC utilities from this repo

Usage (example):

from hdapm_climate import build_and_predict_with_climate
build_and_predict_with_climate(alpha=0.5, use_gbif=False)

"""

from pathlib import Path
import numpy as np
import pandas as pd
import math
import json

# Try to import pygbif if available; but do not fail if missing
try:
    from pygbif import occurrences
    PYG_BIF_AVAILABLE = True
except Exception:
    PYG_BIF_AVAILABLE = False

# For projection, import local NSP (root consistency_projection) with fallback for subdirectory execution
try:
    from consistency_projection import NSP
except ModuleNotFoundError:
    import sys
    parent_root = Path(__file__).resolve().parent.parent
    if str(parent_root) not in sys.path:
        sys.path.insert(0, str(parent_root))
    from consistency_projection import NSP

"""Adjust root resolution: script lives in climate_inclusion/, actual data resides one level up."""
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_KERNEL_DIR = DATA_DIR / "herb_kernel"
DISEASE_KERNEL_DIR = DATA_DIR / "disease_kernel"
ASSOC_DIR = DATA_DIR / "disease_herb"

# Helpful constants
DEFAULT_GBIF_LIMIT = 200
EARTH_RADIUS_KM = 6371.0


def haversine(lon1, lat1, lon2, lat2):
    # all args in degrees
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def pairwise_geo_distance_matrix(coords):
    # coords: list of (lon, lat) or (lat, lon) depending on convention; here use (lon, lat)
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        lon1, lat1 = coords[i]
        for j in range(i, n):
            lon2, lat2 = coords[j]
            d = haversine(lon1, lat1, lon2, lat2)
            D[i, j] = d
            D[j, i] = d
    return D


def distance_to_similarity(dist_matrix, sigma=None):
    # Convert distances (km) to similarity using RBF: exp(-(d/sigma)^2)
    if sigma is None:
        # heuristic: median distance
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        upper = upper[upper > 0]
        if len(upper) == 0:
            sigma = 1.0
        else:
            sigma = np.median(upper)
            if sigma == 0:
                sigma = 1.0
    S = np.exp(- (dist_matrix / sigma) ** 2)
    np.fill_diagonal(S, 1.0)
    return S


def load_herb_ids(herb_id_path=None):
    # load herb ID CSV used in repository (herb_id.csv)
    path = herb_id_path or (HERB_KERNEL_DIR / 'herb_id.csv')
    if not Path(path).exists():
        raise FileNotFoundError(f"Herb id file not found: {path}")
    df = pd.read_csv(path, header=None)
    # assume second column is herb identifier or scientific name
    if df.shape[1] >= 2:
        names = df.iloc[:, 1].astype(str).tolist()
    else:
        names = df.iloc[:, 0].astype(str).tolist()
    return names


def fetch_gbif_coords_for_species(species_list, limit=DEFAULT_GBIF_LIMIT, gbif_fields=None, dry_run=True):
    """
    Fetch occurrence coordinates from GBIF for each species in species_list.
    If pygbif not available or dry_run==True, this function will return an empty
    dict or synthetic coordinates (safe fallback). Set dry_run=False to attempt real calls.

    Returns dict: species -> list of (lon, lat) tuples
    """
    coords = {}
    if dry_run:
        # Do not call GBIF; return empty dict to indicate no data
        return coords

    if not PYG_BIF_AVAILABLE:
        raise RuntimeError("pygbif not available in this environment")

    for sp in species_list:
        try:
            res = occurrences.search(scientificName=sp, limit=limit)
            recs = res.get('results', [])
            sp_coords = []
            for r in recs:
                lon = r.get('decimalLongitude')
                lat = r.get('decimalLatitude')
                if lon is None or lat is None:
                    continue
                sp_coords.append((lon, lat))
            if sp_coords:
                coords[sp] = sp_coords
        except Exception as e:
            # skip species on error
            continue
    return coords


def coords_summary(coords_list):
    # coords_list: list of (lon, lat)
    if len(coords_list) == 0:
        return None
    arr = np.array(coords_list)
    lon = float(np.mean(arr[:, 0]))
    lat = float(np.mean(arr[:, 1]))
    return (lon, lat)


def build_climate_kernel_from_coords_map(species_list, species_to_coords_map):
    """
    species_list: list of species names (order to match existing herb kernel)
    species_to_coords_map: dict species -> list of (lon, lat)

    For each species, compute a representative coordinate (mean of occurrences) and
    build pairwise geo-distance and similarity kernel.
    If a species has no coords, assign NaN and later replace similarity with 0.
    """
    coords = []
    for sp in species_list:
        pts = species_to_coords_map.get(sp, [])
        summary = coords_summary(pts)
        if summary is None:
            coords.append((np.nan, np.nan))
        else:
            coords.append(summary)

    # For missing coords, replace them with column mean after computing dists among available
    valid_idx = [i for i, (lon, lat) in enumerate(coords) if not (np.isnan(lon) or np.isnan(lat))]
    if len(valid_idx) == 0:
        # no coordinates -> return identity kernel (no climate signal)
        n = len(coords)
        return np.eye(n)

    # Build distance matrix where missing rows/cols will be large distance
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[j]
            if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
                d = np.nan
            else:
                d = haversine(lon1, lat1, lon2, lat2)
            D[i, j] = d if d is not None else np.nan
            D[j, i] = D[i, j]

    # Replace NaN distances by a large value (2x median of observed distances)
    observed = D[~np.isnan(D)]
    if len(observed) == 0:
        sigma = 1.0
    else:
        median = np.median(observed)
        large = median * 2.0 if median > 0 else 1.0
        D = np.where(np.isnan(D), large, D)
    S = distance_to_similarity(D)
    return S


def ensure_ecoregion_mapping_csv(species_list, mapping_csv_path):
    """
    Ensure a scaffold CSV exists for RESOLVE/WWF ecoregion mapping.
    Columns: HerbId,scientific_name,biome,ecoregion
    If file does not exist, create it with empty biome/ecoregion for manual fill.
    """
    p = Path(mapping_csv_path)
    if p.exists():
        return str(p)
    rows = []
    for i, name in enumerate(species_list, start=1):
        rows.append({
            'HerbId': i,
            'scientific_name': name,
            'biome': '',
            'ecoregion': ''
        })
    df = pd.DataFrame(rows, columns=['HerbId', 'scientific_name', 'biome', 'ecoregion'])
    df.to_csv(p, index=False)
    return str(p)


def build_ecoregion_kernel_from_mapping(species_list, mapping_csv_path, strategy='ecoregion_then_biome'):
    """
    Build a similarity kernel from an herb->ecoregion mapping CSV.
    strategy controls how similarity is assigned:
      - 'ecoregion_then_biome': 1.0 if same ecoregion; 0.7 if same biome; 0.0 otherwise
      - 'biome_only': 1.0 if same biome else 0.0
    Missing entries result in 0 similarity off-diagonal. Diagonal set to 1.
    """
    p = Path(mapping_csv_path)
    if not p.exists():
        # no mapping -> identity kernel
        n = len(species_list)
        return np.eye(n)
    df = pd.read_csv(p)
    # Normalize names to join
    df['species'] = df['species'].astype(str)
    name_to_row = {row['species']: row for _, row in df.iterrows()}
    n = len(species_list)
    K = np.zeros((n, n), dtype=float)
    for i, a in enumerate(species_list):
        ra = name_to_row.get(str(a), None)
        for j, b in enumerate(species_list):
            if i == j:
                K[i, j] = 1.0
                continue
            rb = name_to_row.get(str(b), None)
            if ra is None or rb is None:
                K[i, j] = 0.0
                continue
            a_eco = str(ra.get('ecoregion', '')).strip()
            b_eco = str(rb.get('ecoregion', '')).strip()
            a_bio = str(ra.get('biome', '')).strip()
            b_bio = str(rb.get('biome', '')).strip()
            sim = 0.0
            if strategy == 'biome_only':
                sim = 1.0 if a_bio and a_bio == b_bio else 0.0
            else:
                if a_eco and a_eco == b_eco:
                    sim = 1.0
                elif a_bio and a_bio == b_bio:
                    sim = 0.7
                else:
                    sim = 0.0
            K[i, j] = sim
    return K


def load_kernel_matrix(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    M = np.loadtxt(p, delimiter=',')
    return M


def save_kernel_matrix(mat, path):
    np.savetxt(path, mat, fmt="%f", delimiter=',')


def fuse_kernels(original_kernel, climate_kernel, alpha=0.5):
    """
    Fuse kernels via weighted average: alpha * climate + (1-alpha) * original
    Both kernels should be same shape.
    """
    if original_kernel.shape != climate_kernel.shape:
        raise ValueError("Kernel shapes must match to fuse")
    fused = alpha * climate_kernel + (1.0 - alpha) * original_kernel
    # normalize to [0,1]
    minv = fused.min()
    maxv = fused.max()
    if maxv - minv > 0:
        fused = (fused - minv) / (maxv - minv)
    return fused


def build_and_predict_with_climate(alpha=0.5, use_gbif=False, gbif_limit=100, dry_run=True,
                                  herb_id_path=None, herb_kernel_path=None, disease_kernel_path=None,
                                  assoc_path=None, output_herb_kernel_path=None, predict_out_path=None,
                                  ecoregion_mapping_csv=None, ecoregion_weight=0.5):
    """
    High-level helper that:
    - loads herb id list and original herb kernel
    - (optionally) fetches GBIF coordinates for herbs (or uses provided coords map)
    - computes climate-derived similarity kernel
    - fuses it with original herb kernel using weight alpha
    - runs NSP projection with fused herb kernel and disease kernel and adjacency
    - saves fused kernel and prediction matrix

    dry_run=True prevents external GBIF calls even if use_gbif=True. Set dry_run=False to allow real calls (if pygbif installed).
    """
    # Paths
    herb_id_path = herb_id_path or (HERB_KERNEL_DIR / 'herb_id.csv')
    herb_kernel_path = herb_kernel_path or (HERB_KERNEL_DIR / 'herb_target.txt')
    disease_kernel_path = disease_kernel_path or (DISEASE_KERNEL_DIR / 'disease_target.txt')
    assoc_path = assoc_path or (ASSOC_DIR / 'disease_herb01.txt')
    output_herb_kernel_path = output_herb_kernel_path or (HERB_KERNEL_DIR / 'herb_kernel_with_climate.txt')
    predict_out_path = predict_out_path or (DATA_DIR / 'predict_with_climate.txt')

    # Load species names
    species_list = load_herb_ids(herb_id_path)

    # Load original herb kernel
    original_kernel = load_kernel_matrix(herb_kernel_path)

    # Load disease kernel and association matrix
    disease_kernel = load_kernel_matrix(disease_kernel_path)
    assoc = np.loadtxt(assoc_path, delimiter=',')

    # Fetch or assemble coordinates
    species_to_coords = {}
    if use_gbif:
        if dry_run:
            print("dry_run=True: skipping real GBIF calls. Provide species_to_coords to use real coords.")
        else:
            print("Fetching GBIF occurrences for species (this may take time) ...")
            species_to_coords = fetch_gbif_coords_for_species(species_list, limit=gbif_limit, dry_run=False)

    # Build climate kernel
    climate_kernel_geo = build_climate_kernel_from_coords_map(species_list, species_to_coords)

    # Optional: ecoregion mapping kernel
    if ecoregion_mapping_csv is None:
        # create scaffold for user to fill in
        ecoregion_mapping_csv = HERB_KERNEL_DIR / 'herb_ecoregion_mapping.csv'
        ensure_ecoregion_mapping_csv(species_list, ecoregion_mapping_csv)
    climate_kernel_eco = build_ecoregion_kernel_from_mapping(species_list, ecoregion_mapping_csv)

    # Combine climate kernels (if both available): Kc = ecoregion_weight * eco + (1-ecoregion_weight) * geo
    if climate_kernel_eco.shape == climate_kernel_geo.shape:
        climate_kernel = ecoregion_weight * climate_kernel_eco + (1.0 - ecoregion_weight) * climate_kernel_geo
    else:
        climate_kernel = climate_kernel_geo

    # Fuse
    fused = fuse_kernels(original_kernel, climate_kernel, alpha=alpha)

    # Save fused
    save_kernel_matrix(fused, output_herb_kernel_path)
    print(f"Saved fused herb kernel to {output_herb_kernel_path}")

    # Ensure adjacency orientation matches NSP expectation: (herbs x diseases)
    # In this repo, disease_herb01.txt is (diseases x herbs). Transpose to (herbs x diseases) if needed.
    A = assoc
    h, w = A.shape
    if h == disease_kernel.shape[0] and w == original_kernel.shape[0]:
        # A is (diseases x herbs) -> transpose
        A = A.T

    # Run projection using NSP (herb_similarity, disease_similarity, adjacency_matrix)
    nsp = NSP(fused, disease_kernel, A)
    prediction = nsp.network_NSP()
    save_kernel_matrix(prediction, predict_out_path)
    print(f"Saved prediction matrix to {predict_out_path}")

    return {
        'fused_kernel_path': str(output_herb_kernel_path),
        'prediction_path': str(predict_out_path),
        'fused_kernel': fused,
        'prediction': prediction
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HDAPM-NCP with climate-enhanced herb kernel')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for climate kernel fusion [0..1]')
    parser.add_argument('--use-gbif', action='store_true', help='fetch GBIF occurrences for herbs')
    parser.add_argument('--gbif-limit', type=int, default=75, help='records per species when using GBIF')
    parser.add_argument('--dry-run', action='store_true', help='do not perform network calls (even if --use-gbif)')
    parser.add_argument('--ecoregion-weight', type=float, default=0.5, help='weight between ecoregion and geo kernels')
    args = parser.parse_args()

    print("Running hdapm_climate.py with:")
    print(json.dumps({
        'alpha': args.alpha,
        'use_gbif': args.use_gbif,
        'gbif_limit': args.gbif_limit,
        'dry_run': args.dry_run,
        'ecoregion_weight': args.ecoregion_weight
    }, indent=2))

    results = build_and_predict_with_climate(
        alpha=args.alpha,
        use_gbif=args.use_gbif,
        gbif_limit=args.gbif_limit,
        dry_run=args.dry_run,
        ecoregion_weight=args.ecoregion_weight
    )
    print("Done. Paths:")
    print(json.dumps({k: v for k, v in results.items() if 'path' in k or isinstance(v, str)}, indent=2))
