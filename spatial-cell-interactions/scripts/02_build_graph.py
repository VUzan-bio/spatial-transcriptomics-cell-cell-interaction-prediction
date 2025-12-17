"""Construct spatial graph with distance encodings."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from spatial_interactions.graph.build_graph import build_spatial_graph, save_graph  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _coords_to_microns(
    adata: "sc.AnnData", coords: np.ndarray, spot_diameter_um: float = 55.0
) -> tuple[np.ndarray, bool]:
    """Convert coordinates to microns using scalefactors if available; fallback to pixels."""
    try:
        spatial_key = next(iter(adata.uns["spatial"].keys()))
    except Exception:
        logger.warning("No spatial scalefactors found; using pixel coordinates.")
        return coords, False
    scalefactors = adata.uns["spatial"][spatial_key].get("scalefactors", {})
    spot_diam_px = scalefactors.get("spot_diameter_fullres")
    if spot_diam_px is None or spot_diam_px == 0:
        logger.warning("spot_diameter_fullres missing; using pixel coordinates.")
        return coords, False
    um_per_px = spot_diameter_um / spot_diam_px
    logger.info(
        "Converting coordinates to microns with %.3f um/px (spot_diameter_fullres=%.3f px)",
        um_per_px,
        spot_diam_px,
    )
    return coords * um_per_px, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatial neighbor graph from AnnData.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Processed .h5ad file")
    parser.add_argument("--out_graph", type=Path, default=None, help="Output graph .pt path")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--graph_type", choices=["knn", "radius"], default="radius", help="Graph construction mode")
    parser.add_argument("--k", type=int, default=None, help="Number of neighbors for kNN graph")
    parser.add_argument(
        "--radius",
        type=str,
        default="auto",
        help="Radius for radius graph (units depend on distance_unit). Use 'auto' for heuristic.",
    )
    parser.add_argument(
        "--distance_unit",
        choices=["auto", "pixel", "um"],
        default="auto",
        help="Units for radius graph. 'auto' tries microns then falls back to pixels.",
    )
    parser.add_argument("--spot_diameter_um", type=float, default=55.0, help="Assumed spot diameter in microns for px->um conversion (if needed)")
    parser.add_argument("--rbf_dim", type=int, default=None, help="RBF embedding dimension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    k = args.k if args.k is not None else cfg["graph"]["k"]
    rbf_dim = args.rbf_dim if args.rbf_dim is not None else cfg["graph"]["rbf_dim"]

    logger.info("Reading AnnData from %s", args.h5ad)
    adata = sc.read_h5ad(args.h5ad)

    coords = np.array(adata.obsm["spatial"])
    conversion_used = False
    if args.graph_type == "radius":
        if args.distance_unit in {"auto", "um"}:
            coords_um, success = _coords_to_microns(
                adata, coords, spot_diameter_um=args.spot_diameter_um
            )
            if success:
                coords = coords_um
                conversion_used = True
                logger.info("Using microns for radius graph.")
            elif args.distance_unit == "um":
                logger.warning("Micron conversion unavailable; falling back to pixel coordinates.")
        # else keep pixel coords

    radius_val: Optional[float] = None
    if args.graph_type == "radius":
        if args.radius == "auto":
            nbrs = NearestNeighbors(n_neighbors=4).fit(coords)
            dists, _ = nbrs.kneighbors(coords)
            median_nn = float(np.median(dists[:, 1]))
            base = 1.5 * median_nn
            min_r = 0.9 * median_nn
            max_r = 3.0 * median_nn
            radius_val = float(np.clip(base, min_r, max_r))
            logger.info(
                "Auto radius (median NN=%.3f) -> %.3f (clamped to [%.3f, %.3f]) in %s",
                median_nn,
                radius_val,
                min_r,
                max_r,
                "microns" if conversion_used else "pixels",
            )
        else:
            radius_val = float(args.radius)

    artifacts = build_spatial_graph(
        adata,
        k=k,
        radius=radius_val if args.graph_type == "radius" else None,
        rbf_dim=rbf_dim,
        coords_override=coords,
    )

    if args.out_graph is None:
        sample = Path(args.h5ad).stem
        args.out_graph = REPO_ROOT / "data" / "processed" / f"{sample}_graph.pt"
    ensure_dir(args.out_graph.parent)
    save_graph(artifacts.data, args.out_graph)
    logger.info("Saved graph to %s", args.out_graph)


if __name__ == "__main__":
    main()
