"""Construct spatial graph with distance encodings."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.graph.build_graph import build_spatial_graph, save_graph  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _coords_to_microns(adata: "sc.AnnData", coords: np.ndarray, spot_diameter_um: float = 55.0) -> np.ndarray:
    """Convert coordinates to microns using scalefactors if available; fallback to pixels."""
    try:
        spatial_key = next(iter(adata.uns["spatial"].keys()))
    except Exception:
        logger.warning("No spatial scalefactors found; using pixel coordinates.")
        return coords
    scalefactors = adata.uns["spatial"][spatial_key].get("scalefactors", {})
    spot_diam_px = scalefactors.get("spot_diameter_fullres")
    if spot_diam_px is None or spot_diam_px == 0:
        logger.warning("spot_diameter_fullres missing; using pixel coordinates.")
        return coords
    um_per_px = spot_diameter_um / spot_diam_px
    logger.info("Converting coordinates to microns with %.3f um/px (spot_diameter_fullres=%.3f px)", um_per_px, spot_diam_px)
    return coords * um_per_px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatial neighbor graph from AnnData.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Processed .h5ad file")
    parser.add_argument("--out_graph", type=Path, default=None, help="Output graph .pt path")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--graph_type", choices=["knn", "radius"], default="knn", help="Graph construction mode")
    parser.add_argument("--k", type=int, default=None, help="Number of neighbors for kNN graph")
    parser.add_argument("--radius", type=float, default=None, help="Radius for radius graph (units depend on distance_unit)")
    parser.add_argument("--distance_unit", choices=["pixel", "um"], default="pixel", help="Units for radius graph")
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
    if args.graph_type == "radius" and args.distance_unit == "um":
        coords_um = _coords_to_microns(adata, coords, spot_diameter_um=args.spot_diameter_um)
        coords = coords_um

    radius = args.radius
    if args.graph_type == "radius" and radius is None:
        # heuristic: median NN distance * 1.5
        nbrs = NearestNeighbors(n_neighbors=4).fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        median_nn = np.median(dists[:, 1])
        radius = float(median_nn * 1.5)
        logger.info("Auto radius set to %.3f based on median NN distance", radius)

    artifacts = build_spatial_graph(
        adata,
        k=k,
        radius=radius if args.graph_type == "radius" else None,
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
