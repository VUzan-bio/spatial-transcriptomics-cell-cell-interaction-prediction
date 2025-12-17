"""Generate quick visualizations: tissue spots and radius-graph overlay."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

# ensure local package available when script run directly
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from spatial_interactions.utils.io import ensure_dir


def main() -> None:
    # Ensure local package import works when executed directly
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from torch.serialization import add_safe_globals

    add_safe_globals([torch.Tensor])

    adata = sc.read_h5ad("data/processed/visium_hd_tiny.h5ad")
    fig_dir = Path("results") / "figures"
    ensure_dir(fig_dir)

    # Visium spots on tissue
    lib = list(adata.uns["spatial"].keys())[0]
    xy = adata.obsm["spatial"]
    pad = 200.0
    left = float(np.min(xy[:, 0]) - pad)
    right = float(np.max(xy[:, 0]) + pad)
    top = float(np.min(xy[:, 1]) - pad)
    bottom = float(np.max(xy[:, 1]) + pad)

    plt.figure(figsize=(6, 6))
    sc.pl.spatial(
        adata,
        library_id=lib,
        img_key="hires",
        color="in_tissue",
        size=1.2,
        alpha_img=1.0,
        crop_coord=(left, right, top, bottom),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "visium_spots_on_tissue.png", dpi=200)
    plt.close()

    # Radius graph overlay
    graph = torch.load(
        "data/processed/visium_hd_tiny_radius_graph.pt",
        map_location="cpu",
        weights_only=False,
    )
    coords = adata.obsm["spatial"].astype(float)
    edge_index = graph.edge_index.numpy()
    rng = np.random.default_rng(0)
    keep = rng.choice(edge_index.shape[1], size=min(4000, edge_index.shape[1]), replace=False)
    ei = edge_index[:, keep]

    x, y = coords[:, 0], coords[:, 1]
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=1, c="black", alpha=0.6)
    for a, b in ei.T:
        plt.plot([x[a], x[b]], [y[a], y[b]], color="tab:blue", alpha=0.03, linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "radius_graph_overlay.png", dpi=200)
    plt.close()

    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
