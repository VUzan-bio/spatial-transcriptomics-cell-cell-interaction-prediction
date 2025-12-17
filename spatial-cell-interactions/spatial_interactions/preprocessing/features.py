"""Feature selection utilities."""

from __future__ import annotations

import scanpy as sc
import numpy as np

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def select_hvgs(
    adata: "sc.AnnData",
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    subset: bool = True,
) -> "sc.AnnData":
    """Compute and optionally subset to highly variable genes."""
    tried_flavors = []
    used_flavor = None
    for flv in [flavor, "cell_ranger", "pearson_residuals"]:
        if flv in tried_flavors:
            continue
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flv, subset=False)
            used_flavor = flv
            break
        except ImportError:
            tried_flavors.append(flv)
            logger.warning(
                "HVG selection with flavor '%s' requires scikit-misc. Trying next fallback.",
                flv,
            )
        except Exception as exc:
            tried_flavors.append(flv)
            logger.warning("HVG selection with flavor '%s' failed (%s). Trying next fallback.", flv, exc)

    if used_flavor is None:
        used_flavor = "variance"
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        var_arr = np.asarray(X.var(axis=0)).ravel()
        top_idx = np.argsort(var_arr)[::-1][: min(n_top_genes, var_arr.shape[0])]
        mask = [False] * var_arr.shape[0]
        for idx in top_idx:
            mask[idx] = True
        adata.var["highly_variable"] = mask

    hvgs = adata.var.index[adata.var["highly_variable"]]
    logger.info("Selected %d highly variable genes using %s", len(hvgs), used_flavor)
    if subset:
        adata = adata[:, hvgs].copy()
    return adata
