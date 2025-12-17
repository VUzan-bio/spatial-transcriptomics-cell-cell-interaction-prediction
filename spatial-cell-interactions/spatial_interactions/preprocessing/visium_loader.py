"""Load Visium data using Scanpy."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import scanpy as sc
from pandas import DataFrame

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_tissue_positions(spatial_dir: Path) -> None:
    """
    Ensure tissue_positions_list.csv exists by converting from parquet or CSV if needed.
    """
    list_csv = spatial_dir / "tissue_positions_list.csv"
    fallback_csv = spatial_dir / "tissue_positions.csv"
    parquet = spatial_dir / "tissue_positions.parquet"

    if list_csv.exists():
        return

    if fallback_csv.exists():
        logger.info("Creating tissue_positions_list.csv from tissue_positions.csv")
        df = pd.read_csv(fallback_csv, header=None)
        df.to_csv(list_csv, index=False, header=False)
        return

    if parquet.exists():
        logger.info("Creating tissue_positions_list.csv from tissue_positions.parquet")
        df = pd.read_parquet(parquet)
        expected_cols = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        if not set(expected_cols).issubset(df.columns):
            raise ValueError(
                f"Parquet tissue positions missing required columns. Found {df.columns}, expected {expected_cols}"
            )
        df = df[expected_cols]
        df.to_csv(list_csv, index=False, header=False)
        return

    raise FileNotFoundError(
        f"Missing tissue positions file in {spatial_dir}. Provide tissue_positions_list.csv, tissue_positions.csv, or tissue_positions.parquet."
    )


def load_visium(
    visium_path: Path,
    count_file: str = "filtered_feature_bc_matrix.h5",
    source_image_path: Optional[Path] = None,
) -> "sc.AnnData":
    """
    Load Visium Space Ranger outputs with expression, spatial coords, and images.

    Parameters
    ----------
    visium_path:
        Path to Space Ranger outs/ directory containing filtered_feature_bc_matrix.h5 and spatial/.
    count_file:
        Count matrix file name.
    source_image_path:
        Optional override for tissue image directory if not at visium_path / "spatial".
    """
    visium_path = visium_path.resolve()
    if not visium_path.exists():
        raise FileNotFoundError(
            f"Visium path {visium_path} does not exist. Expected Space Ranger outs/ directory."
        )

    count_path = visium_path / count_file
    spatial_dir = source_image_path if source_image_path is not None else visium_path / "spatial"
    if not count_path.exists():
        raise FileNotFoundError(
            f"Count file {count_path} missing. Ensure Space Ranger outputs include {count_file}."
        )
    if not spatial_dir.exists():
        raise FileNotFoundError(
            f"Spatial directory {spatial_dir} missing. Expected under Space Ranger outs/."
        )

    _ensure_tissue_positions(spatial_dir)

    logger.info("Loading Visium data from %s", visium_path)
    adata = sc.read_visium(path=visium_path, count_file=count_file, load_images=True)
    if "spatial" not in adata.uns:
        raise ValueError("Scanpy did not populate adata.uns['spatial']; check input structure.")

    adata.var_names_make_unique()
    # Preserve raw counts if available
    adata.layers["counts"] = adata.X.copy()
    return adata
