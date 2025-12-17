"""Utilities to detect, repair, and validate Visium spatial metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


def detect_spatial_files(spatial_dir: Path) -> Dict[str, bool]:
    """Detect available spatial files."""
    return {
        "parquet": (spatial_dir / "tissue_positions.parquet").exists(),
        "csv": (spatial_dir / "tissue_positions.csv").exists(),
        "listcsv": (spatial_dir / "tissue_positions_list.csv").exists(),
    }


def _read_positions(spatial_dir: Path) -> pd.DataFrame:
    """Read positions from best available source."""
    parq = spatial_dir / "tissue_positions.parquet"
    csv = spatial_dir / "tissue_positions.csv"
    listcsv = spatial_dir / "tissue_positions_list.csv"

    if parq.exists():
        df = pd.read_parquet(parq)
    elif csv.exists():
        df = pd.read_csv(csv)
    elif listcsv.exists():
        df = pd.read_csv(listcsv, header=None)
    else:
        raise FileNotFoundError("No tissue_positions.{parquet,csv,_list.csv} found.")
    return df


def load_positions_table(spatial_dir: Path) -> pd.DataFrame:
    """Load positions and normalize column order/dtypes."""
    df = _read_positions(spatial_dir)
    if df.shape[1] == 6:
        df.columns = REQUIRED_COLS
    elif set(REQUIRED_COLS).issubset(df.columns):
        df = df[REQUIRED_COLS]
    else:
        raise ValueError(f"Unexpected columns in tissue positions: {df.columns}")

    df = df.copy()
    df["in_tissue"] = df["in_tissue"].astype(int)
    for col in ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        df[col] = df[col].astype(int)
    return df


def write_tissue_positions_list(spatial_dir: Path, df: pd.DataFrame) -> Path:
    """Write tissue_positions_list.csv without header and keep backup if overwriting."""
    out_path = spatial_dir / "tissue_positions_list.csv"
    if out_path.exists():
        backup = spatial_dir / "tissue_positions_list.backup.csv"
        if backup.exists():
            backup.unlink()
        out_path.rename(backup)
    df.to_csv(out_path, index=False, header=False)
    return out_path


@dataclass
class ValidationReport:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int
    swapped_likely: bool
    out_of_bounds: bool
    notes: str


def validate_pixel_coords_against_image(df: pd.DataFrame, image: np.ndarray) -> ValidationReport:
    """Check coordinate bounds relative to an image."""
    x = df["pxl_col_in_fullres"].to_numpy()
    y = df["pxl_row_in_fullres"].to_numpy()
    h, w = image.shape[:2]
    swapped = (x.min() >= h or x.max() <= h) and (y.min() >= w or y.max() <= w)
    oob = (x.min() < 0) or (y.min() < 0) or (x.max() > w * 5) or (y.max() > h * 5)
    return ValidationReport(
        x_min=float(x.min()),
        x_max=float(x.max()),
        y_min=float(y.min()),
        y_max=float(y.max()),
        width=w,
        height=h,
        swapped_likely=swapped,
        out_of_bounds=oob,
        notes="",
    )


def auto_fix_common_mismatches(
    df: pd.DataFrame,
    image_shape: Tuple[int, int],
    scalefactors: Dict[str, float],
    target_img_key: str = "hires",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Attempt to fix swapped or mis-scaled coordinates."""
    report = {}
    fixed = df.copy()
    h, w = image_shape
    x = fixed["pxl_col_in_fullres"].to_numpy(dtype=float)
    y = fixed["pxl_row_in_fullres"].to_numpy(dtype=float)

    # Swap if clearly swapped
    if x.min() >= h and y.min() >= w:
        fixed["pxl_col_in_fullres"], fixed["pxl_row_in_fullres"] = y, x
        report["swap_xy"] = "applied"
        x, y = y, x

    # Scaling heuristics
    hires_scale = scalefactors.get("tissue_hires_scalef")
    lowres_scale = scalefactors.get("tissue_lowres_scalef")
    if target_img_key == "hires" and hires_scale is not None:
        if x.max() < w * hires_scale * 0.7 and y.max() < h * hires_scale * 0.7:
            scale = 1.0 / hires_scale
            fixed["pxl_col_in_fullres"] = (x * scale).astype(int)
            fixed["pxl_row_in_fullres"] = (y * scale).astype(int)
            report["scaled"] = f"scaled up by {scale:.3f} to match hires"
    if target_img_key == "lowres" and lowres_scale is not None:
        if x.max() > w * 1.5 or y.max() > h * 1.5:
            scale = lowres_scale
            fixed["pxl_col_in_fullres"] = (x * scale).astype(int)
            fixed["pxl_row_in_fullres"] = (y * scale).astype(int)
            report["scaled"] = f"scaled down by {scale:.3f} to match lowres"

    return fixed, report
