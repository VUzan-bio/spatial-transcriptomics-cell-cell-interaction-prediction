"""Detect and repair spatial coordinate alignment for Visium outs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scanpy as sc

# ensure package on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.preprocessing.visium_spatial_fix import (
    auto_fix_common_mismatches,
    detect_spatial_files,
    load_positions_table,
    validate_pixel_coords_against_image,
    write_tissue_positions_list,
)
from spatial_interactions.utils.io import ensure_dir
from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix spatial alignment for Visium outs.")
    p.add_argument("--outs_path", type=Path, required=True, help="Path to Space Ranger outs/ directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spatial_dir = args.outs_path / "spatial"
    if not spatial_dir.exists():
        raise FileNotFoundError(f"Missing spatial directory: {spatial_dir}")

    # Initial load
    adata = sc.read_visium(args.outs_path, load_images=True)
    lib = list(adata.uns["spatial"].keys())[0]
    images = adata.uns["spatial"][lib]["images"]
    scalefactors = adata.uns["spatial"][lib]["scalefactors"]
    img_key = "hires" if "hires" in images else "lowres"
    image = images[img_key]

    # Detect and load positions
    available = detect_spatial_files(spatial_dir)
    logger.info("Detected spatial files: %s", available)
    df = load_positions_table(spatial_dir)

    before = validate_pixel_coords_against_image(df, image)
    logger.info(
        "Before: x range [%.1f, %.1f], y range [%.1f, %.1f], image (w=%d, h=%d)",
        before.x_min,
        before.x_max,
        before.y_min,
        before.y_max,
        before.width,
        before.height,
    )

    fixed_df, fix_report = auto_fix_common_mismatches(df, image.shape[:2], scalefactors, target_img_key=img_key)
    after = validate_pixel_coords_against_image(fixed_df, image)

    out_path = write_tissue_positions_list(spatial_dir, fixed_df)
    logger.info("Wrote corrected tissue_positions_list.csv to %s", out_path)

    # Reload to confirm
    adata_fixed = sc.read_visium(args.outs_path, load_images=True)
    lib_fixed = list(adata_fixed.uns["spatial"].keys())[0]
    coords = adata_fixed.obsm["spatial"]

    report = {
        "outs_path": str(args.outs_path),
        "img_key": img_key,
        "detected_files": available,
        "fixes": fix_report,
        "before": before.__dict__,
        "after": after.__dict__,
        "reloaded_coords_min": coords.min(axis=0).tolist(),
        "reloaded_coords_max": coords.max(axis=0).tolist(),
    }

    # ensure JSON serializable
    def _clean(obj):
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, (int, float, str)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        return str(obj)

    report = _clean(report)

    report_path = Path("results") / "figures" / "alignment_report.json"
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Alignment report saved to %s", report_path)


if __name__ == "__main__":
    sys.exit(main())
