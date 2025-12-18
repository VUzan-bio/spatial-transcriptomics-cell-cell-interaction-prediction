# Spatial graph self-supervision and supervised interaction proxies (Visium/CytAssist)

This repo builds spatial graphs from 10x Space Ranger `outs/` (Visium/CytAssist), trains a self-supervised GNN (GATv2) on link prediction, and adds supervised proxy tasks for ligand–receptor (LR) and immune–epithelial interactions. It ships a full demo on the CytAssist FFPE Human Breast Cancer dataset and a preprint build.

## Quickstart (CPU)
> Dataset is large; download manually from 10x if links change.

1) Install deps:
```bash
pip install -r requirements.txt
```
2) Stage data:
```
data/external/breast_cytassist_ffpe/outs/
  filtered_feature_bc_matrix.h5        # downloaded
  spatial/                             # from spatial.tar.gz
```
3) Prepare h5ad:
```bash
python spatial-cell-interactions/scripts/01_prepare_data.py \
  --visium_path data/external/breast_cytassist_ffpe/outs \
  --count_file filtered_feature_bc_matrix.h5 \
  --out_h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --filter_in_tissue 1 --min_spots_frac 0.001 --n_hvg 2000
```
4) Build pixel-space radius graph:
```bash
python spatial-cell-interactions/scripts/02_build_graph.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --out_graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --graph_type radius --distance_unit pixel --radius auto --rbf_dim 16
```
5) Train SSL (link prediction):
```bash
python spatial-cell-interactions/scripts/03_train_ssl.py \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --out_dir results/run_breast_ssl \
  --config spatial-cell-interactions/configs/default.yaml \
  --device cpu
```
6) Supervised labels (LR + immune–epi):
```bash
python spatial-cell-interactions/scripts/08_build_interaction_labels.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt
```
7) Supervised training (SSL emb; PCA fallback):
```bash
python spatial-cell-interactions/scripts/09_train_supervised_edges.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --labels_dir data/processed/supervised_labels \
  --out_dir results/run_breast_supervised \
  --device cpu
```
8) Plots:
```bash
python spatial-cell-interactions/scripts/10_plot_supervised_interactions.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --preds_lr results/run_breast_supervised/preds_lr.parquet \
  --preds_immune_epi results/run_breast_supervised/preds_immune_epi.parquet \
  --preds_immune_epi_reg results/run_breast_supervised/preds_immune_epi_reg.parquet \
  --out_dir results/figures_supervised --spots_only 1
```
9) Preprint:
```bash
python reports/preprint_breast_ssl/build_preprint.py
```

## Demo facts (CytAssist breast)
- AnnData: 4,169 spots × 18,085 genes.
- Graph: radius ~462 px (median NN ~308 px); ~24,150 edges; RBF edge dim 16.
- SSL (edge reconstruction, CPU): early stop epoch 19; val AUROC 0.914, AP 0.890.

### Supervised (proxy) metrics
- LR (SSL features): test AUROC 0.977, AP 0.920.
- Immune–epithelial regression (SSL features): Spearman 0.893, top-k overlap 0.778; main supervised task.
- PCA baseline matches/exceeds SSL on these proxy labels; improving SSL utility is future work.
- Binary immune–epi is highly imbalanced (~95% positives) and secondary; type_pair exploratory.

## Repo layout
- `spatial-cell-interactions/` — code
  - `scripts/` — data prep, graph build, SSL train, supervised labels/train/plots, figures, end-to-end, alignment fix.
  - `spatial_interactions/` — preprocessing, graph, training, models, utils, visualization.
  - `configs/` — YAML configs.
- `resources/` — `ligand_receptor_pairs.csv`, `marker_sets.yaml`.
- `reports/preprint_breast_ssl/` — manuscript.tex/pdf/md + build script.
- `data/`, `results/` — gitignored; hold downloaded data, processed artifacts, runs, figures.

## Notes / limitations
- LR and immune–epi labels are expression/marker proxies on spot mixtures; not ground-truth biology.
- SSL currently does not outperform PCA on these proxies; future work to improve SSL utility.
- Some Visium HD “tiny” developer datasets are corner-cropped; use the CytAssist breast data for clean visuals.
