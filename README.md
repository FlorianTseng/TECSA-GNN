# TECSA-GNN 🔬

TECSA-GNN predicts electronic thermoelectric transport from crystal structures, composition descriptors, band gap, carrier type and concentration. This repository contains the shared 300 K models, prepared data and scripts accompanying [our paper](https://doi.org/10.1103/m8nb-bbp8) ([arXiv](https://arxiv.org/abs/2512.06697)).

## 🚀 Install

Create a Python environment (Python 3.10 or later), then install PyTorch and PyTorch Geometric following their platform-specific instructions. Install the remaining packages with:

```bash
pip install -r requirements.txt
```

Commands below are run from the repository root. The model automatically uses CUDA when available.

## 🔮 Predict

Run a small included example:

```bash
python -m tecsa_gnn.predict --properties examples/properties.csv \
  --cifs examples/cif_files --global-features examples/global-features.csv \
  --output predictions.csv
```

To run on the included dataset, first extract its structures and use `--limit` to select a manageable number of rows:

```bash
unzip -q data/cif_files.zip -d data
python -m tecsa_gnn.predict --limit 100 --output predictions.csv
```

The output gives `S` in µV/K and both log₁₀ and linear predictions of σ/τ and κₑ/τ. The checkpoints are for **300 K**; temperature is not an input. Supplying a measured or calculated band gap is preferable. The model does not predict the relaxation time τ.

## 🏋️ Train

```bash
python -m tecsa_gnn.train --target s --limit 4096 --output s-trained.pth
python -m tecsa_gnn.train --target sigma --limit 4096 --output sigma-trained.pth
python -m tecsa_gnn.train --target kappa --limit 4096 --output kappa-trained.pth
```

Extract the CIF archive first as above. `--limit 0` uses all rows; `--epochs`, `--batch-size`, `--lr`, and paths can be overridden with `--help`. Defaults use an example 90/10 **row-level** random split, not a composition- or structure-disjoint split. The script saves the lowest-validation-MSE checkpoint; it is a convenient training entry point, not the paper's full cross-validation workflow.

## 🧱 Data and new materials

`data/` includes the original share package's compressed property tables, normalized element/global features, unnormalized reference global features, and CIF archive. They contain 16,638 material IDs; the property tables have 166,380 S, 166,344 σ and 166,337 κₑ rows. Every ID in these tables has a CIF and global feature row. The source [Ricci dataset](https://doi.org/10.5061/dryad.gn001) and the [CraTENet preprocessing guide](https://github.com/lantunes/CraTENet#obtaining-the-training-data-the-dataset-preprocessing-pipeline) describe retrieval and trace extraction. The bundled tables are already in this implementation's format; CraTENet's pickle files are not directly interchangeable.

For a new material, provide:

1. A CIF named `<mpid>.cif` in a structure directory. IDs may be arbitrary unique strings matching the CSVs.
2. A properties CSV with columns `formula,mpid,300,type,dope,gap`. `type` is `n` or `p`; `dope` is carrier concentration divided by 10²⁰ cm⁻³ (e.g. `0.01` for 10¹⁸ cm⁻³); `gap` is in eV. For prediction `300` is a numeric placeholder, such as `0`. See `examples/properties.csv`.
3. One global feature row per ID. To generate it, make a CSV with `formula,mpid,gap` and run:

```bash
python -m tecsa_gnn.prepare_features --input materials.csv --output global-features.csv
python -m tecsa_gnn.predict --properties properties.csv --cifs path/to/cifs \
  --global-features global-features.csv --output predictions.csv
```

The feature generator uses Magpie, stoichiometry, valence-orbital, ion, band-center and elemental-fraction descriptors, orders columns using `data/global-feat.csv.gz`, and applies its min–max statistics. New values can fall outside [0, 1]. It requires compatible matminer descriptor names and a known, physically meaningful gap. The original element feature table is included. For training, the `300` column must contain the S target in µV/K or log₁₀(σ/τ), log₁₀(κₑ/τ), as appropriate. To derive new property tables from the raw database, extract the diagonal tensor values as described by CraTENet, take their mean, associate each selected structure with its band gap and carrier conditions, apply log₁₀ to positive conductivity targets, and write this CSV layout. The supplied tables preserve the original filtering and are the reference for reproduction; the full original filtering and κₑ extraction pipeline are not included.

## ⚠️ Reproducibility notes

The graph construction and line-graph batching follow the shared historical implementation for compatibility with the weights. Line-graph indices are constructed before bond deduplication and PyG's default batching offsets them as atom indices; consequently predictions can depend on batch size and neighboring samples. Use a fixed input order and batch size when comparing results. Changing the graph representation requires retraining and validation. The three weights are taken from the original public prediction folder; the κₑ weight is identical to the checkpoint in its `kappa-toy-train` folder. Their exact training splits and correspondence to every paper result have not been independently established. Feature generation for new materials and complete paper metrics should be independently validated before scientific use.

## 📖 Citation

```bibtex
@article{m8nb-bbp8,
  title = {Learning thermoelectric transport from crystal structures via multiscale graph neural network},
  author = {Zeng, Yuxuan and Cao, Wei and Zuo, Yijing and Lyu, Fang and Xie, Wenhao and Peng, Tan and Hou, Yue and Miao, Ling and Wang, Ziyu and Shi, Jing},
  journal = {Phys. Rev. Appl.},
  volume = {26},
  issue = {1},
  pages = {014019},
  numpages = {18},
  year = {2026},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/m8nb-bbp8},
  url = {https://link.aps.org/doi/10.1103/m8nb-bbp8}
}
```
