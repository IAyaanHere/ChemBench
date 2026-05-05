# ChemBench Datasets

This document provides detailed information about all datasets included in ChemBench, including download links, statistics, and usage instructions.

## Dataset Summary Table

| # | Dataset Name | Samples | Features | Task Type | Application | License | Download Link |
|---|-------------|---------|----------|-----------|-------------|---------|--------------|
| 1 | Tennessee Eastman Process | 500,000+ | 52 | Classification | Fault Detection | Public Domain | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) |
| 2 | QM9 | 133,885 | 19 | Regression | Molecular Properties | CC0 | [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) |
| 3 | CheMixHub | 500,000+ | Varies | Regression | Mixture Properties | MIT | [GitHub](https://github.com/chemcognition-lab/chemixhub) |
| 4 | ESOL | 1,128 | 9 | Regression | Solubility Prediction | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 5 | FreeSolv | 642 | 9 | Regression | Solvation Free Energy | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 6 | Lipophilicity | 4,200 | 9 | Regression | Drug Properties | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 7 | HIV | 41,127 | 9 | Classification | Drug Discovery | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 8 | BACE | 1,513 | 9 | Classification | Drug Discovery | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 9 | Tox21 | 7,831 | 12 | Classification | Toxicity Prediction | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |
| 10 | BBBP | 2,039 | 9 | Classification | Blood-Brain Barrier | Public | [MoleculeNet](https://moleculenet.org/datasets-1) |

---

## Detailed Dataset Information

### 1. Tennessee Eastman Process (TEP)

**Category:** Fault Detection  
**Description:** Simulated chemical process data with 20 different fault types plus normal operation. Industry-standard benchmark for process monitoring and fault diagnosis.

**Statistics:**
- Training samples: 22 × 500 = 11,000 per fault
- Test samples: 22 × 960 = 21,120 per fault
- Total features: 52 (12 manipulated + 22 continuous + 19 composition variables)
- Fault types: 21 (including normal operation)
- Sampling interval: 3 minutes
- Training duration: 25 hours per simulation
- Test duration: 48 hours per simulation

**Task:** Multi-class classification (21 classes: normal + 20 faults)

**Download Instructions:**
```bash
# Method 1: Harvard Dataverse (Recommended)
wget https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/6C3JR1

# Method 2: Direct GitHub
git clone https://github.com/camaramm/tennessee-eastman-profBraatz
```

**Citation:**
```bibtex
@article{rieth2017,
  title={Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation},
  author={Rieth, Cory and Amsel, Ben and Tran, Randy and Cook, Maia},
  journal={Harvard Dataverse},
  year={2017}
}
```

**Usage in ChemBench:**
- Benchmark fault detection algorithms
- Evaluate anomaly detection methods
- Test process monitoring techniques
- Compare statistical vs ML approaches

---

### 2. QM9

**Category:** Molecular Property Prediction  
**Description:** Quantum chemistry dataset with DFT-calculated properties for 133,885 small organic molecules (up to 9 heavy atoms: C, O, N, F).

**Statistics:**
- Molecules: 133,885
- Properties per molecule: 19
- Heavy atoms: ≤9 (CONF)
- Average molecular weight: ~150 Da

**Properties:**
1. Dipole moment
2. Isotropic polarizability
3. HOMO energy
4. LUMO energy
5. HOMO-LUMO gap
6. Electronic spatial extent
7. Zero-point vibrational energy
8. Internal energy at 0K
9. Internal energy at 298K
10. Enthalpy at 298K
11. Free energy at 298K
12. Heat capacity at 298K
13-19. Various spectroscopic properties

**Task:** Multi-task regression (19 properties)

**Download Instructions:**
```python
# Using PyTorch Geometric (Easiest)
from torch_geometric.datasets import QM9
dataset = QM9(root='./datasets/qm9')

# Or manual download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
```

**Citation:**
```bibtex
@article{ramakrishnan2014,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and Von Lilienfeld, O Anatole},
  journal={Scientific data},
  volume={1},
  year={2014}
}
```

**Usage in ChemBench:**
- Benchmark molecular property prediction
- Test graph neural networks
- Evaluate transfer learning
- Compare with DFT calculations

---

### 3. CheMixHub

**Category:** Mixture Properties  
**Description:** Comprehensive benchmark for chemical mixture property prediction with 11 tasks from drug delivery to battery electrolytes.

**Statistics:**
- Total samples: ~500,000
- Tasks: 11
- Source datasets: 7
- Mixture types: Binary to multi-component
- Properties: Viscosity, density, ionic conductivity, etc.

**Included Tasks:**
1. Miscible Solvents - Density
2. Miscible Solvents - Enthalpy of Mixing
3. Miscible Solvents - Partial Molar Enthalpy
4. ILThermo - Ionic Conductivity (116,896 samples)
5. ILThermo - Viscosity (116,896 samples)
6. NIST Viscosity
7. Drug Solubility
8. Solid Polymer Electrolyte Conductivity
9-11. Additional mixture tasks

**Task:** Regression (various mixture properties)

**Download Instructions:**
```bash
# Clone repository
git clone https://github.com/chemcognition-lab/chemixhub.git
cd chemixhub

# Download processed data from Zenodo
# Instructions in README.md
```

**Citation:**
```bibtex
@article{rajaonson2025chemixhub,
  title={CheMixHub: Datasets and Benchmarks for Chemical Mixture Property Prediction},
  author={Rajaonson, Ella Miray and Kochi, Mahyar Rajabi and Mendoza, Luis Martin Mejia and Moosavi, Seyed Mohamad and Sanchez-Lengeling, Benjamin},
  journal={arXiv preprint arXiv:2506.12231},
  year={2025}
}
```

**Usage in ChemBench:**
- Benchmark mixture property prediction
- Test composition-aware models
- Evaluate few-shot learning
- Compare single vs multi-component predictions

---

### 4. ESOL (Water Solubility)

**Category:** Physical Chemistry  
**Description:** Water solubility data (log solubility in mol/L) for common organic small molecules.

**Statistics:**
- Molecules: 1,128
- Features: 9 (molecular descriptors)
- Target: log solubility in mol/L
- Average molecular weight: ~200 Da

**Task:** Regression (solubility prediction)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
```

**Citation:**
```bibtex
@article{delaney2004,
  title={ESOL: estimating aqueous solubility directly from molecular structure},
  author={Delaney, John S},
  journal={Journal of chemical information and computer sciences},
  volume={44},
  number={3},
  pages={1000--1005},
  year={2004}
}
```

**Usage in ChemBench:**
- Benchmark solubility prediction
- Test QSAR models
- Evaluate molecular featurization
- Compare descriptors vs graph methods

---

### 5. FreeSolv

**Category:** Physical Chemistry  
**Description:** Experimental and calculated hydration free energy of small molecules in water.

**Statistics:**
- Molecules: 642
- Features: 9 (molecular descriptors)
- Target: Hydration free energy (kcal/mol)
- Includes both experimental and calculated values

**Task:** Regression (solvation free energy)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_freesolv(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv
```

**Citation:**
```bibtex
@article{mobley2014,
  title={FreeSolv: a database of experimental and calculated hydration free energy of small molecules in water},
  author={Mobley, David L and Guthrie, J Peter},
  journal={Journal of computer-aided molecular design},
  volume={28},
  number={7},
  pages={711--720},
  year={2014}
}
```

**Usage in ChemBench:**
- Benchmark solvation prediction
- Test physics-informed models
- Evaluate uncertainty quantification
- Compare ML vs MD simulations

---

### 6. Lipophilicity

**Category:** Drug Properties  
**Description:** Experimental octanol/water distribution coefficient (logD at pH 7.4) - key for drug membrane permeability.

**Statistics:**
- Molecules: 4,200
- Features: 9 (molecular descriptors)
- Target: logD at pH 7.4
- Range: -1.5 to 4.5

**Task:** Regression (lipophilicity prediction)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
```

**Citation:**
```bibtex
@article{wu2018,
  title={MoleculeNet: a benchmark for molecular machine learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and others},
  journal={Chemical science},
  volume={9},
  number={2},
  pages={513--530},
  year={2018}
}
```

**Usage in ChemBench:**
- Benchmark drug property prediction
- Test ADME models
- Evaluate descriptor importance
- Compare regression methods

---

### 7. HIV

**Category:** Drug Discovery  
**Description:** Ability of compounds to inhibit HIV replication (from NCI-60 screening).

**Statistics:**
- Molecules: 41,127
- Features: 9 (molecular descriptors)
- Target: Binary (active/inactive)
- Class imbalance: ~1.5% active

**Task:** Binary classification (HIV inhibition)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv
```

**Citation:**
```bibtex
@article{hiv_dataset,
  title={MoleculeNet: a benchmark for molecular machine learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and others},
  journal={Chemical science},
  year={2018}
}
```

**Usage in ChemBench:**
- Benchmark drug screening
- Test imbalanced classification
- Evaluate scaffold splitting
- Compare active learning methods

---

### 8. BACE

**Category:** Drug Discovery  
**Description:** Binding results for set of inhibitors of human β-secretase 1 (BACE-1).

**Statistics:**
- Molecules: 1,513
- Features: 9 (molecular descriptors)
- Target: Binary (active/inactive)
- IC50 threshold: 50 nM

**Task:** Binary classification (enzyme inhibition)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
```

**Citation:**
```bibtex
@article{subramanian2016,
  title={Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches},
  author={Subramanian, Govindan and Ramsundar, Bharath and others},
  journal={Journal of chemical information and modeling},
  year={2016}
}
```

**Usage in ChemBench:**
- Benchmark enzyme inhibition prediction
- Test scaffold-based splits
- Evaluate model generalization
- Compare classification metrics

---

### 9. Tox21

**Category:** Toxicity Prediction  
**Description:** Qualitative toxicity measurements on 12 biological targets including nuclear receptors and stress response pathways.

**Statistics:**
- Molecules: 7,831
- Targets: 12
- Features: 9 (molecular descriptors)
- Task type: Multi-task classification
- Missing values: ~30%

**Targets:**
1. Nuclear receptor signaling
2. Stress response pathways
3. Aromatase inhibition
4. Androgen receptor
5. Estrogen receptor
6-12. Other toxicity endpoints

**Task:** Multi-task binary classification (12 tasks)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
```

**Citation:**
```bibtex
@article{tox21,
  title={Toxicology in the 21st century (Tox21)},
  author={Huang, Ruili and others},
  journal={Chemical research in toxicology},
  year={2016}
}
```

**Usage in ChemBench:**
- Benchmark toxicity prediction
- Test multi-task learning
- Evaluate missing data handling
- Compare safety prediction methods

---

### 10. BBBP (Blood-Brain Barrier Penetration)

**Category:** Drug ADME Properties  
**Description:** Binary labels of blood-brain barrier penetration (permeability) for 2,039 compounds.

**Statistics:**
- Molecules: 2,039
- Features: 9 (molecular descriptors)
- Target: Binary (penetrates/doesn't penetrate)
- Class balance: ~50-50

**Task:** Binary classification (BBB penetration)

**Download Instructions:**
```python
# Using DeepChem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='ECFP')

# Or direct download
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
```

**Citation:**
```bibtex
@article{martins2012,
  title={A Bayesian approach to in silico blood-brain barrier penetration modeling},
  author={Martins, Ines Filipa and others},
  journal={Journal of chemical information and modeling},
  year={2012}
}
```

**Usage in ChemBench:**
- Benchmark ADME prediction
- Test CNS drug design models
- Evaluate molecular descriptors
- Compare brain penetration methods

---

## Quick Start Guide

### 1. Download All Datasets

```bash
cd ChemBench
python scripts/download_datasets.py --all
```

### 2. Download Specific Dataset

```bash
python scripts/download_datasets.py --dataset tennessee_eastman
python scripts/download_datasets.py --dataset qm9
python scripts/download_datasets.py --dataset chemixhub
```

### 3. Verify Downloads

```bash
python scripts/verify_datasets.py
```

### 4. Generate Dataset Statistics

```bash
python scripts/dataset_stats.py --output datasets/statistics.md
```

---

## Dataset Organization

```
datasets/
├── tennessee_eastman/
│   ├── raw/
│   │   ├── d00.dat
│   │   ├── d01.dat
│   │   └── ...
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── README.md
│   └── metadata.json
│
├── qm9/
│   ├── raw/
│   │   └── qm9.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── README.md
│   └── metadata.json
│
└── ... (similar structure for each dataset)
```

---

## Data Processing Pipeline

1. **Download**: Fetch raw data from source
2. **Clean**: Handle missing values, outliers
3. **Split**: Create train/val/test splits
4. **Normalize**: Standardize features
5. **Document**: Generate metadata and statistics

---

## Contributing New Datasets

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on adding new datasets to ChemBench.

Requirements for new datasets:
- Publicly available
- Relevant to chemical engineering
- Proper license documentation
- Minimum 500 samples
- Clear task definition
- Benchmark baselines

---

## License

Each dataset has its own license. Please refer to individual dataset documentation for licensing information.

---

## Acknowledgments

We thank all original dataset creators and maintainers for making their data publicly available.
