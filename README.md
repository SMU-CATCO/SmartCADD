<p align="center"><img src="./logo.jpeg" width="200" height="200"></p>

# SmartCADD: An AI-Integrated Drug Designing Platform

## Current Version: **0.1**

### Version **0.01** Notes:

-   Initial version
-   Basic data structure is pandas dataframe

## Installation

```bash
git clone git@github.com:SMU-CATCO/SmartCADD.git
cd SmartCADD

conda create -n smartcadd python=3.10
conda activate smartcadd

pip install poetry
poetry install

# install conda-specific dependencies
poetry run conda install h5py conda-forge::pymol-open-source conda-forge::openbabel conda-forge::pdbfixer
```

## Dependencies

-   PyMol: `conda install -c conda-forge pymol-open-source`

    -   Dependencies: `conda install -y h5py`
    -   [Other options]('https://pymol.org/support.html?#installation')

-   OpenBabel: `conda install -c conda-forge openbabel`
-   PDBFixer: `conda install -c conda-forge pdbfixer`

-   Openmm: `pip install openmm`
-   MDAnalysis: `pip install MDAnalysis`
-   pybel: `pip install pybel`
-   PyTorch: `pip install torch`
