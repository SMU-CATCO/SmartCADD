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

pip install -e
```

## Dependencies

-   PyMol: `conda install -c conda-forge -c schrodinger pymol-bundle`

    -   If MaxOS: `pip install PyQt5`
    -   [Other options]('https://pymol.org/support.html?#installation')

-   PDBFixer: `conda install conda-forge::pdbfixer`

-   Openmm: `pip install openmm`
-   MDAnalysis: `pip install MDAnalysis`
-   pybel: `pip install pybel`
