<p align="center"><img src="./logo.jpeg" width="200" height="200"></p>

# SmartCADD: An AI-Integrated Drug Designing Platform

## Project Overview
SmartCADD is an open-source platform designed to innovate in the field of drug discovery by integrating deep learning, computer-aided drug design (CADD), and quantum mechanics methodologies. Developed in a user-friendly Python framework, SmartCADD aims to provide researchers and developers with powerful tools for virtual screening and drug design.

## Current Version: **0.1**

### Version **0.01** Notes:

-   Initial version
-   Basic data structure is pandas dataframe

## Installation

```bash
git clone git@github.com:SMU-CATCO/SmartCADD.git
cd SmartCADD

conda install -n base conda-forge::mamba
mamba env create -f conda-environment.yml 
mamba activate smartcadd
```


```markdown
# SmartCADD



## Installation Instructions
```bash
# Clone the repository
git clone https://github.com/SMU-CATCO/SmartCADD.git

# Navigate to the SmartCADD folder
cd SmartCADD

# Install using pip
pip install -r requirements.txt
```

## Usage Examples
To get started with SmartCADD, you can run a simple virtual screening process:

```python
from smartcadd.pipeline import BasicCompoundPipeline
from smartcadd.filters import ADMETFilter
from smartcadd.dataset import IterableDataset

# Create a dataset iterator
dataset = IterableDataset(
    root_dir="./data",
    batch_size=10,
)

# Create a pipeline with an ADMET filter
admet_pipeline = BasicCompoundPipeline(
    data_loader=dataset,
    filters=[
        ADMETFilter(
            alert_collection_path="alert_collection.csv",
            output_dir="./results",
            save_results=True,
        ),
    ],
)

# Run ADMET filtering
filtered_results = admet_pipeline.run_filters()
```

## Contribution Guidelines
We welcome contributions to the SmartCADD project! If you're looking to contribute, please start by reading our contribution guidelines in the `CONTRIBUTING.md` file. For any contributions, ensure you follow our code standards and submit a pull request for review.

## License Information
SmartCADD is released under the MIT License. For more details, see the `LICENSE` file in our repository.

## Support and Community
Join our community to get support, discuss new features, and more! Check out the `SUPPORT.md` file for ways to get in touch, or join our discussions on the GitHub repository.
