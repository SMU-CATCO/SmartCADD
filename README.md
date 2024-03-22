# SmartCADD: An AI-Integrated Drug Designing Platform

## Current Version: **0.1**

### Version **0.01** Notes:

-   Initial version
-   Basic data structure is pandas dataframe

## Design

-   [Diagram](https://drive.google.com/file/d/1foG4DBd_m66nAzg3hPwy5EmI1jg1QlPk/view?usp=drive_link)
-   Pipeline:
    1. Dataset
    2. Model Filter
    3. ADMET Filter
    4. 2D Pharmacaphore Filter
    5. 3D Pharmacaphore Filter
    6. QM Filter
    7. Docking Filter

## Modules

-   _Model Filter_
    -   Attributes:
        -   pretrained model
        -   model configurations
        -   target class
        -   filter threshold
-   _ADMET Filter_
    -   ADMET filter types
    -   ADMET filter thresholds
    -   ADMET filter configurations
-   _2D Pharmacaphore Filter_
    -   target pharmacaphores
    -   pharmacaphore lists
-   _3D Pharmacaphore Filter_
    -   target pharmacaphores
    -   pharmacaphore lists
-   _QM Filter_
    -   QM functions
    -   QM target compound
-   _Docking Filter_
    -   target compound
    -   docking threshold
