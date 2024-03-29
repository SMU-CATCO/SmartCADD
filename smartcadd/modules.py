"""
Modules within SmartCADD define non-filtering operations such as data conversion, geometry optimization, explainable AI, and more. 
These modules work in conjunction with the filtering operations defined in the filters.py module and can be added to Pipelines defined in the pipeline.py module.
"""

from multiprocessing import Pool
from typing import List, Dict, Any
import os
from random import shuffle
from rdkit.Chem import AllChem, Mol, AddHs, MolToPDBFile

from .data import Compound


class Module:
    """
    Base Interface for modules
    """

    def __init__(
        self,
        module_config: Dict = None,
        output_dir: str = None,
        nprocesses: int = 1,
    ) -> None:
        self.module_config = module_config
        self.output_dir = output_dir
        self.nprocesses = nprocesses

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def __str__(self) -> str:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def run(self, batch: List[Compound], *args: Any, **kwargs: Any) -> Any:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def save(self, *args: Any, **kwargs: Any) -> None:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )


class DummyModule(Module):
    """
    Dummy Module for testing purposes
    """

    def __init__(
        self,
        module_config: Dict = None,
        output_dir: str = None,
        nprocesses: int = 1,
    ):
        super().__init__(module_config, output_dir, nprocesses)

    def run(self, batch: List[Compound], *args: Any, **kwargs: Any) -> Any:
        """
        Dummy Module that returns a shuffled batch
        """

        return shuffle(batch)


class SmileTo3D(Module):
    """
    Module for generating 3D coordinates for a batch of Compounds.

    Config:
        save (bool): save PDB files to output directory
        modify (bool): update Compound.mol with new 3D coordinates
    """

    def __init__(
        self,
        module_config: Dict = None,
        output_dir: str = None,
        nprocesses: int = 1,
    ):
        super().__init__(module_config, output_dir, nprocesses)

        if "save" in self.module_config:
            self.save = self.module_config["save"]
        else:
            self.save = False

        if "modify" in self.module_config:
            self.modify = self.module_config["modify"]
        else:
            self.modify = False

    def run(self, batch: List[Compound], *args: Any, **kwargs: Any) -> Any:
        """
        Generate 3D coordinates for a batch of Compounds

        Args:
            batch (List[Compound]): list of Compound objects
        """

        if self.output_dir is not None:
            os.makedirs(
                os.path.join(self.output_dir, "3D_coordinates"), exist_ok=True
            )
            save_dir = os.path.join(self.output_dir, "3D_coordinates")
        else:
            os.makedirs("3D_coordinates", exist_ok=True)
            save_dir = os.path.join(".", "3D_coordinates")

        def embed(compound):
            _mol = AddHs(Mol(compound.mol))
            AllChem.EmbedMolecule(_mol)
            if self.save:
                compound.pdb_path = os.path.join(
                    save_dir, f"{compound.id}.pdb"
                )
                MolToPDBFile(_mol, compound.pdb_path)

            if self.modify:
                compound.mol = _mol

        with Pool(self.nprocesses) as pool:
            pool.map(embed, batch)

        return batch


class XTBOptimization(Module):
    """
    Module for running XTB optimization on PDB files
    """

    def __init__(self, module_config: Dict = None):
        super().__init__(module_config)

    def run(self, batch: List[Compound], *args: Any, **kwargs: Any) -> Any:
        """
        Run XTB optimization on a batch of PDB files
        """

        return NotImplementedError(
            "This method should be implemented in the subclass."
        )
