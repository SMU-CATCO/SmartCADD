"""
Modules within SmartCADD define non-filtering operations such as data conversion, geometry optimization, explainable AI, and more. 
These modules work in conjunction with the filtering operations defined in the filters.py module and can be added to Pipelines defined in the pipeline.py module.
"""

from multiprocessing import Pool
import subprocess
from typing import List, Dict, Any
import os
from shutil import copy
from random import shuffle
from rdkit.Chem import AllChem, Mol, AddHs, MolToPDBFile
from dgl.nn import SubgraphX
import networkx as nx

from .data import Compound
from .model_wrappers import ModelWrapper, AttentiveFP_DGLModelWrapper


class Module:
    """
    Base Interface for modules
    """

    def __init__(
        self,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ) -> None:
        self.module_config = module_config
        self.output_dir = output_dir
        self.n_processes = n_processes
        self.save_results = save_results

    def __call__(self) -> Any:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def __str__(self) -> str:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def run(self, batch: List[Compound]) -> Any:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def save(self, batch: List[Compound], output_file: str = None) -> None:
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
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(module_config, output_dir, n_processes)

    def run(self, batch: List[Compound]) -> Any:
        """
        Dummy Module that returns a shuffled batch
        """

        return shuffle(batch)

    def save(self, batch: List[Compound], output_file: str = None) -> None:
        """
        Dummy save method that does nothing
        """

        if output_file is None:
            output_file = "dummy_output.csv"

        with open(os.path.join(self.output_dir, output_file), "w") as f:
            f.write("SMILES,ID\n")
            for compound in batch:
                f.write(f"{compound.smiles},{compound.id}\n")


class SMILETo3D(Module):
    """
    Module for generating 3D coordinates for a batch of Compounds.

    Args:
        modify (bool): modify the Compound object with the 3D coordinates. Default is False
    """

    def __init__(
        self,
        modify: bool = False,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.modify = modify

    def run(self, batch: List[Compound]) -> Any:
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

        with Pool(self.n_processes) as pool:
            pool.map(embed, batch)

        return batch

    def save(self, batch: List[Compound], output_file: str = None) -> None:
        """
        Save results to csv with SMILES, ID, and PDB path

        Args:
            batch (List[Compound]): list of Compound objects
        """

        if output_file is None:
            output_file = "3D_coordinates.csv"

        with open(os.path.join(self.output_dir, output_file), "w") as f:
            f.write("SMILES,ID,PDB_PATH\n")
            for compound in batch:
                f.write(
                    f"{compound.smiles},{compound.id},{compound.pdb_path}\n"
                )


class XTBOptimization(Module):
    """
    Module for running XTB optimization on PDB files

    Args:
        from_file (bool): read PDB files from disk. If False, use Compound.mol
            if already converted to 3D using SMILETo3D. Default is False

        pdb_dir (str): directory containing PDB files. Default is None

    """

    def __init__(
        self,
        from_file: bool = False,
        pdb_dir: str = None,
        module_config: Dict = None,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.from_file = from_file
        self.pdb_dir = pdb_dir

        if self.from_file:
            assert self.pdb_dir is not None

    def run(self, batch: List[Compound]) -> Any:
        """
        Run XTB optimization on a batch of PDB files
        """

        if self.output_dir is not None:
            save_dir = os.path.join(self.output_dir, "XTB_optimized")
            os.makedirs(save_dir, exist_ok=True)

        else:
            save_dir = os.path.join(".", "XTB_optimized")
            os.makedirs(save_dir, exist_ok=True)

        workdir = os.getcwd()
        for compound in batch:
            if self.from_file:
                pdb_path = os.path.join(self.pdb_dir, f"{compound.id}.pdb")
            else:
                try:
                    # Convert generate pdb file
                    _mol = AddHs(Mol(compound.mol))
                    AllChem.EmbedMolecule(_mol)
                    pdb_path = os.path.join(save_dir, f".tmp.pdb")
                    MolToPDBFile(_mol, pdb_path)
                except Exception as e:
                    print(f"Error converting {compound.id} to PDB: {e}")
                    continue

            xtb_command = f"xtb {pdb_path} --opt"
            try:
                subprocess.run(xtb_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"xtb command failed for {compound.id}: {e.returncode}")
                continue

            try:
                os.rename(
                    os.path.join(workdir, "xtbopt.pdb"),
                    os.path.join(save_dir, f"{compound.id}_opt.pdb"),
                )
            except FileNotFoundError as e:
                print(f"Error renaming xtbopt.pdb for {compound.id}: {e}")

            compound.pdb_path = os.path.join(
                save_dir, f"{compound.id}_opt.pdb"
            )

        return batch

    def save(self, batch: List[Compound], output_file: str = None) -> None:
        """
        Save results to csv with SMILES, ID, and optimizedPDB path

        Args:
            batch (List[Compound]): list of Compound objects
        """

        if output_file is None:
            output_file = "XTB_optimized.csv"

        with open(os.path.join(self.output_dir, output_file), "w") as f:
            f.write("SMILES,ID,PDB_PATH\n")
            for compound in batch:
                f.write(
                    f"{compound.smiles},{compound.id},{compound.pdb_path}\n"
                )


class PDBToPDBQT(Module):
    """
    Module for converting PDB files to PDBQT files using OpenBabel
    """

    def __init__(
        self,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

    def run(self, batch: List[Compound]) -> Any:
        """
        Convert PDB files to PDBQT files
        """

        with Pool(self.n_processes) as pool:
            pool.map(self._process, batch)

        return batch

    def save(self, batch: List[Compound], output_file: str = None) -> None:
        """
        Save results to csv with SMILES, ID, and PDBQT path

        Args:
            batch (List[Compound]): list of Compound objects
        """

        if output_file is None:
            output_file = "PDBQT.csv"

        with open(os.path.join(self.output_dir, output_file), "w") as f:
            f.write("SMILES,ID,PDBQT_PATH\n")
            for compound in batch:
                f.write(
                    f"{compound.smiles},{compound.id},{compound.pdbqt_path}\n"
                )

    def _process(self, compound: Compound) -> None:
        """
        Convert PDB file to PDBQT file
        """

        try:
            pdb_path = compound.pdb_path
        except AttributeError:
            print(f"Compound {compound.id} does not have a pdb_path")
            return

        pdbqt_path = os.path.join(self.output_dir, f"{compound.id}.pdbqt")

        obabel_command = f"obabel -i pdb {pdb_path} -o pdbqt -O {pdbqt_path}"
        try:
            subprocess.run(obabel_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"obabel command failed for {compound.id}: {e.returncode}")
            return

        compound.pdbqt_path = pdbqt_path


class ExplainableAI(Module):
    """
    Module for generating explanations for a batch of Compounds

    Args:
        model_wrapper (ModelWrapper): ModelWrapper object
        target (int): target class to explain
        num_hops (int): number of n-hop neighborhoods to use for SubgraphX
        coef (float): hyperparameter for exploration-exploitation tradeoff. Higher = more exploration. Default = 20.0
        node_min (int): minimum number of nodes to include in explanation subgraph
        output_dir (str): output directory
        n_processes (int): number of processes to use
        save_results (bool): save results. Default is False.s

    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        target: int,
        num_hops: int = 1,
        coef: float = 20.0,
        node_min: int = 3,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.model_wrapper = model_wrapper
        self.target = target
        self.num_hops = num_hops
        self.coef = coef
        self.node_min = node_min

    def run(self, batch: List[Compound]) -> Any:
        """
        Generate explanations for a batch of Compounds
        """

        subgraph = SubgraphX(
            self.model_wrapper,
            num_hops=self.num_hops,
            coef=self.coef,
            node_min=self.node_min,
            log=False,
        )

        explained_batch = self._explain(subgraph, batch, target=self.target)

        if self.save:
            self.save(explained_batch)

        return explained_batch

    def save(self, batch: List[Compound], output_file: str = None) -> None:

        if output_file is None:
            output_file = "explanations.csv"

        with open(os.path.join(self.output_dir, output_file), "w") as f:
            f.write("ID,EXPLANATION\n")
            for compound in batch:
                if compound is None:
                    explanation = "NA"
                else:
                    explanation = compound.explanation
                f.write(f"{compound.id},{compound.explanation}\n")

    def _explain(
        self, explainer: SubgraphX, batch: List[Compound], target: int = 0
    ) -> Any:
        """
        Generate explanations for a batch of Compounds
        """

        featurized_batch = self.model_wrapper.featurize(batch)
        dgl_batch = [
            featurized_batch.X[i].to_dgl_graph()
            for i, _ in enumerate(featurized_batch)
        ]

        for i, g in enumerate(dgl_batch):

            try:
                batch[i].explanation = explainer.explain_graph(
                    g, g.ndata["x"], target_class=target
                )
            except Exception as e:
                print(
                    f"Failed to explain compound: {batch[i].id} with error {e}"
                )
                batch[i].explanation = None

        return batch
