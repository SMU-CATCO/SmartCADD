from collections import defaultdict
import os
import subprocess
from typing import List, Dict, Tuple
import pandas as pd
from multiprocessing import Pool
from pymol import cmd
from openbabel import pybel
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
from rdkit import Chem
from rdkit.Geometry import Point3D

from .model_wrappers import ModelWrapper
from .data import Compound, SMARTS_Query
from . import utils


class Filter:
    def __init__(
        self,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        self.output_dir = output_dir
        self.n_processes = n_processes
        self.save_results = save_results

        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            if not os.path.exists(output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, batch: List[Compound]) -> List[Compound]:
        return self.run(batch)

    def __str__(self) -> str:
        return self.__class__.__name__

    def run(self, batch: List[Compound]) -> List[Compound]:
        raise NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def save(self, batch: List[Compound], output_file: str = None) -> None:
        raise NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def _filter(self, compound: Compound) -> Compound:
        raise NotImplementedError(
            "This method should be implemented in the subclass."
        )


class DummyFilter(Filter):
    """
    Dummy filter for testing purposes

    Args:
        filter_config (dict): configuration for filter
    """

    def __init__(
        self,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Dummy filter that returns the input batch

        Args:
            batch: list of Compound objects

        Returns:
            batch: list of Compound objects
        """

        return batch

    def save(
        self, batch: List[Compound], output_file: str = "dummy_filtered.csv"
    ):
        """
        Save results to csv with SMILES, ID, **descriptors

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "dummy_filtered.csv"
        """

        df = pd.DataFrame.from_records(
            [compound.to_dict() for compound in batch]
        )

        df.to_csv(os.path.join(self.output_dir, output_file), index=False)


class ADMETFilter(Filter):
    """
    Filter compounds based on ADMET PAINS patterns

    Args:
        alert_collection_path (str): path to alert collection csv file
        output_dir (str): output directory for saving results
        n_processes (int): number of processes to use for filtering
        save_results (bool): save filtered compounds to csv. Default is False

    Returns:
        filtered_batch: list of filtered Compound objects based on ADMET PAINS patterns

    """

    def __init__(
        self,
        alert_collection_path: str,
        output_dir: int = ".",
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.alert_collection_path = alert_collection_path
        self._pains_patterns = self._init(
            alert_collection_path=self.alert_collection_path
        )

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Filters compounds based on ADMET PAINS patterns

        Args:
            batch: list of Compound objects

        Returns:
            filtered_batch: list of filtered Compound objects
        """

        with Pool(self.n_processes) as pool:
            mask = pool.map(self._filter, batch)

        # remove None values
        filtered_batch = [
            compound for compound, keep in zip(batch, mask) if keep
        ]

        if self.save_results:
            self.save(zip(batch, mask))

        return filtered_batch

    def save(
        self, batch: List[Compound], output_file: str = "ADMET_filtered.csv"
    ) -> None:
        """
        Save results to csv with SMILES, ID, **descriptors

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "ADMET_filtered.csv"
        """

        df = pd.DataFrame.from_records(
            [compound.to_dict() | {"keep": keep} for compound, keep in batch]
        )

        # save to csv
        save_path = os.path.join(self.output_dir, output_file)
        if not os.path.isfile(save_path):
            df.to_csv(save_path, index=False, mode="a")
        else:
            df.to_csv(save_path, index=False, mode="a", header=False)

    def _init(self, alert_collection_path: str) -> List[SMARTS_Query]:
        """
        Initialize ADMET PAINS patterns for filtering
        """

        try:
            rule_df = pd.read_csv(alert_collection_path)
        except Exception as e:
            print(f"Error reading alert collection file: {e}")
            raise e

        PAINS_df = rule_df[rule_df["rule_set_name"] == "PAINS"]

        temp_list = [
            SMARTS_Query(smarts, max_val, desc)
            for smarts, max_val, desc in PAINS_df[
                ["smarts", "max", "description"]
            ].values
        ]

        return [query for query in temp_list if query.pattern]

    def _filter(self, compound: Compound) -> Compound:
        """
        Filter a single compound based on ADMET PAINS patterns
        """

        if compound.descriptors["status"] == "INVALID":
            return False

        for smarts_query in self._pains_patterns:
            if (
                len(compound.mol.GetSubstructMatches(smarts_query.pattern))
                > smarts_query.max_val
            ):
                compound.descriptors["status"] = (
                    f"{smarts_query.desc} > {smarts_query.max_val}"
                )
                return False

        return True


class ModelFilter(Filter):
    """
    Filter compounds based on model prediction probabilities

    Args:
        model_wrapper (ModelWrapper) wrapper for the model
        target (int): target class index to filter with
        threshold (float): threshold for filtering

    Config:
        save_results (bool): save filtered compounds to csv. Default is False

    Returns:
        filtered_batch: list of filtered Compound objects based on model
            prediction probabilities > threshold
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        target: int = 0,
        threshold: float = 0.5,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.model_wrapper = model_wrapper
        self.target = target
        self.threshold = threshold

        # load model weights
        try:
            self.model_wrapper.load()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Filter compounds based on model prediction probabilities

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of filtered Compound objects

        """

        predictions = self._predict(batch, self.target)

        # filter based on threshold
        filtered_batch = [
            compound
            for idx, compound in enumerate(batch)
            if predictions[idx] > self.threshold
        ]

        if self.save_results:
            self.save(zip(batch, predictions))

        return filtered_batch

    def save(
        self, batch: List[Compound], output_file: str = "model_filtered.csv"
    ) -> None:
        """
        Save results to csv with SMILES, ID, prediction

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "model_filtered.csv"
        """

        # make df
        df = pd.DataFrame.from_records(
            [
                {
                    "SMILES": compound.smiles,
                    "ID": compound.id,
                    "Prediction": round(float(prediction), 3),
                }
                for compound, prediction in batch
            ]
        )

        # save to csv
        save_path = os.path.join(self.output_dir, output_file)
        if not os.path.isfile(save_path):
            df.to_csv(save_path, index=False, mode="a")
        else:
            df.to_csv(save_path, index=False, mode="a", header=False)

    def _predict(self, batch: List[Compound], target: int = 0) -> List[float]:
        """
        Predict on input batch using the model.

        Args:
            batch (list): list of Compound objects

        Config:
            save_results (bool): save filtered compounds to csv. Default is False

        Returns:
            list: list of prediction probabilities

        """

        featurized_batch = self.model_wrapper.featurize(batch)

        return self.model_wrapper.predict(featurized_batch, self.target)


class PharmacophoreFilter2D(Filter):
    """
    Filter compounds based on 2D pharmacophore features of a set of template compounds

    Args:
        template_compounds (List[Compound]): list of template Compound objects to use for filtering

    Config:
        save_results (bool): save filtered compounds to csv. Default is False

    Returns:
        filtered_batch: list of filtered Compound objects based on 2D pharmacophore features
    """

    def __init__(
        self,
        template_compounds: List[Compound],
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.template_dict = self._preprocess_templates(template_compounds)

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Filter compounds based on 2D pharmacophore features of template compounds

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of filtered Compound objects
        """

        with Pool(self.n_processes) as pool:
            mask = pool.map(self._filter, batch)

        if self.save_results:
            self.save(zip(batch, mask))

        filtered_batch = [
            compound for compound, keep in zip(batch, mask) if keep
        ]

        return filtered_batch

    def save(
        self,
        batch: List[Compound],
        output_file: str = "2D_pharmacophore_filtered.csv",
    ) -> None:
        """
        Save results to csv with SMILES, ID, keep

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "pharmacophore_filtered.csv"
        """

        df = pd.DataFrame.from_records(
            [compound.to_dict() | {"keep": keep} for compound, keep in batch]
        )

        # save to csv
        save_path = os.path.join(self.output_dir, output_file)
        if not os.path.isfile(save_path):
            df.to_csv(save_path, index=False, mode="a")
        else:
            df.to_csv(save_path, index=False, mode="a", header=False)

    def _filter(self, compound: Compound) -> Compound:
        """
        Filter a single compound based on 2D pharmacophore features of template compounds

        Args:
            compound (Compound): compound object to filter

        Returns:
            Compound: filtered compound object
        """

        if compound.descriptors["status"] == "INVALID":
            return None

        flattened_compound = compound.to_dict()
        keep = (
            (
                flattened_compound["total_N_aro_members"]
                == self.template_dict["total_N_aro_members"]
            )
            and (
                flattened_compound["total_N_ali_members"]
                == self.template_dict["total_N_ali_members"]
            )
            and (
                flattened_compound["total_aro_N_count"]
                >= self.template_dict["total_aro_N_count"]
            )
            and (
                flattened_compound["total_aro_O_count"]
                >= self.template_dict["total_aro_O_count"]
            )
            and (
                flattened_compound["total_ali_O_count"]
                >= self.template_dict["total_ali_O_count"]
            )
            and (
                flattened_compound["total_ali_N_count"]
                >= self.template_dict["total_ali_N_count"]
            )(
                flattened_compound["NumHDonors"]
                >= self.template_dict["NumHDonors"]
            )
            and (
                flattened_compound["NumHAcceptors"]
                >= self.template_dict["NumHAcceptors"]
            )
        )

        if keep:
            return True

        return False

    def _preprocess_templates(
        self, template_compounds: List[Compound]
    ) -> Dict:
        """
        Preprocess template compounds to extract 2D pharmacophore features

        Args:
            template_compounds (list): list of template Compound objects

        Returns:
            pd.DataFrame: dataframe containing 2D pharmacophore features
        """

        template_df = pd.concat(
            [compound.to_df() for compound in template_compounds]
        )

        out_dict = template_df.min(axis=0).to_dict()

        return out_dict


class PharmacophoreFilter3D(Filter):
    """
    Filter compounds based on 3D pharmacophore features of a set of template compounds

    Args:
        template_copmounds (List[Compound]): list of template Compound objects to use for filtering

    Returns:
        filtered_batch: list of filtered Compound objects based on 3D pharmacophore features
    """

    def __init__(
        self,
        template_compounds: List[Compound],
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.template_compounds = self._preprocess_templates(
            template_compounds
        )

        self.conformer_generator = Chem.AllChem.ETKDGv2()
        self.conformer_generator.numThreads = 0  # use all threads
        self.conformer_generator.useRandomCoords = (
            True  # use random starting coordinates
        )
        self.conformer_generator.randomSeed = 42

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Filter compounds based on 3D pharmacophore features of template compounds

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of filtered Compound objects
        """

        with Pool(self.n_processes) as pool:
            processed = pool.map(self._process_leads, batch)

        # with Pool(self.n_processes) as pool:
        #     mask = pool.map(self._filter, batch)

        # flatten processed list
        processed = [item for sublist in processed for item in sublist]

        if self.save_results:
            self.save(processed)

        # filtered_batch = [
        #     compound for compound, keep in zip(batch, mask) if keep
        # ]

        return batch

    def save(
        self,
        batch: List[Dict],
        output_file: str = "3D_pharmacophore_filtered.csv",
    ) -> None:
        """
        Save results to csv with SMILES, ID, keep

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "pharmacophore_filtered.csv"
        """

        df = pd.DataFrame(batch)

        # save to csv
        save_path = os.path.join(self.output_dir, output_file)
        if not os.path.isfile(save_path):
            df.to_csv(save_path, index=False, mode="a")
        else:
            df.to_csv(save_path, index=False, mode="a", header=False)

    def _filter(self, compound: Compound) -> Compound:
        """
        Filter a single compound based on 3D pharmacophore features of template compounds

        Args:
            compound (Compound): compound object to filter

        Returns:
            Compound: filtered compound
        """
        return NotImplementedError("Not implemented yet")

    def _process_leads(self, compound: Compound) -> Dict:
        """
        Process lead compounds to extract 3D pharmacophore features

        Args:
            compound (Compound): compound object

        Returns:
            Dict: dictionary containing 3D pharmacophore features
        """
        hydrogen_bonds, midpoints, align_coordinates = (
            self._gather_coordinates(compound)
        )
        if (
            hydrogen_bonds is None
            or midpoints is None
            or align_coordinates is None
        ):
            return None

        conformations = []
        for template in self.template_compounds:

            if len(midpoints) >= 2:
                if len(align_coordinates) == 6:
                    constrains_alignment = list(
                        zip(align_coordinates, template.align_coordinates)
                    )

                    # generate conformers
                    conformations = self._generate_conformers(
                        compound, template.mol, constrains_alignment
                    )
                    for conformation in conformations:
                        idx, conf_mol, score = conformation
                        conf_hbs, conf_mid_pts, _ = self._get_poses(conf_mol)
                        zero_mid_points = utils.find_zero_midpoints(
                            midpoints, conf_mid_pts
                        )
                        tani = Chem.rdShapeHelpers.ShapeTanimotoDist(
                            conf_mol, template.mol
                        )
                        prtr = Chem.rdShapeHelpers.ShapeProtrudeDist(
                            conf_mol, template.mol
                        )
                        if zero_mid_points is not None:
                            template_acc_don_distances = utils.acc_don_dist(
                                zero_mid_points, template.hydrogen_bonds
                            )
                            lead_acc_don_distances = utils.acc_don_dist(
                                zero_mid_points, conf_hbs
                            )
                            acc_don_score = utils.scoring_function(
                                template_acc_don_distances,
                                lead_acc_don_distances,
                                zero_mid_points,
                            )
                            ring_scoring = utils.other_middle_points(
                                template.midpoints,
                                conf_mid_pts,
                                zero_mid_points,
                            )
                            total_score = [
                                x + y
                                for x, y in zip(acc_don_score, ring_scoring)
                            ] / (
                                len(lead_acc_don_distances)
                                + len(template_acc_don_distances)
                            )
                            conformations.append(
                                {
                                    "lead_id": compound.id,
                                    "conformer_idx": idx,
                                    "template_id": template.id,
                                    "tani": tani,
                                    "prtr": prtr,
                                    "conformer_score": score,
                                    "score_0.1": total_score[0],
                                    "score_0.2": total_score[1],
                                    "score_0.3": total_score[2],
                                    "score_0.4": total_score[3],
                                    "score_0.5": total_score[4],
                                }
                            )
            elif len(midpoints) == 1:
                constrains_alignment = list(
                    zip(align_coordinates, template.align_coordinates)
                )
                conformations = self._generate_conformers(
                    compound, template.mol, constrains_alignment
                )
                for conformation in conformations:
                    idx, conf_mol, score = conformation
                    conf_hbs, conf_mid_pts, _ = self._get_poses(conf_mol)
                    template_acc_don_distances = utils.acc_don_dist(
                        midpoints, template.hydrogen_bonds
                    )
                    lead_acc_don_distances = utils.acc_don_dist(
                        midpoints, conf_hbs
                    )
                    zero_mid_points = utils.find_zero_midpoints(
                        midpoints, conf_mid_pts
                    )
                    acc_don_score = utils.scoring_function(
                        template_acc_don_distances,
                        lead_acc_don_distances,
                        midpoints,
                    )
                    conformations.append(
                        {
                            "lead_id": compound.id,
                            "conformer_idx": idx,
                            "template_id": template.id,
                            "tani": None,
                            "prtr": None,
                            "conformer_score": score,
                            "score_0.1": acc_don_score[0],
                            "score_0.2": acc_don_score[1],
                            "score_0.3": acc_don_score[2],
                            "score_0.4": acc_don_score[3],
                            "score_0.5": acc_don_score[4],
                        }
                    )

        return conformations

    def _generate_conformers(
        self, compound: Compound, template: Compound, constraints: List[Tuple]
    ) -> Compound:
        """
        Generate conformers for compound based on template

        Args:
            compound (Compound): compound object
            template (Compound): template compound object
            constraints (List[Tuple]): list of constraints for alignment

        Returns:
            Compound: compound object with generated conformers
        """

        lead = Chem.MolFromPDBFile(compound.pdb_path, removeHs=False)
        lead = Chem.AllChem.AssignBondOrdersFromTemplate(compound.mol, lead)
        drug = Chem.MolFromPDBFile(template.pdb_path, removeHs=False)
        drug = Chem.AllChem.AssignBondOrdersFromTemplate(template.mol, drug)

        conf_num = []
        num_conformers = Chem.AllChem.EmbedMultipleConfs(
            compound.mol, 100, self.conformer_generator
        )
        for i, conf in enumerate(num_conformers):
            mol_with_conf = Chem.Mol(
                lead
            )  # Create a copy of the original molecule
            mol_with_conf.RemoveAllConformers()  # Remove any existing conformers
            mol_with_conf.AddConformer(
                lead.GetConformer(conf), assignId=True
            )  # Add the desired conformer
            o3d = Chem.rdMolAlign.GetO3A(
                mol_with_conf, drug, constraintMap=constraints
            )
            o3d.Align()
            score = o3d.Score()
            conf_num.append([i, mol_with_conf, score])
        return conf_num

    def _gather_coordinates(self, compound: Compound) -> Compound:
        """
        Gather 3D coordinates for compound

        Args:
            compound (Compound): compound object

        Returns:
            Dict: dictionary containing 3D coordinates
        """

        # get poses
        hydrogen_bonds = []
        try:
            pdb_mol = Chem.MolFromPDBFile(compound.pdb_path, removeHs=False)
            pdb_mol = Chem.AllChem.AssignBondOrdersFromTemplate(
                compound.mol, pdb_mol
            )
        except Exception as e:
            print(f"Error processing Mol from pdb for {compound.id}: {e}")
            return None, None, None

        try:
            # get hydrogen bond coordinates
            for idx in range(pdb_mol.GetNumAtoms()):
                symbol = pdb_mol.GetAtomWithIdx(idx).GetSymbol()
                if symbol == "H" or symbol == "O":
                    conformer = pdb_mol.GetConformer(0)
                    coords = conformer.GetAtomPosition(idx)
                    hydrogen_bonds.append(coords)

            # get midpoints of ring systems
            midpoints = []
            for idx, system in enumerate(compound.ring_systems):
                ring_indices = system
                x_sum = sum(
                    pdb_mol.GetConformer(0).GetAtomPosition(i).x
                    for i in ring_indices
                )
                y_sum = sum(
                    pdb_mol.GetConformer(0).GetAtomPosition(i).y
                    for i in ring_indices
                )
                z_sum = sum(
                    pdb_mol.GetConformer(0).GetAtomPosition(i).z
                    for i in ring_indices
                )

                middle_point = Point3D(
                    x_sum / len(ring_indices),
                    y_sum / len(ring_indices),
                    z_sum / len(ring_indices),
                )
                midpoints.append(middle_point)

            # align coordinates
            align_coordinates = []
            if len(compound.ring_systems) == 3:
                ring = list(compound.ring_systems[1])
                atom = pdb_mol.GetAtomWithIdx(ring[0])
                if atom.GetIsAromatic():
                    align_coordinates.append(ring)
            elif len(compound.ring_systems) == 2:
                for ring in compound.ring_systems:
                    ring = list(ring)
                    atom = pdb_mol.GetAtomWithIdx(ring[0])
                    is_aromatic = atom.GetIsAromatic()
                    if is_aromatic:
                        align_coordinates.append(ring)
            align_coordinates = align_coordinates[0]

        except Exception as e:
            print(f"Error processing coordinates for {compound.id}: {e}")
            return None, None, None

        return hydrogen_bonds, midpoints, align_coordinates

    def _process_templates(self, template_compounds: List[Compound]) -> Dict:
        """
        Process template compounds to extract 3D pharmacophore features

        Args:
            template_compounds (list): list of template Compound objects

        Returns:
            List[Compound]: list of Compound objects with 3D pharmacophore features added as attributes
        """

        for template in template_compounds:
            hydrogen_bonds, midpoints, align_coordinates = (
                self._gather_coordinates(template)
            )
            template.hydrogen_bonds = hydrogen_bonds
            template.midpoints = midpoints
            template.align_coordinates = align_coordinates

        return template_compounds

    def _get_poses(self, mol: Chem.Mol):

        # get hydrogen bond coordinates
        hydrogen_bonds = []
        for index in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(index)
            atom_symbol = atom.GetSymbol()
            if atom_symbol == "N" or atom_symbol == "O":
                conf = mol.GetConformer(0)
                coord = conf.GetAtomPosition(index)
                hydrogen_bonds.append(coord)

        # get ring systems
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon and (nInCommon > 1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems

        # get midpoints
        mid_points = []
        for index, ring_indices in enumerate(systems):
            x_sum = sum(
                mol.GetConformer(0).GetAtomPosition(i).x for i in ring_indices
            )
            y_sum = sum(
                mol.GetConformer(0).GetAtomPosition(i).y for i in ring_indices
            )
            z_sum = sum(
                mol.GetConformer(0).GetAtomPosition(i).z for i in ring_indices
            )

            middle_point = Point3D(
                x_sum / len(ring_indices),
                y_sum / len(ring_indices),
                z_sum / len(ring_indices),
            )
            mid_points.append(middle_point)

        return hydrogen_bonds, mid_points, systems


class SminaDockingFilter(Filter):
    """
    Filter compounds based on docking scores using Smina

    Args:
        protein_code (str): PDB code for target protein
        optimized_pdb_dir (str): path to optimized pdb files
        protein_path (str): path to protein PDB
        n_processes (int): number of processes to use for filtering

    Config:
        save_results (bool): save filtered compounds to csv. Default is False
        optimized_pdb_dir (str): path to optimized pdb files
        load_protein (bool): load protein file for docking. If False, fetch from RCSB


    Returns:
        filtered_batch: list of filtered Compound objects based on docking scores
    """

    def __init__(
        self,
        protein_code: str,
        optimized_pdb_dir: str,
        protein_path: str = None,
        output_dir: str = None,
        n_processes: int = 1,
        save_results: bool = False,
    ):
        super().__init__(output_dir, n_processes, save_results)

        self.protein_code = protein_code
        self.optimized_pdb_dir = optimized_pdb_dir
        self.protein_path = protein_path

    def run(self, batch: List[Compound]) -> List[Compound]:
        """
        Filter compounds based on docking scores using Smina

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of filtered Compound objects
        """

        # load protein
        self._load_and_preprocess_protein()

        for compound in batch:
            cmd.load(
                filename=self.protein_code + "_lig.mol2",
                format="mol2",
                object="Lig",
            )
            center, size = self._get_box(selection="Lig", extending=5.0)

            smina_cmd = [
                "smina",
                "--receptor",
                self.protein_code + ".pdbqt",
                "--ligand",
                compound.pdbqt_path,
                "--center_x",
                str(center["center_x"]),
                "--center_y",
                str(center["center_y"]),
                "--center_z",
                str(center["center_z"]),
                "--size_x",
                str(size["size_x"]),
                "--size_y",
                str(size["size_y"]),
                "--size_z",
                str(size["size_z"]),
                "--num_modes",
                "1",
                "--exhaustiveness",
                "8",
                "-o",
                os.path.join(self.output_dir, compound.id + "_docked.pdb"),
                "--scoring",
                "vinardo",
            ]

            try:
                subprocess.call(smina_cmd)
            except Exception as e:
                print(f"Error running smina for {compound.id}: {e}")
                continue

            compound.docked_pdb_path = os.path.join(
                self.output_dir, compound.id + "_docked.pdb"
            )

        if self.save_results:
            self.save(batch)

        return batch

    def save(
        self,
        batch: List[Compound],
        output_file: str = "docking_filtered.csv",
    ) -> None:
        """
        Save results to csv with SMILES, ID, docked_pdb_file

        Args:
            batch (List[Compound]): list of Compound objects
            output_file (str): output file path. Default is "docking_filtered.csv"
        """

        df = pd.DataFrame.from_records(
            [
                compound.to_dict()
                | {"docked_pdb_file": compound.docked_pdb_path}
                for compound in batch
            ]
        )

        # save to csv
        save_path = os.path.join(self.output_dir, output_file)
        if not os.path.isfile(save_path):
            df.to_csv(save_path, index=False, mode="a")
        else:
            df.to_csv(save_path, index=False, mode="a", header=False)

    def _load_and_preprocess_protein(
        self, addHs_pH=7.4, renumber_residues=True
    ):
        """
        Load receptor file and preprocess for docking
        """

        output_protein_path = self.protein_code + ".pdb"
        output_protein_path_clean = self.protein_code + "_clean.pdb"
        output_ligand_path = self.protein_code + "_lig.mol2"

        # clear pymol session
        cmd.delete("all")

        if self.protein_path is not None:
            cmd.load(self.protein_path)
        else:
            cmd.fetch(code=self.protein_code, type="pdb1")

        cmd.select(name="Prot", selection="polymer.protein")
        cmd.select(name="Lig", selection="organic")
        cmd.save(
            filename=output_protein_path,
            format="pdb",
            selection="Prot",
        )
        cmd.save(
            filename=output_ligand_path,
            format="mol2",
            selection="Lig",
        )
        cmd.delete("all")

        # fix protein
        fixer = PDBFixer(filename=output_protein_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(addHs_pH)

        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(output_protein_path_clean, "w"),
        )

        # renumber residues
        if renumber_residues:
            try:
                original = mda.Universe(output_protein_path)
                from_fix = mda.Universe(output_protein_path_clean)

                resNum = [res.resid for res in original.residues]
                for idx, res in enumerate(from_fix.residues):
                    res.resid = resNum[idx]

                save = PDB.PDBWriter(filename=output_protein_path_clean)
                save.write(from_fix)
                save.close()
            except Exception as e:
                print(f"Not possible to renumber residues with error: {e}")

        # clean ligand
        mol = [
            m
            for m in pybel.readfile(filename=output_ligand_path, format="mol2")
        ][0]
        mol.addh()

        with pybel.Outputfile(
            filename=output_ligand_path, format="mol2", overwrite=True
        ) as out:
            out.write(mol)

        # convert to pdbqt
        subprocess.call(
            f"obabel -ipdb {output_protein_path_clean} -opdbqt -O {self.protein_code + '.pdbqt'}",
            shell=True,
        )
