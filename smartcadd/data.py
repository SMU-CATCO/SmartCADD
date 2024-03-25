from typing import List, Dict
from rdkit import Chem
from rdkit.Chem.Descriptors import (
    MolWt,
    MolLogP,
    NumHDonors,
    NumHAcceptors,
    TPSA,
)
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.Lipinski import NumAromaticRings
from deepchem.feat import MolGraphConvFeaturizer
import numpy as np
import pandas as pd


class Compound(object):
    """
    Class to represent a chemical compound

    Args:
    smiles (str): SMILES representation of the compound
    id (str): unique identifier for the compound


    """

    def __init__(self, smiles: str, id: str):
        self.smiles = smiles
        self.id = id

        self.mol = Chem.MolFromSmiles(self.smiles)
        self.descriptors = self._compute_descriptors()
        self.ring_system_descriptors = self._compute_ring_system_descriptors()
        self.graph_data = self._featurize()

    def _compute_descriptors(self):

        if self.mol is None:
            return {
                "status": "INVALID",
                "MolWt": -999,
                "MolLogP": -999,
                "NumHDonors": -999,
                "NumHAcceptors": -999,
                "TPSA": -999,
                "CalcNumRotatableBonds": -999,
                "NumAromaticRings": -999,
            }
        else:
            return {
                "status": "OK",
                "MolWt": MolWt(self.mol),
                "MolLogP": MolLogP(self.mol),
                "NumHDonors": NumHDonors(self.mol),
                "NumHAcceptors": NumHAcceptors(self.mol),
                "TPSA": TPSA(self.mol),
                "CalcNumRotatableBonds": CalcNumRotatableBonds(self.mol),
                "NumAromaticRings": NumAromaticRings(self.mol),
            }

    def _compute_ring_system_descriptors(self):

        total_N_aro_members = total_aro_N_count = total_aro_O_count = 0
        total_N_ali_members = total_ali_N_count = total_ali_O_count = 0

        # Use RDKit's built-in method to find ring info
        ring_info = self.mol.GetRingInfo()

        # Gets atom indices in each ring
        atom_rings = ring_info.AtomRings()

        # Iterate over rings
        for ring in atom_rings:
            for idx in ring:
                atom = self.mol.GetAtomWithIdx(idx)
                atom_symbol = atom.GetSymbol()
                is_aromatic = atom.GetIsAromatic()

                if is_aromatic:
                    total_N_aro_members += 1
                    if atom_symbol == "N":
                        total_aro_N_count += 1
                    elif atom_symbol == "O":
                        total_aro_O_count += 1
                else:
                    total_N_ali_members += 1
                    if atom_symbol == "N":
                        total_ali_N_count += 1
                    elif atom_symbol == "O":
                        total_ali_O_count += 1

        return {
            "total_N_aro_members": total_N_aro_members,
            "total_N_ali_members": total_N_ali_members,
            "total_aro_N_count": total_aro_N_count,
            "total_aro_O_count": total_aro_O_count,
            "total_ali_O_count": total_ali_O_count,
            "total_ali_N_count": total_ali_N_count,
        }

    def _featurize(self):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        return featurizer.featurize(self.mol)

    def __str__(self) -> str:
        return str(
            {
                "smiles": self.smiles,
                "id": self.id,
                "descriptors": self.descriptors,
                "ring_system_descriptors": self.ring_system_descriptors,
            }
        )

    def __repr__(self) -> str:
        return str(
            {
                "smiles": self.smiles,
                "id": self.id,
                "descriptors": self.descriptors,
                "ring_system_descriptors": self.ring_system_descriptors,
            }
        )

    def to_dict(self):
        return {
            "smiles": self.smiles,
            "id": self.id,
            **self.descriptors,
            **self.ring_system_descriptors,
        }

    def to_df(self):
        return pd.DataFrame.from_dict(self.to_dict(), orient="index").T


class SMARTS_Query(object):
    def __init__(self, smarts: str, max_val: float, desc: bool):
        self.smarts = smarts
        self.max_val = max_val
        self.desc = desc

        self.pattern = Chem.MolFromSmarts(self.smarts)

    def __str__(self) -> str:
        return str(
            {
                "smarts": self.smarts,
                "max_val": self.max_val,
                "desc": self.desc,
            }
        )
