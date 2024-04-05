import pytest
from data import Compound


@pytest.fixture
def ethanol():
    return Compound("CCO")


def test_ethanol_properties(ethanol):
    assert (
        ethanol.mol_weight > 0
    ), "Molecular weight should be positive for ethanol"
    assert ethanol.log_p > 0, "LogP should be positive for ethanol"
    assert (
        ethanol.rotatable_bonds == 0
    ), "Ethanol should have 0 rotatable bonds"
    assert ethanol.smiles == "CCO", "SMILES representation should be 'CCO'"
    assert ethanol.id == "ethanol", "ID should be 'ethanol'"
    assert ethanol.mol is not None, "Molecule should not be None"
    assert ethanol.descriptors is not None, "Descriptors should not be None"
    assert (
        ethanol.descriptors["NumHDonors"] == 1
    ), "Ethanol should have 1 H donor"
    assert (
        ethanol.descriptors["NumHAcceptors"] == 1
    ), "Ethanol should have 1 H acceptor"
    assert ethanol.descriptors["status"] == "VALID", "Status should be 'VALID'"


def test_str(ethanol):
    assert str(ethanol) == "ethanol", "__str__ should return the ID"


def test_to_dict(ethanol):
    assert (
        ethanol.to_dict()["id"] == "ethanol"
    ), "to_dict() should return a dictionary with ID"
    assert (
        ethanol.to_dict()["mol_weight"] > 0
    ), "to_dict() should return a dictionary with mol_weight"
    assert (
        ethanol.to_dict()["log_p"] > 0
    ), "to_dict() should return a dictionary with log_p"
    assert (
        ethanol.to_dict()["rotatable_bonds"] == 0
    ), "to_dict() should return a dictionary with rotatable_bonds"
    assert (
        ethanol.to_dict()["descriptors"] is not None
    ), "to_dict() should return a dictionary with descriptors"
    assert (
        ethanol.to_dict()["descriptors"]["NumHDonors"] == 1
    ), "to_dict() should return a dictionary with NumHDonors"
    assert (
        ethanol.to_dict()["descriptors"]["NumHAcceptors"] == 1
    ), "to_dict() should return a dictionary with NumHAcceptors"
    assert (
        ethanol.to_dict()["descriptors"]["status"] == "VALID"
    ), "to_dict() should return a dictionary with status"
    assert (
        ethanol.to_dict()["mol"] is not None
    ), "to_dict() should return a dictionary with mol"
    assert (
        ethanol.to_dict()["mol"] == ethanol.mol
    ), "to_dict() should return a dictionary with mol"


def test_to_df(ethanol):
    assert ethanol.to_df().shape == (
        1,
        1,
    ), "to_df() should return a DataFrame with 1 row"
    assert ethanol.to_df().columns == [
        "smiles"
    ], "to_df() should return a DataFrame with 'smiles' column"
    assert (
        ethanol.to_df().iloc[0, 0] == "CCO"
    ), "to_df() should return a DataFrame with 'CCO' in the first row"


def test_invalid_smiles():
    with pytest.raises(ValueError):
        Compound("InvalidSMILES")
