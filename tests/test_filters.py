import pytest
from filters import Filter
from data import Compound


class SimpleFilter(Filter):
    def run(self, batch):
        # A simple filter that filters out compounds with MolWt > 50
        return [compound for compound in batch if compound.mol_weight <= 50]


@pytest.fixture
def compounds():
    return [Compound("H2O"), Compound("CCO"), Compound("CCCC")]


def test_simple_filter(compounds):
    simple_filter = SimpleFilter()
    filtered = simple_filter(compounds)
    assert len(filtered) == 2, "Should filter out compounds with MolWt > 50"
