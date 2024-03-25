from typing import Any, List
from .data import Compound
from .filters import Filter

class Pipeline:
    """
    Pipeline interface for processing Compounds through a list of Filters.

    Args:
        filters (List[Filter]): list of Filters to apply to the Compounds
    """
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def __call__(self, compounds: List[Compound], *args: Any, **kwargs: Any) -> Any:
        return self.run(compounds, *args, **kwargs)

    def run(self, compounds: List[Compound], *args: Any, **kwargs: Any) -> Any:
        return NotImplementedError("This method should be implemented in the subclass.")



class BasicCompoundPipeline(Pipeline):
    """
    Basic implementation of a Compound Pipeline
    """
    def __init__(self, filters: List[Filter]):
        super().__init__(filters)

    def run(self, compounds: List[Compound]) -> List[Compound]:
        for filter in self.filters:
            compounds = filter(compounds)
        return compounds