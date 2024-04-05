from typing import Any, List, Iterable
from .data import Compound
from .filters import Filter
from .dataset import IterableDataset


class Pipeline:
    """
    Pipeline interface for processing Compounds through a list of Filters.

    Args:
        filters (List[Filter]): list of Filters to apply to the Compounds
    """

    def __init__(self, data_loader: IterableDataset, filters: List[Filter]):
        self.data_loader = data_loader
        self.filters = filters

    def __call__(
        self, compounds: List[Compound], *args: Any, **kwargs: Any
    ) -> Any:
        return self.run(compounds, *args, **kwargs)

    def __str__(self) -> str:
        string = (
            f"\n{self.__class__.__name__}:"
            + f" [ {' -> '.join([f'({i+1}) {str(f)}' for i,f in enumerate(self.filters)])}"
            + " -> Done ]"
        )
        return string

    def run_filters(
        self, compounds: List[Compound], *args: Any, **kwargs: Any
    ) -> Any:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def append_filter(self, filter: Filter) -> Any:
        self.filters.append(filter)

    def get_data(self) -> Iterable[Compound]:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )

    def summary(self) -> str:
        print(self.__str__())

    def save_results(self, results: Any, *args: Any, **kwargs: Any) -> None:
        return NotImplementedError(
            "This method should be implemented in the subclass."
        )


class BasicCompoundPipeline(Pipeline):
    """
    Basic implementation of a Compound Pipeline
    """

    def __init__(self, data_loader: IterableDataset, filters: List[Filter]):
        super().__init__(data_loader, filters)

    def run_filters(self, compounds: List[Compound]) -> List[Compound]:
        for filter in self.filters:
            compounds = filter(compounds)
        return compounds

    def get_data(self) -> Iterable[Compound]:
        for batch in self.data_loader:
            yield batch
