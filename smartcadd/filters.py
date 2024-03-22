import pandas as pd
from multiprocessing import Pool, cpu_count

from model_wrappers import ModelWrapper
from data import Compound, SMARTS_Query


class Filter:
    def __init__(self, filter_config):
        self.filter_config = filter_config

    def run(self, batch):
        pass


class ADMETFilter(Filter):
    """
    Filter compounds based on ADMET PAINS patterns

    Args:
    filter_config (dict): configuration for filter
    n_processes (int): number of processes to use for filtering

    """

    def __init__(self, filter_config, n_processes=1):
        super().__init__(filter_config)
        self.n_processes = n_processes

        assert (
            "alert_collection_path" in filter_config.keys()
        ), "alert_collection_path not found in filter_config"

        self._pains_patterns = self._init(
            alert_collection_path=filter_config["alert_collection_path"]
        )

    def run(self, batch):
        """
        Filters compounds based on ADMET PAINS patterns

        Args:
            batch: list of Compound objects

        Returns:
            filtered_batch: list of filtered Compound objects
        """

        with Pool(self.n_processes) as pool:
            filtered_batch = pool.map(self._filter, batch)

        return filtered_batch

    def _init(self, alert_collection_path):
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

    def _filter(self, compound):
        """
        Filter a single compound based on ADMET PAINS patterns
        """

        if compound.descriptors["status"] == "INVALID":
            return compound

        for smarts_query in self._pains_patterns:
            if (
                len(compound.mol.GetSubstructMatches(smarts_query.pattern))
                > smarts_query.max_val
            ):
                compound.descriptors["status"] = (
                    f"{smarts_query.desc} > {smarts_query.max_val}"
                )
                break

        return compound


class ModelFilter(Filter):
    """
    Filter compounds based on model prediction probabilities

    Args:
        filter_config (dict): configuration for filter
        threshold (float): threshold for filtering
    """

    def __init__(self, model_wrapper, filter_config, target, threshold=0.5):
        super().__init__(filter_config)

        self.model_wrapper = model_wrapper
        self.target = target
        self.threshold = threshold

        # load model weights
        try:
            self.model_wrapper.load()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e

    def predict(self, batch):
        """
        Predict on input batch using the model.

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of prediction probabilities

        """

        featurized_batch = self.model.featurize(batch)

        return self.model_wrapper.predict(featurized_batch, self.target)

    def run(self, batch):
        """
        Filter compounds based on model prediction probabilities

        Args:
            batch (list): list of Compound objects

        Returns:
            list: list of filtered Compound objects

        """

        predictions = self.predict(batch, self.target)

        # filter based on threshold
        filtered_batch = [
            compound
            for idx, compound in enumerate(batch)
            if predictions[idx] > self.threshold
        ]

        return filtered_batch
