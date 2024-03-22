import numpy as np

from data import Compound


class ModelWrapper:
    """
    Base wrapper class for prediction models

    Args:
    model (str): model name
    model_params_path (str): path to model parameters
    **kwargs: keyword arguments required for model

    """

    def __init__(self, model_params_path, **kwargs):
        self.model_params_path = model_params_path

        self.model = None

    def predict(self, batch, target):
        """
        Predict on input data using the model
        """
        return NotImplementedError

    def featurize(self, batch):
        """
        Featurize input data using the model
        """
        raise NotImplementedError

    def load(self):
        """
        Load the model from checkpoint
        """
        raise NotImplementedError


class AttentiveFPWrapper(ModelWrapper):
    """
    Wrapper class for AttentiveFP model

    Args:
    model (str): model name
    model_config (dict): configuration for model
    model_params_path (str): path to model parameters

    """

    def __init__(self, model_params_path, **kwargs):
        from deepchem.models import AttentiveFPModel

        super().__init__(model_params_path)

        self.model = AttentiveFPModel(**kwargs)

    def predict(self, batch, target):
        """
        Predict on input featurized batch using the model
        """

        if isinstance(batch[0], Compound):
            batch = self.featurize(batch)

        predictions = self.model.predict(batch)s

        return [p[target] for p in predictions]

    def featurize(self, batch):
        from deepchem.data import NumpyDataset

        """
        Featurize input data using the model
        """
        graph_features = [compound.graph_features for compound in batch]

        return NumpyDataset(graph_features, np.ones(len(graph_features)))

    def load(self):
        """
        Load the model from checkpoint
        """

        self.model.restore(self.model_params_path)
