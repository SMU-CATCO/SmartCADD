from types import MethodType
from typing import List, Dict, Any
import numpy as np
from deepchem.data import NumpyDataset
from deepchem.models import AttentiveFPModel

from .data import Compound


class ModelWrapper:
    """
    Base wrapper class for prediction models

    Args:
    model_params_path (str): path to model parameters
    **kwargs: keyword arguments required for model

    """

    def __init__(self, model_params_path: str, **kwargs):
        self.model_params_path = model_params_path

        self.model = None

    def predict(self, batch: List[Compound], target: int) -> List[float]:
        """
        Predict on input data using the model
        """
        return NotImplementedError(
            "This method should be implemented in the subclass"
        )

    def featurize(self, batch: List[Compound]) -> Any:
        """
        Featurize input data using the model
        """
        return NotImplementedError(
            "This method should be implemented in the subclass"
        )

    def load(self):
        """
        Load the model from checkpoint
        """
        return NotImplementedError(
            "This method should be implemented in the subclass"
        )


class AttentiveFPWrapper(ModelWrapper):
    """
    Wrapper class for AttentiveFP model

    Args:
    model_config (dict): configuration for model
    model_params_path (str): path to model parameters

    """

    def __init__(self, model_params_path: str, **kwargs):

        super().__init__(model_params_path)

        self.model = AttentiveFPModel(**kwargs)
        self.load()

    def predict(self, batch: List[Compound], target: int) -> List[float]:
        """
        Predict on input featurized batch using the model
        """

        if isinstance(batch, Compound):
            batch = self.featurize(batch)

        predictions = self.model.predict(batch)

        return [p[target] for p in predictions]

    def featurize(self, batch: List[Compound]) -> Any:
        """
        Featurize input data using the model
        """
        graph_features = [compound.graph_data for compound in batch]

        return NumpyDataset(X=graph_features, y=np.ones(len(graph_features)))

    def load(self):
        """
        Load the model from checkpoint
        """

        self.model.restore(self.model_params_path)


class AttentiveFP_DGLModelWrapper(AttentiveFPWrapper):
    def __init__(self, model_params_path: str, **kwargs):

        super().__init__(model_params_path, **kwargs)

        self.dgl_model = self._init_dgl(self.model)

        self.model = self.dgl_model.model
        self._gnn = self.dgl_model.model.gnn
        self._readout = self.dgl_model.model.readout
        self._predict = self.dgl_model.model.predict

    def forward(self, g, nfeats):
        node_feats = g.ndata[self.model.nfeat_name]
        edge_feats = g.edata[self.model.efeat_name]

        out = self._gnn(g, node_feats, edge_feats)
        out = self._readout(g, out)
        out = self._predict(out)

        return out

    def _init_dgl(self, model):

        def get_embeddings(self, g):
            model.eval()

            node_feats = g.ndata[model.nfeat_name]
            edge_feats = g.edata[model.efeat_name]

            out = model.model.gnn(g, node_feats, edge_feats)
            out = model.model.readout(g, out)

            return out.detach().numpy()

        model.output_types = ["prediction", "loss", "embedding"]
        model._other_outputs = [2]
        model.get_embeddings = MethodType(get_embeddings, model)

        return model
