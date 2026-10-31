"""Target Task Classifier (TTC)."""
from .classifiers import MLPClassifier
from .config import PSDFConfig


class TTC(MLPClassifier):
    """Predict motor imagery classes from task-token features."""

    def __init__(self, cfg: PSDFConfig) -> None:
        dropout = cfg.dropout if cfg.classifier_dropout is None else cfg.classifier_dropout
        super().__init__(cfg.embed_dim, cfg.n_classes, dropout, cfg.classifier_hidden)
