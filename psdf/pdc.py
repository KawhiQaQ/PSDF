"""Augmentation-prediction head for variation-sensitive representations."""
from .classifiers import MLPClassifier
from .config import PSDFConfig


class PDC(MLPClassifier):
    """Augmentation-prediction head for variation-sensitive representations."""

    def __init__(self, cfg: PSDFConfig) -> None:
        dropout = cfg.dropout if cfg.classifier_dropout is None else cfg.classifier_dropout
        super().__init__(cfg.embed_dim, cfg.num_aug_ops, dropout, cfg.classifier_hidden)
