from dataclasses import asdict, dataclass


@dataclass
class PSDFConfig:
    fs: int = 250
    n_channels: int = 22
    n_classes: int = 4
    segment_seconds: float = 1.0
    overlap_seconds: float = 0.0
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    pretrained_vit: bool = False
    classifier_hidden: int | None = None
    classifier_dropout: float | None = None
    num_aug_ops: int = 6  # PDC output classes: identity plus five operators.
    spectral_transform: str = "ortho_power"
    normalize_content: bool = True
    channel_embedding_std: float = 1.0

    def __post_init__(self):
        if min(self.fs, self.n_channels, self.embed_dim, self.depth, self.num_heads) <= 0:
            raise ValueError("Model dimensions and fs must be positive")
        if self.n_classes < 2 or self.embed_dim % self.num_heads:
            raise ValueError("Require >=2 classes and embed_dim divisible by num_heads")
        if not 0 <= self.overlap_seconds < self.segment_seconds:
            raise ValueError("Require 0 <= overlap_seconds < segment_seconds")
        if self.mlp_ratio <= 0 or self.layer_norm_eps <= 0 or self.channel_embedding_std <= 0:
            raise ValueError("MLP ratio, normalization epsilon and embedding scale must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.classifier_hidden is not None and self.classifier_hidden < 1:
            raise ValueError("classifier_hidden must be positive")
        if self.classifier_dropout is not None and not 0 <= self.classifier_dropout < 1:
            raise ValueError("classifier_dropout must lie in [0, 1)")
        if self.spectral_transform not in ("raw_power", "ortho_power", "log_power"):
            raise ValueError("Unknown spectral transform")
        if self.num_aug_ops != 6:
            raise ValueError("PDC requires six output classes")

    def to_dict(self):
        return asdict(self)
