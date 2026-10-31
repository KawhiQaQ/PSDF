from __future__ import annotations

import torch
from torch import nn

from .config import PSDFConfig
from .fmie import FMIE
from .backbone import ViTBackbone
from .ttc import TTC
from .pdc import PDC


class PSDFModel(nn.Module):
    def __init__(self, cfg: PSDFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fmie = FMIE(cfg)
        self.backbone = ViTBackbone(cfg)
        self.task_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.dis_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.ttc = TTC(cfg)
        self.pdc = PDC(cfg)
        nn.init.trunc_normal_(self.task_token, std=0.02)
        nn.init.trunc_normal_(self.dis_token, std=0.02)
        if cfg.pretrained_vit:
            self.load_timm_vit_base_weights()

    def load_timm_vit_base_weights(self) -> None:
        if (self.cfg.embed_dim, self.cfg.depth, self.cfg.num_heads, self.cfg.mlp_ratio) != (768, 12, 12, 4.0):
            raise ValueError("pretrained_vit requires ViT-B dimensions: dim=768, depth=12, heads=12")
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to load ImageNet ViT-B weights") from exc

        vit = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
        state = vit.state_dict()
        with torch.no_grad():
            cls_token = state.get("cls_token")
            if cls_token is not None and cls_token.shape[-1] == self.cfg.embed_dim:
                self.task_token.copy_(cls_token)
                self.dis_token.copy_(cls_token)

            for idx, block in enumerate(self.backbone.blocks):
                prefix = f"blocks.{idx}."
                block.norm1.weight.copy_(state[prefix + "norm1.weight"])
                block.norm1.bias.copy_(state[prefix + "norm1.bias"])
                qkv_weight = state[prefix + "attn.qkv.weight"].chunk(3, dim=0)
                qkv_bias = state[prefix + "attn.qkv.bias"].chunk(3, dim=0)
                block.attn.q_proj.weight.copy_(qkv_weight[0])
                block.attn.k_proj.weight.copy_(qkv_weight[1])
                block.attn.v_proj.weight.copy_(qkv_weight[2])
                block.attn.q_proj.bias.copy_(qkv_bias[0])
                block.attn.k_proj.bias.copy_(qkv_bias[1])
                block.attn.v_proj.bias.copy_(qkv_bias[2])
                block.attn.out_proj.weight.copy_(state[prefix + "attn.proj.weight"])
                block.attn.out_proj.bias.copy_(state[prefix + "attn.proj.bias"])
                block.norm2.weight.copy_(state[prefix + "norm2.weight"])
                block.norm2.bias.copy_(state[prefix + "norm2.bias"])
                block.ffn[0].weight.copy_(state[prefix + "mlp.fc1.weight"])
                block.ffn[0].bias.copy_(state[prefix + "mlp.fc1.bias"])
                block.ffn[3].weight.copy_(state[prefix + "mlp.fc2.weight"])
                block.ffn[3].bias.copy_(state[prefix + "mlp.fc2.bias"])

            self.backbone.norm.weight.copy_(state["norm.weight"])
            self.backbone.norm.bias.copy_(state["norm.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map EEG [batch, channels, time] to logits [batch, n_classes]."""
        return self.forward_task(x)[0]

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.fmie(x)

    def forward_task_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.fmie.add_identity(tokens)
        bsz = tokens.shape[0]
        cls = self.task_token.expand(bsz, -1, -1)
        out = self.backbone(torch.cat([cls, tokens], dim=1))
        feat = out[:, 0]
        return self.ttc(feat), feat

    def forward_dis_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.fmie.add_identity(tokens)
        bsz = tokens.shape[0]
        cls = self.dis_token.expand(bsz, -1, -1)
        out = self.backbone(torch.cat([cls, tokens], dim=1))
        feat = out[:, 0]
        return self.pdc(feat), feat

    def forward_task(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_task_tokens(self.encode_tokens(x))

    def forward_dis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits [batch, num_aug_ops] and features [batch, embed_dim]."""
        return self.forward_dis_tokens(self.encode_tokens(x))
