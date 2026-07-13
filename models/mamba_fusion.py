from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install dependencies with: pip install -r requirements.txt") from exc


class TwoStreamMamba(nn.Module):
    """Shared-backbone C/G two-stream vision Mamba classifier."""

    def __init__(
        self,
        backbone: str = "vim_tiny_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
        share_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.share_backbone = share_backbone
        self.c_backbone = self._create_backbone(backbone, pretrained)
        feature_dim = self.c_backbone.num_features
        self.g_backbone = self.c_backbone if share_backbone else self._create_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, num_classes),
        )

    @staticmethod
    def _create_backbone(backbone: str, pretrained: bool):
        try:
            return timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        except RuntimeError as exc:
            candidates = timm.list_models("*mamba*") + timm.list_models("*vim*")
            hint = ", ".join(candidates[:20]) if candidates else "no mamba/vim models found in your timm version"
            raise RuntimeError(
                f"Cannot create Mamba backbone '{backbone}'. "
                f"Try upgrading timm or pass one of the available model names with --backbone. "
                f"Available candidates: {hint}"
            ) from exc

    def forward(self, c_img: torch.Tensor, g_img: torch.Tensor) -> torch.Tensor:
        c_feat = self.c_backbone(c_img)
        g_feat = self.g_backbone(g_img)
        return self.head(torch.cat([c_feat, g_feat], dim=1))
