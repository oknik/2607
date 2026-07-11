from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install dependencies with: pip install -r requirements.txt") from exc


class TwoStreamVit(nn.Module):
    """Shared-backbone C/G two-stream ViT classifier."""

    def __init__(
        self,
        backbone: str = "deit_tiny_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
        share_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.share_backbone = share_backbone
        self.c_backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.c_backbone.num_features
        if share_backbone:
            self.g_backbone = self.c_backbone
        else:
            self.g_backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward(self, c_img: torch.Tensor, g_img: torch.Tensor) -> torch.Tensor:
        c_feat = self.c_backbone(c_img)
        g_feat = self.g_backbone(g_img)
        return self.head(torch.cat([c_feat, g_feat], dim=1))
