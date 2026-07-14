from __future__ import annotations

import torch
from torch import nn

from .local_vim import create_vim_model


class TwoStreamMamba(nn.Module):
    """Shared-backbone C/G two-stream vision Mamba classifier."""

    def __init__(
        self,
        backbone: str = "vim_tiny_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
        share_backbone: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.share_backbone = share_backbone
        self.c_backbone = self._create_backbone(backbone, pretrained, image_size)
        feature_dim = self.c_backbone.num_features
        self.g_backbone = self.c_backbone if share_backbone else self._create_backbone(backbone, pretrained, image_size)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, num_classes),
        )

    @staticmethod
    def _create_backbone(backbone: str, pretrained: bool, image_size: int):
        return create_vim_model(backbone=backbone, num_classes=0, img_size=image_size, pretrained=pretrained)

    def forward(self, c_img: torch.Tensor, g_img: torch.Tensor) -> torch.Tensor:
        c_feat = self.c_backbone(c_img)
        g_feat = self.g_backbone(g_img)
        return self.head(torch.cat([c_feat, g_feat], dim=1))
