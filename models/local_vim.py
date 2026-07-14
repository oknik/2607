from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=192):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(f"Input image size {(height, width)} does not match model size {self.img_size}.")
        return self.proj(x).flatten(2).transpose(1, 2)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.float()
    C = C.float()
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    state = A.new_zeros((batch, dim, dstate))
    outputs = []
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    for idx in range(u.shape[2]):
        state = delta_a[:, :, idx] * state + delta_b_u[:, :, idx]
        y = torch.einsum("bdn,bn->bd", state, C[:, :, idx])
        outputs.append(y)
    out = torch.stack(outputs, dim=2)
    if D is not None:
        out = out + u * D[None, :, None]
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


class MambaMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bimamba_type: str = "v2",
        if_divide_out: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        if bimamba_type == "v2":
            self.conv1d_b = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner)
            self.A_b_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
            self.D_b = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()
        self._init_dt()

    def _init_dt(self):
        for dt_proj in [self.dt_proj, getattr(self, "dt_proj_b", None)]:
            if dt_proj is None:
                continue
            nn.init.uniform_(dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

    def _scan(self, x, conv, x_proj, dt_proj, A_log, D):
        seqlen = x.shape[-1]
        x = self.act(conv(x)[..., :seqlen])
        x_dbl = x_proj(x.transpose(1, 2).reshape(-1, self.d_inner))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = dt.view(self.d_inner, x.shape[0], seqlen).permute(1, 0, 2).contiguous()
        B = B.view(x.shape[0], seqlen, self.d_state).transpose(1, 2).contiguous()
        C = C.view(x.shape[0], seqlen, self.d_state).transpose(1, 2).contiguous()
        A = -torch.exp(A_log.float())
        return selective_scan_ref(x, dt, A, B, C, D.float(), delta_bias=dt_proj.bias.float(), delta_softplus=True)

    def forward(self, hidden_states, inference_params=None):
        del inference_params
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        out = self._scan(x, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        if self.bimamba_type == "v2":
            out_b = self._scan(
                x.flip(-1),
                self.conv1d_b,
                self.x_proj_b,
                self.dt_proj_b,
                self.A_b_log,
                self.D_b,
            ).flip(-1)
            out = (out + out_b) / 2 if self.if_divide_out else out + out_b
        out = out * F.silu(z)
        return self.out_proj(out.transpose(1, 2))


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mixer = mixer_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, hidden_states, residual=None, inference_params=None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.mixer(self.norm(residual), inference_params=inference_params)
        return hidden_states, residual


class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=12,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        final_pool_type="mean",
        if_cls_token=True,
        use_middle_cls_token=True,
        bimamba_type="v2",
        if_divide_out=True,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.final_pool_type = final_pool_type
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.patch_embed = PatchEmbed(img_size, patch_size, stride, channels, embed_dim)
        num_tokens = 1 if if_cls_token else 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if if_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        mixer_cls = partial(MambaMixer, d_state=d_state, bimamba_type=bimamba_type, if_divide_out=if_divide_out)
        self.layers = nn.ModuleList([Block(embed_dim, mixer_cls, drop_path=dpr[i]) for i in range(depth)])
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch, num_patches, _ = x.shape
        token_position = None
        if self.if_cls_token:
            cls_token = self.cls_token.expand(batch, -1, -1)
            token_position = num_patches // 2 if self.use_middle_cls_token else 0
            x = torch.cat((x[:, :token_position], cls_token, x[:, token_position:]), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = residual + self.drop_path(hidden_states) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual)

        if self.if_cls_token:
            return hidden_states[:, token_position]
        if self.final_pool_type == "mean":
            return hidden_states.mean(dim=1)
        return hidden_states[:, -1]

    def forward(self, x, return_features=False):
        features = self.forward_features(x)
        return features if return_features else self.head(features)


def vim_tiny_patch16_224(num_classes=0, img_size=224, pretrained=False, **kwargs):
    del pretrained
    return VisionMamba(
        img_size=img_size,
        patch_size=16,
        stride=16,
        embed_dim=192,
        depth=12,
        num_classes=num_classes,
        **kwargs,
    )


def vim_small_patch16_224(num_classes=0, img_size=224, pretrained=False, **kwargs):
    del pretrained
    return VisionMamba(
        img_size=img_size,
        patch_size=16,
        stride=16,
        embed_dim=384,
        depth=12,
        num_classes=num_classes,
        **kwargs,
    )


def create_vim_model(backbone: str = "vim_tiny_patch16_224", num_classes: int = 0, img_size: int = 224, **kwargs):
    factories = {
        "vim_tiny_patch16_224": vim_tiny_patch16_224,
        "vim_small_patch16_224": vim_small_patch16_224,
    }
    if backbone not in factories:
        raise ValueError(f"Unknown local Vim backbone '{backbone}'. Available: {', '.join(factories)}")
    return factories[backbone](num_classes=num_classes, img_size=img_size, **kwargs)
