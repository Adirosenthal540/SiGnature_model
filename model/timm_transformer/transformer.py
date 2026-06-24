import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from .config import use_fused_attn
from .helpers import to_2tuple
import os
import torch
import matplotlib.pyplot as plt

__all__ = ["VisionTransformer"]  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


def diagonal_band_mask(size: int, band: int = 3):
    """
    Returns a size×size mask with ones in the diagonal band
    |i-j| ≤ band  and zeros elsewhere.
    """
    idx = np.arange(size)
    return np.abs(idx[:, None] - idx) <= band  # bool


def cosine_matrix(A, B, eps=1e-8):
    """Cosine similarity between two matrices as single scalar."""
    a, b = A.ravel(), B.ravel()
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def save_attention_grid(attn: torch.Tensor, out_dir: str, fname: str = "attention_grid.png"):
    """
    attn: torch.Tensor of shape (B,H,T,T) (e.g., (1,4,196,196))
    out_dir: folder where to save the image
    """
    os.makedirs(out_dir, exist_ok=True)

    attn_h = attn[0].detach().cpu().float().numpy()  # (H, T, T)
    H = attn_h.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()

    # consistent scaling across heads
    vmin, vmax = attn_h.min(), attn_h.max()

    for h in range(H):
        im = axes[h].imshow(attn_h[h], vmin=vmin, vmax=vmax, cmap="viridis", origin="upper", aspect="equal")
        axes[h].set_title(f"Head {h}", fontsize=12, pad=6)
        axes[h].axis("off")

    # adjust layout: leave space for colorbar & title
    fig.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.05, wspace=0.1, hspace=0.2)

    # add a dedicated axis for the colorbar (so it doesn’t overlap maps)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention weight", fontsize=12)

    # subtitle with space above
    fig.suptitle("sa_out_atn", fontsize=16, y=0.98)

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved attention grid to {out_path}")


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn(experimental=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
        mode: Optional[str] = None,
    ):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        # ------ support "transfer" by swapping q ------

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
            attn = None
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def rolling_std_bct(x_bct: torch.Tensor, win: int = 30, eps: float = 1e-8):
    """
    x_bct: (B, C, T)
    returns: (B, C, T) rolling std over a centered window of length `win`
    """
    B, C, T = x_bct.shape
    win = min(win, T)
    left = win // 2
    right = win - 1 - left
    x_pad = F.pad(x_bct, (left, right), mode="reflect")

    w = torch.ones(C, 1, win, device=x_bct.device, dtype=x_bct.dtype)  # (C,1,win)
    sum_x = F.conv1d(x_pad, w, groups=C)  # (B,C,T)
    sum_x2 = F.conv1d(x_pad**2, w, groups=C)  # (B,C,T)

    mean = sum_x / win
    var = (sum_x2 / win) - mean.pow(2)
    std = (var.clamp_min(0) + eps).sqrt()
    return std  # (B,C,T)


def rolling_rms_speed_bct(x_bct: torch.Tensor, win: int = 30, eps: float = 1e-8):
    """
    x_bct: (B, C, T)
    returns: (B, C, T) rolling RMS of first temporal difference (speed proxy)
    """
    B, C, T = x_bct.shape
    # depthwise diff kernel: (C,1,2) with [1, -1] per channel
    k = torch.tensor([1.0, -1.0], device=x_bct.device, dtype=x_bct.dtype).view(1, 1, 2)
    k = k.repeat(C, 1, 1)  # (C,1,2)
    dx = F.conv1d(F.pad(x_bct, (1, 0), mode="replicate"), k, groups=C)  # (B,C,T)

    win = min(win, T)
    left = win // 2
    right = win - 1 - left
    w = torch.ones(C, 1, win, device=x_bct.device, dtype=x_bct.dtype)  # (C,1,win)
    mean_sq = F.conv1d(F.pad(dx * dx, (left, right), mode="reflect"), w, groups=C) / win
    rms = (mean_sq + eps).sqrt()
    return rms  # (B,C,T)


def window_std_mask_bct(x_bct: torch.Tensor, win: int = 49, thr: float = 0.04):
    """
    x_bct: (B, C, T). Returns:
      std_map_bct: (B, C, T)  std duplicated per 49-frame window
      mask_bct   : (B, C, T)  boolean mask (std > thr) duplicated to T
    """
    B, C, T = x_bct.shape
    assert T % win == 0, "T must be a multiple of win for simple duplication"

    # (B, C, nwin, win)
    x_win = x_bct.unfold(dimension=2, size=win, step=win)
    std_win = x_win.std(dim=-1, unbiased=False)  # (B, C, nwin)

    # duplicate each window’s std back over its 49 frames
    std_map = std_win.repeat_interleave(win, dim=-1)  # (B, C, nwin*win) == (B,C,T)
    mask = (std_win > thr).repeat_interleave(win, dim=-1)  # (B, C, T), bool
    return std_map, mask


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        sa_out = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(sa_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
