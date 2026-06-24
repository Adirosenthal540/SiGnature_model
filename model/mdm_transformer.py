import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask
import torch.nn.functional as F
from yaml.tokens import TagToken
from model.mdm_multiheadattention import multi_head_attention_forward

F.multi_head_attention_forward = multi_head_attention_forward
from typing import Dict, Optional, Any, Union, Callable
from torch import Tensor, LongTensor

# derived and partially replicated from torch/nn/modules/transformer.py (pytorch 2.3.1, pytorch-cuda 12.1)


class MDM_TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, **kwargs):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = TagToken
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for layer_i, mod in enumerate(self.layers):
            output, _ = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class MDM_TransformerDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super(MDM_TransformerDecoderLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype
        )
        self.nhead = nhead

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        mode: Optional[str] = None,
    ) -> Tensor:

        self.layer_get_feat = {}
        x = tgt  # hml: n_frames, n_samples, n_features
        if self.norm_first:
            raise NotImplementedError("[norm_first] is not supported at the moment")
        else:
            # self attn
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, mode))
            # cross attn
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, mode=mode))
            # feed forward
            x = self.norm3(x + self._ff_block(x, mode))

        return x, self.layer_get_feat

    def _self_attn_wrap(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        q, k, v = x, x, x

        # average_attn_weights should be False because we sometimes want to extract attention weights
        x = self.self_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            average_attn_weights=False,
        )
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self._self_attn_wrap(x, attn_mask, key_padding_mask, is_causal)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, _ = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=is_get, average_attn_weights=False
        )
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout3(x)


class MDM_TransformerEncoder(nn.TransformerEncoder):
    """
    • Supports the same `mode ∈ {None, "get", "transfer"}` switch
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int, norm: Optional[nn.Module] = None, **kwargs):
        super().__init__(encoder_layer, num_layers, norm)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ):
        output = src
        for i, layer in enumerate(self.layers):
            output, _ = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output


# ------------------------------------------------------------
# Single encoder layer with feature‑capture hooks
# ------------------------------------------------------------
class MDM_TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    • Mirrors the sub‑blocks from your decoder layer
    • Keeps 'transfer' mode for leader/follower logic
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        self.nhead = nhead

    # ---------- PUBLIC FORWARD ----------
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ):
        self.layer_get_feat: Dict[str, Tensor] = {}
        x = src  # shape: (T, B, D)  or  (B, T, D) if batch_first

        if self.norm_first:
            raise NotImplementedError("[norm_first] is not supported")

        # ------ self‑attention (with optional transfer) ------
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))

        # ------ feed‑forward ------
        x = self.norm2(x + self._ff_block(x))

        return x, self.layer_get_feat

    # ---------- SUB‑BLOCKS ----------
    def _self_attn_wrap(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool):
        """Handles transfer‑mode trickery and regular SA call."""
        q = k = v = x

        out, attn_dict = self.self_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            average_attn_weights=False,
        )
        return out, attn_dict

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool):
        x, _ = self._self_attn_wrap(x, attn_mask, key_padding_mask, is_causal)

        return self.dropout1(x)

    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)


# ------------------------------------------------------------
# small util (same as torch internals) to infer seq‑len
# ------------------------------------------------------------
def _get_seq_len(t: Tensor, batch_first: bool) -> int:
    return t.shape[1] if batch_first else t.shape[0]
