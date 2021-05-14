import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoderLayer, \
                                      TransformerEncoder
from onmt.modules.ape.modules.multi_headed_attn import MultiHeadedAttentionForAPE
from onmt.modules.ape.modules.position_ffn import PositionwiseFeedForwardForAPE
from onmt.utils.misc import sequence_mask
# for type hints and annotation
from argparse import Namespace
from onmt.modules.embeddings import Embeddings
from torch import Tensor
from typing import Dict, Optional, Tuple


class TransformerEncoderLayerForSRCOfAPE(TransformerEncoderLayer):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, ffn_activation="relu"):
        super().__init__(d_model, heads, d_ff, dropout, attention_dropout,
                         max_relative_positions=max_relative_positions)

        self.feed_forward = PositionwiseFeedForwardForAPE(
            d_model, d_ff, dropout, ffn_activation=ffn_activation)


class TransformerEncoderLayerForMTOfAPE(TransformerEncoderLayer):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, ffn_activation="relu"):
        super().__init__(d_model, heads, d_ff, dropout, attention_dropout,
                         max_relative_positions=max_relative_positions)

        self.attention_to_src = MultiHeadedAttentionForAPE(
            heads, d_model, dropout=attention_dropout
        )
        self.feed_forward = PositionwiseFeedForwardForAPE(
            d_model, d_ff, dropout, ffn_activation=ffn_activation
        )
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(d_model, eps=1e-6),
             nn.LayerNorm(d_model, eps=1e-6)])

    def forward(
        self,
        inp: Dict[str, Tensor],
        mask: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        src, mt = inp["src_repr"], inp["mt"]
        mt_norm = self.layer_norm[0](mt)
        query, _ = self.self_attn(mt_norm, mt_norm, mt_norm,
                                  mask=mask["mt"], attn_type="self")
        query = self.dropout(query) + mt
        query_norm = self.layer_norm[1](query)
        mid, attns = self.attention_to_src(src, src, query_norm,
                                           mask=mask["src_repr"])
        output = self.feed_forward(self.dropout(mid) + query)
        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.attention_to_src.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoderForAPE(EncoderBase):
    """APE-Transformer encoder from
    "Transformer-based Automatic Post-Editing Model
    with Joint Encoder and Multi-source Attention of Decoder".
    :cite:`DBLP:conf/wmt/LeeSL19`

    .. mermaid::

       graph BT
          A[src]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          AA[mt]
          BB[multi-head self-attn]
          BBB[multi-head context-attn]
          CC[feed forward]
          OO[output]
          A --> B
          B --> C
          C --> O
          AA --> BB
          BB --> BBB
          C --> BBB
          BBB --> CC
          CC --> OO

    Args:
        num_layers (int): number of encoder layers
        src_d (int): size of the src encoder
        mt_d (int): size of the src encoder
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(
        self,
        num_layers: int,
        src_d: int,
        mt_d: int,
        heads: int,
        d_ff: int,
        dropout: float,
        attention_dropout: float,
        embeddings: Dict[str, Embeddings],
        max_relative_positions: int,
        ffn_activation: str):

        super().__init__()

        self.src_embeddings = embeddings["src"]
        self.mt_embeddings = embeddings["mt"]
        self.transformer = nn.ModuleDict(
            {"src_enc": nn.ModuleList(
                [TransformerEncoderLayerForSRCOfAPE(
                    src_d, heads, d_ff, dropout, attention_dropout,
                    max_relative_positions=max_relative_positions,
                    ffn_activation=ffn_activation)
                 for i in range(num_layers)]),
             "mt_enc": nn.ModuleList(
                    [TransformerEncoderLayerForMTOfAPE(
                        mt_d, heads, d_ff, dropout, attention_dropout,
                        max_relative_positions=max_relative_positions,
                        ffn_activation=ffn_activation)
                     for i in range(num_layers)])})
        self.src_layer_norm = nn.LayerNorm(src_d, eps=1e-6)
        self.mt_layer_norm = nn.LayerNorm(mt_d, eps=1e-6)

    @classmethod
    def from_opt(cls, opt: Namespace, embeddings: Dict[str, Embeddings]):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.src_model_dim,
            opt.mt_model_dim,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.ffn_activation)

    def forward(
        self,
        inp: Dict[str, Tensor],
        lengths: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor],
               Dict[str, Tensor],
               Dict[str, Tensor]]:
        """See :func:`EncoderBase.forward()`"""
        self._check_args(inp["src"], lengths["src"])
        self._check_args(inp["mt"], lengths["mt"])

        emb = {"src": self.src_embeddings(inp["src"]),
               "mt": self.mt_embeddings(inp["mt"])}

        src_out = self.src_embeddings(inp["src"]).transpose(0, 1).contiguous()
        src_mask = ~sequence_mask(lengths["src"]).unsqueeze(1)
        for layer in self.transformer["src_enc"]:
            src_out = layer(src_out, src_mask)
        src_out = self.src_layer_norm(src_out)

        mt_out = self.mt_embeddings(inp["mt"]).transpose(0, 1).contiguous()
        mt_mask = ~sequence_mask(lengths["mt"]).unsqueeze(1)
        out = {"src_repr": src_out, "mt": mt_out}
        mask = {"src_repr": src_mask, "mt": mt_mask}
        for layer in self.transformer["mt_enc"]:
            mt_out, attns = layer(out, mask)
        mt_out = self.mt_layer_norm(mt_out)

        out = {"src": src_out.transpose(0, 1).contiguous(),
               "mt": mt_out.transpose(0, 1).contiguous()}

        return emb, out, lengths

    def update_dropout(self, dropout, attention_dropout):
        self.src_embeddings.update_dropout(dropout)
        self.mt_embeddings.update_dropout(dropout)
        for encoder in self.transformer:
            for layer in encoder:
                layer.update_dropout(dropout, attention_dropout)
