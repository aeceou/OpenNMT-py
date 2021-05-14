import torch
import torch.nn as nn

from onmt.decoders.transformer import TransformerDecoderLayer, \
                                      TransformerDecoder
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.ape.modules.multi_headed_attn import MultiHeadedAttentionForAPE
from onmt.modules.ape.modules.position_ffn import PositionwiseFeedForwardForAPE
from onmt.utils.misc import sequence_mask
# for type hints and annotation
from argparse import Namespace
from onmt.modules.embeddings import Embeddings
from torch import Tensor
from typing import Callable, Dict, Tuple, Optional


class TransformerDecoderLayerForPEOfAPE(TransformerDecoderLayer):

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        attention_dropout: float,
        self_attn_type: str = "scaled-dot",
        max_relative_positions: int = 0,
        aan_useffn: bool = False,
        full_context_alignment=False,
        alignment_heads=0,
        multi_src_attn: Optional[str] = "stack",
        ffn_activation: str = "relu"):
        super().__init__(
            d_model, heads, d_ff, dropout, attention_dropout,
            self_attn_type=self_attn_type,
            max_relative_positions=max_relative_positions,
            aan_useffn=aan_useffn, full_context_alignment=full_context_alignment,
            alignment_heads=alignment_heads)

        self.context_attn = nn.ModuleDict(
            {"mt": MultiHeadedAttentionForAPE(
                    heads, d_model, dropout=attention_dropout)})
        self.feed_forward = PositionwiseFeedForwardForAPE(
            d_model, d_ff, dropout, ffn_activation=ffn_activation)

        self.multi_src_attn = multi_src_attn
        if self.multi_src_attn != None:
            self.context_attn["src"] = MultiHeadedAttentionForAPE(
                heads, d_model, dropout=attention_dropout)
            self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

    def _forward(
        self,
        inputs: Tensor,
        memory_bank: Dict[str, Tensor],
        enc_pad_mask: Dict[str, Tensor],
        pe_pad_mask: Tensor,
        layer_cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
        step: int = None,
        future: bool = False) -> Tuple[Tensor, Tensor]:

        dec_mask = None

        if step is None:
            pe_len = pe_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [pe_len, pe_len],
                    device=pe_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, pe_len, pe_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(pe_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = pe_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        if self.multi_src_attn != None:
            if self.multi_src_attn == "stack":
                mid, attns = self.context_attn["mt"](memory_bank["mt"],
                                                     memory_bank["mt"],
                                                     query_norm,
                                                     mask=enc_pad_mask["mt"],
                                                     layer_cache=layer_cache,
                                                     attn_type="mt")
                mid = self.drop(mid) + query
                mid_norm = self.layer_norm_3(mid)
                out, _ = self.context_attn["src"](memory_bank["src"],
                                                  memory_bank["src"],
                                                  mid_norm,
                                                  mask=enc_pad_mask["src"],
                                                  layer_cache=layer_cache,
                                                  attn_type="src")
                output = self.feed_forward(self.drop(out) + mid)
            else:
                raise NotImplementedError(
                    f"Such multi-source attention ({self.multi_src_attn}) "
                    "has not been implemented yet"
                )
        else:
            mid, attns = self.context_attn["mt"](memory_bank["mt"],
                                                 memory_bank["mt"],
                                                 query_norm,
                                                 mask=enc_pad_mask["mt"],
                                                 layer_cache=layer_cache,
                                                 attn_type="mt")
            output = self.feed_forward(self.drop(mid) + query)

        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        if self.multi_src_attn != None:
            self.context_attn["src"].update_dropout(attention_dropout)
        self.context_attn["mt"].update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoderForAPE(TransformerDecoder):
    """APE-Transformer decoder from
    "Transformer-based Automatic Post-Editing Model
    with Joint Encoder and Multi-source Attention of Decoder".
    :cite:`DBLP:conf/wmt/LeeSL19`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head joint-attn]
          BBB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> BBB
          BBB --> C
          C --> O


    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        heads: int,
        d_ff: int,
        copy_attn: bool,
        self_attn_type: str,
        dropout: float,
        attention_dropout: float,
        embeddings: Embeddings,
        max_relative_positions: int,
        aan_useffn: bool,
        full_context_alignment: bool,
        alignment_layer: int,
        alignment_heads: int,
        multi_src_attn: str,
        ffn_activation: str):
        super().__init__(num_layers, d_model, heads, d_ff,
                         copy_attn, self_attn_type, dropout, attention_dropout,
                         embeddings, max_relative_positions, aan_useffn,
                         full_context_alignment, alignment_layer,
                         alignment_heads)

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayerForPEOfAPE(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             full_context_alignment=full_context_alignment,
             alignment_heads=alignment_heads,
             multi_src_attn=multi_src_attn,
             ffn_activation=ffn_activation)
             for i in range(num_layers)])
        self.multi_src_attn = multi_src_attn

    @classmethod
    def from_opt(cls, opt: Namespace, embeddings: Embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.pe_model_dim,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            multi_src_attn=opt.multi_src_attn,
            ffn_activation=opt.ffn_activation)

    def init_state(
        self,
        enc_inp: Dict[str, Tensor],
        memory_bank: Dict[str, Tensor],
        enc_hidden: Tensor):
        """Initialize decoder state."""
        self.state["src"] = enc_inp["src"]
        self.state["mt"] = enc_inp["mt"]
        self.state["cache"] = None

    def map_state(self, fn: Callable):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        self.state["mt"] = fn(self.state["mt"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()
        self.state["mt"] = self.state["mt"].detach()

    def forward(
        self,
        pe: Tensor,
        memory_bank: Dict[str, Tensor],
        step: int = None,
        **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        pe_words = pe[:, :, 0].transpose(0, 1)

        emb = self.embeddings(pe, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        trs_mem_bank = dict()
        for k, v in memory_bank.items():
            trs_mem_bank[k] = v.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        enc_inp_lens = kwargs["memory_lengths"]
        enc_max_len = self.state["mt"].shape[0]
        enc_pad_mask = dict()
        if self.multi_src_attn != None:
            enc_pad_mask["src"] = ~sequence_mask(enc_inp_lens["src"],
                                                 enc_max_len).unsqueeze(1)
        enc_pad_mask["mt"] = ~sequence_mask(enc_inp_lens["mt"],
                                            enc_max_len).unsqueeze(1)
        pe_pad_mask = pe_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(
                output,
                trs_mem_bank,
                enc_pad_mask,
                pe_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank: Dict[str, Tensor]):
        self.state["cache"] = {}
        batch_size = memory_bank["mt"].size(1)
        depth = memory_bank["mt"].size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": {"src": None, "mt": None},
                           "memory_values": {"src": None, "mt": None}}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank["mt"].device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
