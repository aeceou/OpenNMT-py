""" Onmt APE Model base class definition """
from onmt.models.model import NMTModel
# for type hints and annotation
from onmt.modules.ape.decoders import TransformerDecoderForAPE
from onmt.modules.ape.encoders import TransformerEncoderForAPE
from torch import Tensor
from typing import Dict, Union


class APEModel(NMTModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(
        self,
        encoder: Union[TransformerEncoderForAPE],
        decoder: Union[TransformerDecoderForAPE]):
        super().__init__(encoder, decoder)

    def forward(
        self,
        src: Tensor,
        mt: Tensor,
        pe: Tensor,
        lengths: Dict[str, Tensor],
        bptt: bool = False,
        with_align: bool = False):
        dec_in = pe[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder({"src": src,
                                                        "mt": mt},
                                                       lengths)

        if bptt is False:
            self.decoder.init_state({"src": src, "mt": mt}, memory_bank,
                                    enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns
