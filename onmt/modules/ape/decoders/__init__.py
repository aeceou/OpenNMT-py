"""Module defining decoders for APE."""
from onmt.modules.ape.decoders.lee_shin_wmt19 \
    import TransformerDecoderForAPE


str2dec = {"lee_shin_wmt19": TransformerDecoderForAPE}

__all__ = ["TransformerDecoderForAPE"]
