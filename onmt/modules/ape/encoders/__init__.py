"""Module defining encoders for APE."""
from onmt.modules.ape.encoders.lee_shin_wmt19 \
    import TransformerEncoderForAPE


str2enc = {"lee_shin_wmt19": TransformerEncoderForAPE}

__all__ = ["TransformerEncoderForAPE", "str2enc"]
