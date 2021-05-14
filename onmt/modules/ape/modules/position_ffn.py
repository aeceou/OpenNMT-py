"""Position feed-forward network from "Attention is All You Need"."""

import torch.nn as nn
from onmt.modules.position_ffn import PositionwiseFeedForward


class PositionwiseFeedForwardForAPE(PositionwiseFeedForward):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, ffn_activation="relu"):
        super().__init__(d_model, d_ff, dropout=dropout)
        if ffn_activation == "relu":
            self.relu = nn.ReLU()
        elif ffn_activation == "gelu":
            self.relu = None
            self.gelu = nn.GELU()
        else:
            raise ValueError

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        if self.relu != None:
            inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        else:
            inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x
