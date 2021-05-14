import torch
import torch.nn as nn

from onmt.utils.misc import aeq
# for type hints and annotation
from torch import Tensor
from torchtext.vocab import Vocab
from typing import Optional, List


def collapse_copy_scores(
    scores: Tensor,
    batch: dict,
    pe_vocab: Vocab,
    mt_vocabs: Optional[List[Vocab]] = None,
    batch_dim: int = 1,
    batch_offset: Optional[List[int]] = None) -> Tensor:
    """
    Given the scores corresponding to an expanded dictionary,
    which contains just tokens making up the mt sequences in a batch,
    add those scores (the scores of the copied tokens)
    to the pe vocabulary's scores of the same words.
    """
    offset = len(pe_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if mt_vocabs == None:
            mt_vocab = batch.mt_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset != None else b
            index = batch.indices.data[batch_id]
            mt_vocab = mt_vocabs[index]

        for i in range(1, len(mt_vocab)):
            mw = mt_vocab.itos[i]
            pi = pe_vocab.stoi[mw]
            if pi != 0:
                blank.append(offset + i)
                fill.append(pi)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores
