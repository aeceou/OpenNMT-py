"""
This includes: LossComputeBaseForAPE and the standard APELossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
from onmt.modules.ape.modules.copy_generator import collapse_copy_scores
from onmt.modules.ape.utils.statistics import StatisticsForAPE
from onmt.utils.loss import shards, LabelSmoothingLoss, LossComputeBase


def build_loss_compute(model, pe_field, opt, train=True):
    """
    Returns a LossComputeForAPE subclass which wraps around an nn.Module
    subclass (such as nn.NLLLoss) which defines the loss criterion. The
    LossComputeForAPE object allows this loss to be computed in shards
    and passes the relevant data to a StatisticsForAPE object which handles
    training/validation logging. Currently, the APELossCompute class
    handles all loss computation except for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = pe_field.vocab.stoi[pe_field.pad_token]
    unk_idx = pe_field.vocab.stoi[pe_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(pe_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(pe_field.vocab), ignore_index=padding_idx
        )
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the APELossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    if opt.copy_attn:
        compute = CopyGeneratorLossComputeForAPE(
            criterion, loss_gen, pe_field.vocab, opt.copy_loss_by_seqlength,
            lambda_coverage=opt.lambda_coverage)
    else:
        compute = APELossCompute(
            criterion, loss_gen, lambda_coverage=opt.lambda_coverage,
            lambda_align=opt.lambda_align)
    compute.to(device)

    return compute


class LossComputeBaseForAPE(LossComputeBase):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the post-edit vocabulary.
        pe_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the post-edit output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBaseForAPE, self).__init__(criterion,
                                                    generator)

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[pe_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[pe_len x batch x mt_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.pe.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = StatisticsForAPE()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, post_ed, batch):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            post_ed (:obj:`FloatTensor`): true post-edits

        Returns:
            :obj:`onmt.utils.StatisticsForAPE` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = post_ed.ne(self.padding_idx)
        num_correct = pred.eq(post_ed).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        trs_pred = pred.view(-1, batch.batch_size).transpose(0, 1).contiguous()
        trs_post_ed = post_ed.view(-1, batch.batch_size).transpose(0, 1).contiguous()
        bleu = self._compute_bleu(trs_pred, trs_post_ed)
        ed = self._compute_ed(trs_pred, trs_post_ed)
        stat = StatisticsForAPE(loss.item(), num_non_padding, num_correct,
                                bleu=bleu, ed=ed, n_sent=batch.batch_size)
        return stat

    def _compute_bleu(self, pred, post_ed):
        hyp_lst = [ape.masked_select(pe.ne(self.padding_idx)).tolist()
                   for ape, pe in zip(pred, post_ed)]
        ref_lst = [[pe.masked_select(pe.ne(self.padding_idx)).tolist()
                   for pe in post_ed]]
        sent_bleu = [sentence_bleu(ref, hyp,
                                   smoothing_function=SmoothingFunction().method1)
                     for hyp, ref in zip(hyp_lst, ref_lst)]
        return sum(sent_bleu)

    def _compute_ed(self, pred, post_ed):
        d = []
        pe_len = []
        for ape, pe in zip(pred, post_ed):
            non_padding = pe.ne(self.padding_idx)
            ape = ape.masked_select(non_padding).tolist()
            pe = pe.masked_select(non_padding).tolist()
            ed = edit_distance(ape, pe)
            d.append(ed)
            pe_len.append(len(pe))
        d = np.array(d)
        pe_len = np.array(pe_len)
        return sum(np.divide(d, pe_len))


class APELossCompute(LossComputeBaseForAPE):
    """
    Standard APE Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0):
        super(APELossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "post_ed": batch.pe[range_[0] + 1 : range_[1], :, 0],
        }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        if self.lambda_align != 0.0:
            # attn_align should be in (batch_size, pad_pe_size, pad_src_size)
            attn_align = attns.get("align", None)
            # align_idx should be a Tensor in size([N, 3]), N is total number
            # of align src-pe pair in current batch, each as
            # ['sent_N°_in_batch', 'pe_id+1', 'src_id'] (check AlignField)
            align_idx = batch.align
            assert attns is not None
            assert attn_align != None, "lambda_align != 0.0 requires " \
                "alignement attention head"
            assert align_idx != None, "lambda_align != 0.0 requires " \
                "provide guided alignement"
            pad_pe_size, batch_size, _ = batch.pe.size()
            pad_src_size = batch.src[0].size(0)
            align_matrix_size = [batch_size, pad_pe_size, pad_src_size]
            ref_align = onmt.utils.make_batch_align_matrix(
                align_idx, align_matrix_size, normalize=True)
            # NOTE: pe-src ref alignement that in range_ of shard
            # (coherent with batch.pe)
            shard_state.update({
                "align_head": attn_align,
                "ref_align": ref_align[:, range_[0] + 1: range_[1], :]
            })
        return shard_state

    def _compute_loss(self, batch, output, post_ed, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):

        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = post_ed.view(-1)

        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)
            loss += align_loss
        stats = self._stats(loss.clone(), scores, gtruth, batch)

        return loss, stats

class CopyGeneratorLossComputeForAPE(APELossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, pe_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossComputeForAPE, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.pe_vocab = pe_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super()._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        })
        return shard_state

    def _compute_loss(self, batch, output, post_ed, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            post_ed: the validate post-edit to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        post_ed = post_ed.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.mt_map
        )
        loss = self.criterion(scores, align, post_ed)

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.pe_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct post_ed copy token instead of <unk>
        # pe[i] = align[i] + len(pe_vocab)
        # for i such that pe[i] == 0 and align[i] != 0
        post_ed_data = post_ed.clone()
        unk = self.criterion.unk_index
        correct_mask = (post_ed_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.pe_vocab)
        post_ed_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, post_ed_data,
                            batch)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            pe_lens = batch.pe[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, pe_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
