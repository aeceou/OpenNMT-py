from __future__ import division
import time
import math
import sys

from onmt.utils.logging import logger
from onmt.utils.statistics import Statistics
# for type hints and annotation
from torch.utils.tensorboard import SummaryWriter


class StatisticsForAPE(Statistics):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0,
                 bleu=0, ed=0, n_sent=0):
        super().__init__(loss, n_words, n_correct)
        self.bleu = bleu
        self.ed = ed
        self.n_mt_words = 0
        self.n_sent = n_sent

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `StatisticsForAPE` object accross multiple process/nodes

        Args:
            stat(:obj:StatisticsForAPE): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `StatisticsForAPE`, the update stats object
        """
        stats = StatisticsForAPE.all_gather_stats_list(
            [stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `StatisticsForAPE` list accross all processes/nodes

        Args:
            stat_list(list([`StatisticsForAPE`])): list of statistics
                objects to gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`StatisticsForAPE`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_inp_words=True)
        return our_stats

    def update(self, stat, update_n_inp_words=False):
        """
        Update statistics by suming values with another `StatisticsForAPE`
        object

        Args:
            stat: another statistic object
            update_n_inp_words(bool): whether to update (sum) `n_src_words`
                and `n_mt_words` or not; i.e. to 'gather' up those values

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.bleu += stat.bleu
        self.ed += stat.ed
        self.n_sent += stat.n_sent

        if update_n_inp_words:
            self.n_src_words += stat.n_src_words
            self.n_mt_words += stat.n_mt_words

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_mt_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self,
                        prefix: str,
                        writer: SummaryWriter,
                        learning_rate: float,
                        step: int):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
        if prefix == "valid":
            writer.add_scalar(prefix + "/bleu", self.norm_bleu(), step)
            writer.add_scalar(prefix + "/ed", self.norm_ed(), step)

    def norm_bleu(self):
        return self.bleu / self.n_sent

    def norm_ed(self):
        return self.ed / self.n_sent

    def mixed_ppl(self):
        return 0.8 * self.ppl() + self.norm_ed()