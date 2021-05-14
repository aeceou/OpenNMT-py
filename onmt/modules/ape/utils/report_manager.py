""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime

import onmt

from onmt.utils.logging import logger
from onmt.modules.ape.utils.statistics import StatisticsForAPE
from onmt.utils.report_manager import ReportMgrBase


def build_report_manager(opt, gpu_rank):
    if opt.tensorboard and gpu_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBaseForAPE(ReportMgrBase):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        super().__init__(report_every, start_time)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(StatisticsForAPE): old StatisticsForAPE instance.
        Returns:
            report_stats(StatisticsForAPE): updated StatisticsForAPE instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    StatisticsForAPE.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            return StatisticsForAPE()
        else:
            return report_stats


class ReportMgr(ReportMgrBaseForAPE):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   step)
        report_stats = StatisticsForAPE()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())

            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step)
