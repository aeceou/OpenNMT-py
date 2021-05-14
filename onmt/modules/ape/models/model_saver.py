import os
import torch

from collections import deque
from onmt.utils.logging import logger

from copy import deepcopy
from onmt.models.model_saver import ModelSaverBase
# for type hints and annotation
from argparse import Namespace
from onmt.modules.ape.models.model import APEModel
from onmt.modules.ape.utils.statistics import StatisticsForAPE
from operator import itemgetter
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict, List, Optional, Union


def build_model_saver(
    model_opt: Namespace,
    opt: Namespace,
    model: Module,
    fields: dict,
    optim: Optimizer):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint,
                             opt.only_top_rank,
                             opt.saving_cycle)
    return model_saver


class ModelSaverBaseForAPE(ModelSaverBase):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(
        self,
        base_path: str,
        model: Union[APEModel],
        model_opt: Namespace,
        fields: dict,
        optim: Optimizer,
        keep_checkpoint: int = -1,
        only_top_rank: bool = False,
        saving_cycle: Optional[int] = None):
        super().__init__(base_path, model, model_opt, fields, optim,
                         keep_checkpoint=keep_checkpoint)
        self.only_top_rank = only_top_rank
        self.saving_cycle = saving_cycle

    def save(
        self,
        step: int,
        moving_average: Optional[List] = None,
        stat: Optional[StatisticsForAPE] = None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        if self.only_top_rank:
            chkpt_inf = stat.mixed_ppl()
            if len(self.checkpoint_queue) > 0:
                self.checkpoint_queue = deque(sorted(self.checkpoint_queue,
                                                     key=itemgetter(1),
                                                     reverse=True),
                                              maxlen=self.keep_checkpoint)
                is_top = chkpt_inf < self.checkpoint_queue[0][1]
            else:
                is_top = True
            if is_top:
                chkpt, chkpt_name = self._save(step, save_model, stat=stat)
                self.last_saved_step = step
        else:
            chkpt, chkpt_name = self._save(step, save_model)
            self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if not self.only_top_rank:
                if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                    todel = self.checkpoint_queue.popleft()
                    self._rm_checkpoint(todel)
                self.checkpoint_queue.append(chkpt_name)
            else:
                if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen \
                   and is_top:
                    todel = self.checkpoint_queue.popleft()[0]
                    self._rm_checkpoint(todel)
                self.checkpoint_queue.append((chkpt_name, chkpt_inf))

        if self.saving_cycle != None:
            if step % self.saving_cycle == 0:
                self.checkpoint_queue = deque([], maxlen=self.keep_checkpoint)

    def _save(self, step, stat):
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBaseForAPE):
    """Simple model saver to filesystem"""

    def _save(self, step, model, stat=None):
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = model.generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in ["src", "mt", "pe"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': vocab,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        if stat != None:
            if self.only_top_rank:
                saving_condition = "_only_top_rank_"
            else:
                saving_condition = ""
            checkpoint_path = f"{self.base_path}{saving_condition}step_{step}_" \
                              f"ppl_{stat.ppl():.4f}_xent_{stat.xent():.4f}_" \
                              f"bleu_{stat.norm_bleu():.4f}_" \
                              f"ed_{stat.norm_ed():.4f}.pt"
        else:
            checkpoint_path = f"{self.base_path}_step_{step}.pt"
        logger.info(f"Saving checkpoint {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)
