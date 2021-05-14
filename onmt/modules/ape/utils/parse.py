import configargparse as cfargparse
import os

import torch

import onmt.opts as opts
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
# for type hints and annotation


class ArgumentParserForAPE(ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super().__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        if model_opt.word_vec_size > 0:
            model_opt.src_word_vec_size = model_opt.word_vec_size
            model_opt.mt_word_vec_size = model_opt.word_vec_size
            model_opt.pe_word_vec_size = model_opt.word_vec_size

        if model_opt.layers > 0:
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.rnn_size > 0:
            model_opt.src_model_dim = model_opt.rnn_size
            model_opt.mt_model_dim = model_opt.rnn_size
            model_opt.pe_model_dim = model_opt.rnn_size
        else:
            assert model_opt.src_model_dim == model_opt.mt_model_dim \
                   and model_opt.mt_model_dim == model_opt.pe_model_dim, \
                   "You must give the same value to all the three \
                    model dimensons unless you set one common value \
                    on '--rnn_size'."
            model_opt.rnn_size = model_opt.src_model_dim

        if model_opt.copy_attn_type is None:
            model_opt.copy_attn_type = model_opt.global_attention

        if model_opt.alignment_layer is None:
            model_opt.alignment_layer = -2
            model_opt.lambda_align = 0.0
            model_opt.full_context_alignment = False

    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.model_type in ["text"], \
            "Unsupported model type %s" % model_opt.model_type

        if model_opt.share_embeddings:
            if model_opt.model_type != "text":
                raise AssertionError(
                    "--share_embeddings requires --model_type text.")
        if model_opt.lambda_align > 0.0:
            assert model_opt.ape_decoder_type in {"lee_shin_wmt19"}, \
                "Only transformer is supported to joint learn alignment."
            assert model_opt.alignment_layer < model_opt.dec_layers and \
                model_opt.alignment_layer >= -model_opt.dec_layers, \
                "N° alignment_layer should be smaller than number of layers."
            logger.info("Joint learn alignment at layer [{}] "
                        "with {} heads in full_context '{}'.".format(
                            model_opt.alignment_layer,
                            model_opt.alignment_heads,
                            model_opt.full_context_alignment))

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if opt.epochs:
            raise AssertionError(
                  "-epochs is deprecated please use -train_steps.")
        if opt.truncated_decoder > 0 and max(opt.accum_count) > 1:
            raise AssertionError("BPTT is not compatible with -accum > 1")

        if opt.gpuid:
            raise AssertionError(
                  "gpuid is deprecated see world_size and gpu_ranks")
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if opt.world_size < len(opt.gpu_ranks):
            raise AssertionError(
                  "parameter counts of -gpu_ranks must be less or equal "
                  "than -world_size.")
        if opt.world_size == len(opt.gpu_ranks) and \
                min(opt.gpu_ranks) > 0:
            raise AssertionError(
                  "-gpu_ranks should have master(=0) rank "
                  "unless -world_size is greater than len(gpu_ranks).")
        assert len(opt.data_ids) == len(opt.data_weights), \
            "Please check -data_ids and -data_weights options!"

        assert len(opt.dropout) == len(opt.dropout_steps), \
            "Number of dropout values must match accum_steps values"

        assert len(opt.attention_dropout) == len(opt.dropout_steps), \
            "Number of attention_dropout values must match accum_steps values"

    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

    @classmethod
    def validate_preprocess_args(cls, opt):
        assert opt.max_shard_size == 0, \
            "-max_shard_size is deprecated. Please use \
            -shard_size (number of examples) instead."
        assert opt.shuffle == 0, \
            "-shuffle is not implemented. Please shuffle \
            your data before pre-processing."

        # checking on training data files
        assert len(opt.train_src) == len(opt.train_pe) \
               and len(opt.train_src) == len(opt.train_mt), \
            "Please provide the same number of training data files \
            for src, mt, and pe."

        assert len(opt.train_src) == len(opt.train_ids), \
            "Please provide proper '--train_ids' to identify \
            each data file."

        for fl in opt.train_src + opt.train_mt + opt.train_pe:
            assert os.path.isfile(fl), \
                f"Please check if {fl} is the right path."

        if len(opt.train_align) == 1 and opt.train_align[0] is None:
            opt.train_align = [None] * len(opt.train_src)
        else:
            assert len(opt.train_align) == len(opt.train_src), \
                "Please provide same number of word alignment train \
                files as src/pe!"
            for file in opt.train_align:
                assert os.path.isfile(file), "Please check path of %s" % file

        # checking on validation data files
        assert not opt.valid_align or os.path.isfile(opt.valid_align), \
            "Please check if '--valid_align' has the right path."

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check if '--valid_src' has the right path."
        assert not opt.valid_mt or os.path.isfile(opt.valid_mt), \
            "Please check if '--valid_mt' has the right path."
        assert not opt.valid_pe or os.path.isfile(opt.valid_pe), \
            "Please check if '--valid_pe' has the right path."

        # checking on vocabulary files
        assert not opt.src_vocab or os.path.isfile(opt.src_vocab), \
            "Please check if '--src_vocab' has the right path."
        assert not opt.mt_vocab or os.path.isfile(opt.mt_vocab), \
            "Please check if '--mt_vocab' has the right path."
        assert (not opt.pe_vocab) or os.path.isfile(opt.pe_vocab), \
            "Please check if '--pe_vocab' has the right path."
