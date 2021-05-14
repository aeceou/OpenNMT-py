""" Implementation of all available APE options """
from __future__ import print_function

import configargparse
import onmt

from onmt.models.sru import CheckSRU
from onmt.modules.ape.modules.source_noise import MultiNoise
# for type hints and annotation
from onmt.utils.parse import ArgumentParser


def config_opts(parser):
    ...


def model_opts(parser: ArgumentParser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group("APE Model-Embeddings")
    group.add("--mt_word_vec_size", "-mt_word_vec_size",
              type=int, default=500,
              help="Word embedding size for mt.")
    group.add("--pe_word_vec_size", "-pe_word_vec_size",
              type=int, default=500,
              help="Word embedding size for pe.")

    # Encoder-Decoder Options
    group = parser.add_argument_group("APE Model- Encoder-Decoder")
    group.add("--ape_encoder_type", "-ape_encoder_type", type=str,
              default="lee_shin_wmt19",
              choices=["lee_shin_wmt19"])
    group.add("--ape_decoder_type", "-ape_decoder_type", type=str,
              default="lee_shin_wmt19",
              choices=["lee_shin_wmt19"])

    group.add("--src_model_dim", "-src_model_dim", type=int, default=500,
              help="Size of src hidden states. "
                   "Must be equal to mt_model_dim "
                   "and pe_model_dim.")
    group.add("--mt_model_dim", "-mt_model_dim", type=int, default=500,
              help="Size of mt hidden states. "
                   "Must be equal to src_model_dim "
                   "and pe_model_dim.")
    group.add("--pe_model_dim", "-pe_model_dim", type=int, default=500,
              help="Size of pe hidden states. "
                   "Must be equal to src_model_dim "
                   "and mt_model_dim.")

    # Attention options
    group = parser.add_argument_group("APE Model- Attention")
    group.add("--multi_src_attn", "-multi_src_attn",
                type=str, default=None, choices=["stack"])
    group.add("--ffn_activation", "-ffn_activation",
                type=str, default="relu", choices=["relu", "gelu"])


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Task Options
    group = parser.add_argument_group("APE Task")
    group.add("--ape", "-ape", action="store_true")
    
    # Data options
    group = parser.add_argument_group('Data')
    group.add("--train_mt", "-train_mt", nargs="+",
                help="Path(s) to the training mt data")
    group.add("--train_pe", "-train_pe", nargs="+",
                help="Path(s) to the training pe data")

    group.add("--valid_mt", "-valid_mt",
                help="Path to the validation mt data")
    group.add("--valid_pe", "-valid_pe",
                help="Path to the validation pe data")

    # Dictionary options, for text corpus

    group = parser.add_argument_group("APE Vocab")
    # if you want to pass an existing vocab.pt file,
    # i.e. existing_fields, that you have already,
    # pass it to -src_vocab alone as it already contains mt and pe vocabs.
    group.add("--mt_vocab", "-mt_vocab", default="",
                help="Path to an existing MT output vocabulary. Format: "
                     "one word per line.")
    group.add("--pe_vocab", "-pe_vocab", default="",
                help="Path to an existing post-edit vocabulary. Format: "
                     "one word per line.")
    group.add("--mt_vocab_size", "-mt_vocab_size", type=int, default=50000,
              help="Size of the MT output vocabulary")
    group.add("--pe_vocab_size", "-pe_vocab_size", type=int, default=50000,
              help="Size of the post-edit vocabulary")

    group.add("--mt_words_min_frequency",
              "-mt_words_min_frequency", type=int, default=0)
    group.add("--pe_words_min_frequency",
              "-pe_words_min_frequency", type=int, default=0)

    # Truncation options, for text corpus
    group = parser.add_argument_group("APE Pruning")
    group.add("--mt_seq_length", "-mt_seq_length", type=int, default=50,
                help="Maximum MT output sequence length to keep.")
    group.add("--mt_seq_length_trunc", "-mt_seq_length_trunc",
                type=int, default=None,
                help="Truncate Mt output sequence length.")
    group.add("--pe_seq_length", "-pe_seq_length", type=int, default=50,
                help="Maximum post-edit sequence length to keep.")
    group.add("--pe_seq_length_trunc", "-pe_seq_length_trunc",
                type=int, default=None,
                help="Truncate post-edit sequence length.")


def train_opts(parser):
    """ Training and saving options """
    # Task Options
    group = parser.add_argument_group("APE Task")
    group.add("--ape", "-ape", action="store_true")

    group = parser.add_argument_group("General")
    group.add("--only_top_rank", "-only_top_rank", action='store_true',
                help="Save just top-N models w.r.t the measure "
                    "(0.8 * ppl + ed)")
    group.add("--saving_cycle", "-saving_cycle",
                type=int, default=None,
                help="Save top-N models every M steps.")

    # Pretrained word vectors
    group.add("--pre_word_vecs_src", "-pre_word_vecs_src",
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings for src. "
                   "See README for specific formatting instructions.")
    group.add("--pre_word_vecs_mt", "-pre_word_vecs_mt",
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings for mt. "
                   "See README for specific formatting instructions.")
    group.add("--pre_word_vecs_pe", "-pre_word_vecs_pe",
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings for pe. "
                   "See README for specific formatting instructions.")
    # Fixed word vectors
    group.add("--fix_word_vecs_src", "-fix_word_vecs_src",
              action="store_true",
              help="Fix word embeddings for src.")
    group.add("--fix_word_vecs_mt", "-fix_word_vecs_mt",
              action="store_true",
              help="Fix word embeddings for mt.")
    group.add("--fix_word_vecs_pe", "-fix_word_vecs_pe",
              action="store_true",
              help="Fix word embeddings for pe.")

    # Optimization options
    group = parser.add_argument_group("APE Optimization- Type")
    group.add("--ape_src_noise", "-ape_src_noise", type=str, nargs='+',
              default=[],
              choices=MultiNoise.NOISES.keys())
    group.add("--mt_noise", "-mt_noise", type=str, nargs='+',
              default=[],
              choices=MultiNoise.NOISES.keys())
    group.add("--mt_noise_prob", "-mt_noise_prob", type=float, nargs="+",
              default=[],
              help="Probabilities of mt_noise functions")


def translate_opts(parser):
    """ Translation / inference options """
    # Task Options
    group = parser.add_argument_group("APE Task")
    group.add("--ape", "-ape", action="store_true")

    group = parser.add_argument_group("APE Data")
    group.add("--mt", "-mt",
              help="MT output sequence to decode (one line per "
                   "sequence)")
    group.add("--pe", "-pe",
              help="True post-edit sequence (optional)")
