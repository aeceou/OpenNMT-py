#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.modules.ape.inputters.inputter import _build_fields_vocab, \
                                                build_vocab, \
                                                filter_example, \
                                                get_fields, \
                                                _load_vocab

from functools import partial
from multiprocessing import Pool
from onmt.modules.ape.inputters.dataset_base import DatasetForAPE


def check_existing_pt_files(opt, corpus_type, ids, existing_fields):
    """ Check if there are existing .pt files to avoid overwriting them """
    existing_shards = []
    for maybe_id in ids:
        if maybe_id:
            shard_base = corpus_type + "_" + maybe_id
        else:
            shard_base = corpus_type
        pattern = opt.save_data + '.{}.*.pt'.format(shard_base)
        if glob.glob(pattern):
            if opt.overwrite:
                maybe_overwrite = ("will be overwritten because "
                                   "`-overwrite` option is set.")
            else:
                maybe_overwrite = ("won't be overwritten, pass the "
                                   "`-overwrite` option if you want to.")
            logger.warning("Shards for corpus {} already exist, {}"
                           .format(shard_base, maybe_overwrite))
            existing_shards += [maybe_id]
    return existing_shards


def process_one_shard(corpus_params, params):
    corpus_type, fields, src_reader, mt_reader, pe_reader, \
        align_reader, opt, existing_fields, \
        src_vocab, mt_vocab, pe_vocab = corpus_params
    i, (src_shard, mt_shard, pe_shard,
        align_shard, maybe_id, filter_pred) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(mt_shard) \
           and len(src_shard) == len(pe_shard)
    logger.info("Building shard %d." % i)

    src_data = {"reader": src_reader, "data": src_shard, "dir": opt.src_dir}
    mt_data = {"reader": mt_reader, "data": mt_shard, "dir": None}
    pe_data = {"reader": pe_reader, "data": pe_shard, "dir": None}
    align_data = {"reader": align_reader, "data": align_shard, "dir": None}
    _readers, _data, _dir = DatasetForAPE.config(
        [('src', src_data), ("mt", mt_data), ("pe", pe_data),
         ("align", align_data)])

    dataset = DatasetForAPE(
        fields, readers=_readers, data=_data, dirs=_dir,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred,
        corpus_id=maybe_id
    )
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            sub_sub_counter['corpus_id'].update(
                ["train" if maybe_id is None else maybe_id])
            for name, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and
                                 src_vocab is not None) or \
                                (sub_n == "mt" and
                                 mt_vocab is not None) or \
                                (sub_n == "pe" and
                                 pe_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".\
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def maybe_load_vocab(corpus_type, counters, opt):
    src_vocab = None
    mt_vocab = None
    pe_vocab = None
    existing_fields = None
    if corpus_type == "train":
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        if opt.mt_vocab != "":
            mt_vocab, mt_vocab_size = _load_vocab(
                opt.mt_vocab, "mt", counters,
                opt.mt_words_min_frequency)
        if opt.pe_vocab != "":
            pe_vocab, pe_vocab_size = _load_vocab(
                opt.pe_vocab, "pe", counters,
                opt.pe_words_min_frequency)
    return src_vocab, mt_vocab, pe_vocab, existing_fields


def build_save_dataset(corpus_type, fields, src_reader, mt_reader,
                       pe_reader, align_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        mts = opt.train_mt
        pes = opt.train_pe
        ids = opt.train_ids
        aligns = opt.train_align
    elif corpus_type == 'valid':
        counters = None
        srcs = [opt.valid_src]
        mts = [opt.valid_mt]
        pes = [opt.valid_pe]
        ids = [None]
        aligns = [opt.valid_align]

    src_vocab, mt_vocab, pe_vocab, \
        existing_fields = maybe_load_vocab(
            corpus_type, counters, opt)

    existing_shards = check_existing_pt_files(
        opt, corpus_type, ids, existing_fields)

    # every corpus has shards, no new one
    if existing_shards == ids and not opt.overwrite:
        return

    def shard_iterator(srcs, mts, pes, ids, aligns, existing_shards,
                       existing_fields, corpus_type, opt):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, mt, pe, maybe_id, maybe_align \
            in zip(srcs, mts, pes, ids, aligns):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}"
                                   .format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None,\
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if ((corpus_type == "train" or opt.filter_valid)
                    and pes != None):
                filter_pred = partial(
                    filter_example,
                    use_src_len=opt.data_type == "text",
                    use_mt_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_mt_len=opt.mt_seq_length,
                    max_pe_len=opt.pe_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            mt_shards = split_corpus(mt, opt.shard_size)
            pe_shards = split_corpus(pe, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)
            for i, (ss, ms, ts, a_s) in enumerate(
                    zip(src_shards, mt_shards, pe_shards, align_shards)):
                yield (i, (ss, ms, ts, a_s, maybe_id, filter_pred))

    shard_iter = shard_iterator(srcs, mts, pes, ids, aligns,
                                existing_shards, existing_fields,
                                corpus_type, opt)

    with Pool(opt.num_threads) as p:
        dataset_params = (corpus_type, fields,
                          src_reader, mt_reader, pe_reader,
                          align_reader, opt, existing_fields,
                          src_vocab, mt_vocab, pe_vocab)
        func = partial(process_one_shard, dataset_params)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.mt_vocab_size, opt.mt_words_min_frequency,
                opt.pe_vocab_size, opt.pe_words_min_frequency,
                subword_prefix=opt.subword_prefix,
                subword_prefix_is_joiner=opt.subword_prefix_is_joiner)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.mt_vocab, opt.mt_vocab_size, opt.mt_words_min_frequency,
        opt.pe_vocab, opt.pe_vocab_size, opt.pe_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def preprocess(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)

    init_logger(opt.log_file)

    logger.info("Extracting features...")

    src_nfeats = 0
    mt_nfeats = 0
    pe_nfeats = 0
    for src, mt, pe in zip(opt.train_src, opt.train_mt, opt.train_pe):
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
        mt_nfeats += count_features(mt) if opt.data_type == "text" \
            else 0
        pe_nfeats += count_features(pe)  # pe always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(f" * number of source features: {mt_nfeats}.")
    logger.info(f" * number of target features: {pe_nfeats}.")

    logger.info("Building `Fields` object...")
    fields = get_fields(
        opt.data_type,
        src_nfeats,
        mt_nfeats,
        pe_nfeats,
        dynamic_dict=opt.dynamic_dict,
        with_align=opt.train_align[0] is not None,
        src_truncate=opt.src_seq_length_trunc,
        mt_truncate=opt.mt_seq_length_trunc,
        pe_truncate=opt.pe_seq_length_trunc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    mt_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    pe_reader = inputters.str2reader["text"].from_opt(opt)
    align_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        "train", fields, src_reader, mt_reader, pe_reader,
        align_reader, opt)

    if opt.valid_src and opt.valid_mt and opt.valid_pe:
        logger.info("Building & saving validation data...")
        build_save_dataset(
            "valid", fields, src_reader, mt_reader, pe_reader,
            align_reader, opt)
