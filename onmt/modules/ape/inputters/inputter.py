# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math

from collections import Counter, defaultdict
from itertools import chain, cycle

import torch
import torchtext.data
from torchtext.data import Field, RawField, LabelField
from torchtext.vocab import Vocab
from torchtext.data.utils import RandomShuffler

from onmt.inputters.text_dataset import text_fields, TextMultiField
from onmt.inputters.image_dataset import image_fields
from onmt.inputters.audio_dataset import audio_fields
from onmt.inputters.vec_dataset import vec_fields
from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401
from onmt.inputters.image_dataset import (  # noqa: F401
    batch_img as make_img)

import gc
from onmt.inputters.inputter import _build_fv_from_multifield, \
                                    _load_vocab, \
                                    make_src, \
                                    make_tgt, \
                                    _old_style_vocab, \
                                    _old_style_nesting, \
                                    _pad_vocab_to_multiple, \
                                    AlignField, \
                                    DatasetLazyIter, \
                                    MultipleDatasetIterator


class AlignFieldForAPE(LabelField):
    """
    Parse ['<src>-<pe>', ...] into ['<src>','<pe>', ...]
    """

    def __init__(self, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['preprocessing'] = parse_align_idx
        super().__init__(**kwargs)

    def process(self, batch, device=None):
        """ Turn a batch of align-idx to a sparse align idx Tensor"""
        sparse_idx = []
        for i, example in enumerate(batch):
            for src, mt, pe in example:
                # +1 for pe side to keep coherent after "bos" padding,
                # register ['N°_in_batch', 'pe_id+1', 'src_id']
                sparse_idx.append([i, pe + 1, src])

        align_idx = torch.tensor(sparse_idx, dtype=self.dtype, device=device)

        return align_idx


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <pe>], ...]
    """
    align_list = align_pharaoh.strip().split(' ')
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, pe_idx = align.split('-')
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(pe_idx)])
    return flatten_align_idx


def get_fields(
    inp_data_type,
    n_src_feats,
    n_mt_feats,
    n_pe_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    dynamic_dict=False,
    with_align=False,
    src_truncate=None,
    mt_truncate=None,
    pe_truncate=None
):
    """
    Args:
        inp_data_type: type of src and mt. Options are [text].
        n_src_feats (int): the number of source features (not counting tokens)
            to create a :class:`torchtext.data.Field` for. (If
            ``src_data_type=="text"``, these fields are stored together
            as a ``TextMultiField``).
        n_pe_feats (int): See above.
        pad (str): Special pad symbol. Used on src and tgt side.
        bos (str): Special beginning of sequence symbol. Only relevant
            for tgt.
        eos (str): Special end of sequence symbol. Only relevant
            for tgt.
        dynamic_dict (bool): Whether or not to include source map and
            alignment fields.
        with_align (bool): Whether or not to include word align.
        src_truncate: Cut off src sequences beyond this (passed to
            ``src_data_type``'s data reader - see there for more details).
        tgt_truncate: Cut off tgt sequences beyond this (passed to
            :class:`TextDataReader` - see there for more details).

    Returns:
        A dict mapping names to fields. These names need to match
        the dataset example attributes.
    """

    assert inp_data_type in ["text"], \
        "Data type not implemented"
    assert not dynamic_dict or inp_data_type == "text", \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {}

    fields_getters = {"text": text_fields}

    src_field_kwargs = {"n_feats": n_src_feats,
                        "include_lengths": True,
                        "pad": pad, "bos": None, "eos": None,
                        "truncate": src_truncate,
                        "base_name": "src"}
    fields["src"] = fields_getters[inp_data_type](**src_field_kwargs)

    mt_field_kwargs = {"n_feats": n_mt_feats,
                       "include_lengths": True,
                       "pad": pad, "bos": None, "eos": None,
                       "truncate": mt_truncate,
                       "base_name": "mt"}
    fields["mt"] = fields_getters[inp_data_type](**mt_field_kwargs)

    pe_field_kwargs = {"n_feats": n_pe_feats,
                       "include_lengths": False,
                       "pad": pad, "bos": bos, "eos": eos,
                       "truncate": pe_truncate,
                       "base_name": "pe"}
    fields["pe"] = fields_getters["text"](**pe_field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    corpus_ids = Field(use_vocab=True, sequential=False)
    fields["corpus_id"] = corpus_ids

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["mt_map"] = src_map

        mt_ex_vocab = RawField()
        fields["mt_ex_vocab"] = mt_ex_vocab

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = align

    if with_align:
        word_align = AlignField()
        fields["align"] = word_align

    return fields


def load_old_vocab(vocab, data_type="text", dynamic_dict=False):
    """Update a legacy vocab/field format.

    Args:
        vocab: a list of (field name, torchtext.vocab.Vocab) pairs. This is the
            format formerly saved in *.vocab.pt files. Or, text data
            not using a :class:`TextMultiField`.
        data_type (str): text, img, or audio
        dynamic_dict (bool): Used for copy attention.

    Returns:
        a dictionary whose keys are the field names and whose values Fields.
    """

    if _old_style_vocab(vocab):
        # List[Tuple[str, Vocab]] -> List[Tuple[str, Field]]
        # -> dict[str, Field]
        vocab = dict(vocab)
        n_src_features = sum('src_feat_' in k for k in vocab)
        n_mt_features = sum("mt_feat_" in k for k in vocab)
        n_pe_features = sum("pe_feat_" in k for k in vocab)
        fields = get_fields(
            data_type, n_src_features, n_mt_features,
            n_pe_features, dynamic_dict=dynamic_dict)
        for n, f in fields.items():
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(n, f)]
            for sub_n, sub_f in f_iter:
                if sub_n in vocab:
                    sub_f.vocab = vocab[sub_n]
        return fields

    if _old_style_field_list(vocab):  # upgrade to multifield
        # Dict[str, List[Tuple[str, Field]]]
        # doesn't change structure - don't return early.
        fields = vocab
        for base_name, vals in fields.items():
            if ((base_name == 'src' and data_type == 'text') or
                (base_name == "mt" and data_type == "text")
                or base_name == "pe"):
                assert not isinstance(vals[0][1], TextMultiField)
                fields[base_name] = [(base_name, TextMultiField(
                    vals[0][0], vals[0][1], vals[1:]))]

    if _old_style_nesting(vocab):
        # Dict[str, List[Tuple[str, Field]]] -> List[Tuple[str, Field]]
        # -> dict[str, Field]
        fields = dict(list(chain.from_iterable(vocab.values())))

    return fields


def _old_style_field_list(vocab):
    """Detect old-style text fields.

    Not old style vocab, old nesting, and text-type fields not using
    ``TextMultiField``.

    Args:
        vocab: some object loaded from a *.vocab.pt file

    Returns:
        Whether ``vocab`` is not an :func:`_old_style_vocab` and not
        a :class:`TextMultiField` (using an old-style text representation).
    """

    # if pe isn't using TextMultiField, then no text field is.
    return (not _old_style_vocab(vocab)) and _old_style_nesting(vocab) and \
        (not isinstance(vocab["pe"][0][1], TextMultiField))


def old_style_vocab(vocab):
    """The vocab/fields need updated."""
    return _old_style_vocab(vocab) or _old_style_field_list(vocab) or \
        _old_style_nesting(vocab)


def filter_example(ex, use_src_len=True, use_mt_len=True,
                   use_pe_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_mt_len=1, max_mt_len=float("inf"),
                   min_pe_len=1, max_pe_len=float("inf")):
    """Return whether an example is an acceptable length.

    If used with a dataset as ``filter_pred``, use :func:`partial()`
    for all keyword arguments.

    Args:
        ex (torchtext.data.Example): An object with a ``src``,
            ``mt``, and ``pe`` property.
        use_src_len (bool): Filter based on the length of ``ex.src``.
        use_pe_len (bool): Similar to above.
        min_src_len (int): A non-negative minimally acceptable length
            (examples of exactly this length will be included).
        min_pe_len (int): Similar to above.
        max_src_len (int or float): A non-negative (possibly infinite)
            maximally acceptable length (examples of exactly this length
            will be included).
        max_pe_len (int or float): Similar to above.
    """

    src_len = len(ex.src[0])
    mt_len = len(ex.mt[0])
    pe_len = len(ex.pe[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
           (not use_mt_len or min_mt_len <= mt_len <= max_mt_len) \
           and (not use_pe_len or min_pe_len <= pe_len <= max_pe_len)


def _build_fields_vocab(fields, counters, data_type, share_vocab,
                        vocab_size_multiple,
                        src_vocab_size, src_words_min_frequency,
                        mt_vocab_size, mt_words_min_frequency,
                        pe_vocab_size, pe_words_min_frequency,
                        subword_prefix="▁",
                        subword_prefix_is_joiner=False):
    build_fv_args = defaultdict(dict)
    build_fv_args["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency)
    build_fv_args["mt"] = dict(
        max_size=mt_vocab_size, min_freq=mt_words_min_frequency)
    build_fv_args["pe"] = dict(
        max_size=pe_vocab_size, min_freq=pe_words_min_frequency)
    pe_multifield = fields["pe"]
    _build_fv_from_multifield(
        pe_multifield,
        counters,
        build_fv_args,
        size_multiple=vocab_size_multiple if not share_vocab else 1)

    if fields.get("corpus_id", False):
        fields["corpus_id"].vocab = fields["corpus_id"].vocab_cls(
            counters["corpus_id"])

    if data_type == 'text':
        src_multifield = fields["src"]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1)
        mt_multifield = fields["mt"]
        _build_fv_from_multifield(
            mt_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1)

        if share_vocab:
            # `mt_vocab_size` and `pe_vocab_size` is ignored
            # when sharing vocabularies
            logger.info(" * merging src, mt, and pe vocab...")
            src_field = src_multifield.base_field
            mt_field = mt_multifield.base_field
            pe_field = pe_multifield.base_field
            _merge_field_vocabs(
                src_field, mt_field, pe_field,
                vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency,
                vocab_size_multiple=vocab_size_multiple)
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

        build_noise_field(
            src_multifield.base_field,
            mt_multifield.base_field,
            subword_prefix=subword_prefix,
            is_joiner=subword_prefix_is_joiner)
    return fields


def build_noise_field(src_field, mt_field, subword=True,
                      subword_prefix="▁", is_joiner=False,
                      sentence_breaks=[".", "?", "!"]):
    """In place add noise related fields i.e.:
         - word_start
         - end_of_sentence
    """
    if subword:
        def is_word_start(x): return (x.startswith(subword_prefix) ^ is_joiner)
        sentence_breaks = [subword_prefix + t for t in sentence_breaks]
    else:
        def is_word_start(x): return True

    src_vocab_size = len(src_field.vocab)
    word_start_mask = torch.zeros([src_vocab_size]).bool()
    end_of_sentence_mask = torch.zeros([src_vocab_size]).bool()
    for i, t in enumerate(src_field.vocab.itos):
        if is_word_start(t):
            word_start_mask[i] = True
        if t in sentence_breaks:
            end_of_sentence_mask[i] = True
    src_field.word_start_mask = word_start_mask
    src_field.end_of_sentence_mask = end_of_sentence_mask

    mt_vocab_size = len(mt_field.vocab)
    word_start_mask = torch.zeros([mt_vocab_size]).bool()
    end_of_sentence_mask = torch.zeros([mt_vocab_size]).bool()
    for i, t in enumerate(mt_field.vocab.itos):
        if is_word_start(t):
            word_start_mask[i] = True
        if t in sentence_breaks:
            end_of_sentence_mask[i] = True
    mt_field.word_start_mask = word_start_mask
    mt_field.end_of_sentence_mask = end_of_sentence_mask


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                mt_vocab_path, mt_vocab_size, mt_words_min_frequency,
                pe_vocab_path, pe_vocab_size, pe_words_min_frequency,
                vocab_size_multiple=1):
    """Build the fields for all data sides.

    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict[str, Field]): fields to build vocab for.
        data_type (str): A supported data type string.
        share_vocab (bool): share src, mt, and pe vocabulary?
        src_vocab_path (str): Path to src vocabulary file.
        src_vocab_size (int): size of the src vocabulary.
        src_words_min_frequency (int): the minimum frequency needed to
            include a src word in the vocabulary.
        pe_vocab_path (str): Path to pe vocabulary file.
        pe_vocab_size (int): size of the pe vocabulary.
        pe_words_min_frequency (int): the minimum frequency needed to
            include a pe word in the vocabulary.
        vocab_size_multiple (int): ensure that the vocabulary size is a
            multiple of this value.

    Returns:
        Dict of Fields
    """

    counters = defaultdict(Counter)

    if src_vocab_path:
        try:
            logger.info("Using existing vocabulary...")
            vocab = torch.load(src_vocab_path)
            # return vocab to dump with standard name
            return vocab
        except torch.serialization.pickle.UnpicklingError:
            logger.info("Building vocab from text file...")
            # empty train_dataset_files so that vocab is only loaded from
            # given paths in src_vocab_path, mt_vocab_path,
            # and pe_vocab_path
            train_dataset_files = []

    # Load vocabulary
    if src_vocab_path:
        src_vocab, src_vocab_size = _load_vocab(
            src_vocab_path, "src", counters,
            src_words_min_frequency)
    else:
        src_vocab = None

    if mt_vocab_path:
        mt_vocab, mt_vocab_size = _load_vocab(
            mt_vocab_path, "mt", counters,
            mt_words_min_frequency)
    else:
        mt_vocab = None

    if pe_vocab_path:
        pe_vocab, pe_vocab_size = _load_vocab(
            pe_vocab_path, "pe", counters,
            pe_words_min_frequency)
    else:
        pe_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
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
                    has_vocab = (sub_n == 'src' and src_vocab) or \
                                (sub_n == "mt" and mt_vocab) \
                                or (sub_n == "pe" and pe_vocab)
                    if sub_f.sequential and not has_vocab:
                        val = fd
                        counters[sub_n].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    fields = _build_fields_vocab(
        fields, counters, data_type,
        share_vocab, vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        mt_vocab_size, mt_words_min_frequency,
        pe_vocab_size, pe_words_min_frequency)

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, mt_field, pe_field, vocab_size,
                        min_freq, vocab_size_multiple):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with all the src, mt, and pe data?
    specials = [pe_field.unk_token, pe_field.pad_token,
                pe_field.init_token, pe_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs,
         mt_field.vocab.freqs,
         pe_field.vocab.freqs],
        Counter())
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
    src_field.vocab = merged_vocab
    mt_field.vocab = merged_vocab
    pe_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(mt_field.vocab) \
           and len(src_field.vocab) == len(pe_field.vocab)


class MultipleDatasetIteratorForAPE(MultipleDatasetIterator):
    """
    This takes a list of iterable objects (DatasetLazyIter) and their
    respective weights, and yields a batch in the wanted proportions.
    """
    def __init__(self,
                 train_shards,
                 fields,
                 device,
                 opt):
        self.index = -1
        self.iterables = []
        for shard in train_shards:
            self.iterables.append(
                build_dataset_iter(shard, fields, opt, multi=True))
        self.init_iterators = True
        self.weights = opt.data_weights
        self.batch_size = opt.batch_size
        self.batch_size_fn = max_tok_len \
            if opt.batch_type == "tokens" else None
        self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
        self.device = device
        # Temporarily load one shard to retrieve sort_key for data_type
        temp_dataset = torch.load(self.iterables[0]._paths[0])
        self.sort_key = temp_dataset.sort_key
        self.random_shuffler = RandomShuffler()
        self.pool_factor = opt.pool_factor
        del temp_dataset


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src, mt, or pe tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src, mt, and pe length in the current batch
    global max_src_in_batch, max_mt_in_batch, max_pe_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_mt_in_batch = 0
        max_pe_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # mt: [<bos> w1 ... wN <eos>]
    max_mt_in_batch = max(max_mt_in_batch, len(new.mt[0]) + 2)
    # pe: [w1 ... wM <eos>]
    max_pe_in_batch = max(max_pe_in_batch, len(new.pe[0]) + 1)
    src_elements = count * max_src_in_batch
    mt_elements = count * max_mt_in_batch
    pe_elements = count * max_pe_in_batch
    return max(src_elements, mt_elements, pe_elements)


def build_dataset_iter(corpus_type, fields, opt, is_train=True, multi=False):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_glob = opt.data + '.' + corpus_type + '.[0-9]*.pt'
    dataset_paths = list(sorted(
        glob.glob(dataset_glob),
        key=lambda p: int(p.split(".")[-2])))

    if not dataset_paths:
        if is_train:
            raise ValueError('Training data %s not found' % dataset_glob)
        else:
            return None
    if multi:
        batch_size = 1
        batch_fn = None
        batch_size_multiple = 1
    else:
        batch_size = opt.batch_size if is_train else opt.valid_batch_size
        batch_fn = max_tok_len \
            if is_train and opt.batch_type == "tokens" else None
        batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1

    device = "cuda" if opt.gpu_ranks else "cpu"

    return DatasetLazyIter(
        dataset_paths,
        fields,
        batch_size,
        batch_fn,
        batch_size_multiple,
        device,
        is_train,
        opt.pool_factor,
        repeat=not opt.single_pass,
        num_batches_multiple=max(opt.accum_count) * opt.world_size,
        yield_raw_example=multi)


def build_dataset_iter_multiple(train_shards, fields, opt):
    return MultipleDatasetIteratorForAPE(
        train_shards, fields, "cuda" if opt.gpu_ranks else "cpu", opt)
