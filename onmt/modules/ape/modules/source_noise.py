import math
import torch
from onmt.modules.source_noise import NoiseBase, MaskNoise, \
                                      SenShufflingNoise, InfillingNoise


class NoiseBaseForAPE(NoiseBase):
    def __init__(self, prob, pad_idx=1, device_id="cpu",
                 ids_to_noise=[], **kwargs):
        super().__init__(prob, pad_idx=pad_idx, device_id=device_id,
                         ids_to_noise=ids_to_noise, **kwargs)
    def noise_batch(self, batch):
        src, src_len = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, [None] * batch.src.size(1))
        # noise_skip = batch.noise_skip
        # aeq(len(batch.noise_skip) == src.size(1))
        mt, mt_len = batch.mt if isinstance(batch.mt, tuple) \
            else (batch.mt, [None] * batch.mt.size(1))

        # src is [src_len x bs x feats]
        src_skipped = src[:self.skip_first, :, :]
        src = src[self.skip_first:]
        # mt is [mt_len * bs * feats]
        mt_skipped = mt[:self.skip_first, :, :]
        mt = mt[self.skip_first:]
        
        for i in range(src.size(1)):
            if hasattr(batch, 'corpus_id'):
                corpus_id = batch.corpus_id[i]
                if corpus_id.item() not in self.ids_to_noise:
                    continue
            src_tkn = src[:, i, 0]
            src_mask = src_tkn.ne(self.pad_idx)
            mt_tkn = mt[:, i, 0]
            mt_mask = mt_tkn.ne(self.pad_idx)

            masked_src = src_tkn[src_mask]
            noisy_src, noisy_src_len = self.noise_source(
                masked_src, length=src_len[i], side="src")
            masked_mt = mt_tkn[mt_mask]
            noisy_mt, noisy_mt_len = self.noise_source(
                masked_mt, length=mt_len[i], side="mt")

            src_len[i] = noisy_src_len
            mt_len[i] = noisy_mt_len

            # src might increase length
            # so we need to resize the whole tensor
            src_delta = noisy_src_len - (src.size(0) - self.skip_first)
            if src_delta > 0:
                src_pad = torch.ones([src_delta],
                                     device=src.device,
                                     dtype=src.dtype)
                src_pad *= self.pad_idx
                src_pad = src_pad.unsqueeze(1).expand(-1, 15).unsqueeze(2)

                src = torch.cat([src, src_pad])
            src[:noisy_src.size(0), i, 0] = noisy_src

            # mt might also increase length
            # so we need to resize the whole tensor
            mt_delta = noisy_mt_len - (mt.size(0) - self.skip_first)
            if mt_delta > 0:
                mt_pad = torch.ones([mt_delta],
                                    device=mt.device,
                                    dtype=mt.dtype)
                mt_pad *= self.pad_idx
                mt_pad = mt_pad.unsqueeze(1).expand(-1, 15).unsqueeze(2)

                mt = torch.cat([mt, mt_pad])
            mt[:noisy_mt.size(0), i, 0] = noisy_mt

        src = torch.cat([src_skipped['src'], src])
        mt = torch.cat([mt_skipped, mt])

        # remove useless pad
        src_max_len, mt_max_len = src_len.max(), mt_len.max()
        src, mt = src[:src_max_len, :, :], mt[:mt_max_len, :, :]

        batch.src = src, src_len
        batch.mt = mt, mt_len
        return batch


class MaskNoiseForAPE(NoiseBaseForAPE):
    def noise_batch(self, batch):
        raise ValueError("MaskNoise has not been updated to tensor noise")
    # def s(self, tokens):
    #     prob = self.prob
    #     r = torch.rand([len(tokens)])
    #     mask = False
    #     masked = []
    #     for i, tok in enumerate(tokens):
    #         if tok.startswith(subword_prefix):
    #             if r[i].item() <= prob:
    #                 masked.append(mask_tok)
    #                 mask = True
    #             else:
    #                 masked.append(tok)
    #                 mask = False
    #         else:
    #             if mask:
    #                 pass
    #             else:
    #                 masked.append(tok)
    #     return masked


class SenShufflingNoiseForAPE(NoiseBaseForAPE):
    def __init__(self, *args, end_of_sentence_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert end_of_sentence_mask is not None
        self.end_of_sentence_mask = self.to_device(end_of_sentence_mask)

    def is_end_of_sentence(self, source):
        return self.end_of_sentence_mask.gather(0, source)

    def noise_source(self, source, length=None, **kwargs):
        # aeq(source.size(0), length)
        full_stops = self.is_end_of_sentence(source)
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-1] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * self.prob[kwargs["side"]])
                                   / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        index = 0
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i >
                               0 else 1):sentence_ends[i]]
            result[index:index + sentence.size(0)] = sentence
            index += sentence.size(0)
        # aeq(source.size(0), length)
        return result, length


class InfillingNoiseForAPE(NoiseBaseForAPE):
    def __init__(self, *args, infilling_poisson_lambda=3.0,
                 word_start_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.poisson_lambda = infilling_poisson_lambda
        self.mask_span_distribution = self._make_poisson(self.poisson_lambda)
        self.mask_idx = 0
        assert word_start_mask is not None
        self.word_start_mask = self.to_device(word_start_mask)

        # -1: keep everything (i.e. 1 mask per token)
        #  0: replace everything (i.e. no mask)
        #  1: 1 mask per span
        self.replace_length = 1

    def _make_poisson(self, poisson_lambda):
        # fairseq/data/denoising_dataset.py
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.tensor(ps, device=torch.device(self.device_id))
        return torch.distributions.Categorical(ps)

    def is_word_start(self, source):
        # print("src size: ", source.size())
        # print("ws size: ", self.word_start_mask.size())
        # print("max: ", source.max())
        # assert source.max() < self.word_start_mask.size(0)
        # assert source.min() >= 0
        return self.word_start_mask.gather(0, source)

    def noise_source(self, source, **kwargs):

        is_word_start = self.is_word_start(source)
        # assert source.size() == is_word_start.size()
        # aeq(source.eq(self.pad_idx).long().sum(), 0)

        # we manually add this hypothesis since it's required for the rest
        # of the function and kindof make sense
        is_word_start[-1] = 0

        p = self.prob[kwargs["side"]]
        num_to_mask = (is_word_start.float().sum() * p).ceil().long()
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(
                sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat([
                    lengths,
                    self.mask_span_distribution.sample(
                        sample_shape=(num_to_mask,))
                ], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(
                    source, num_inserts / source.size(0))
            # assert (lengths > 0).all()
        else:
            raise ValueError("Not supposed to be there")
            lengths = torch.ones((num_to_mask,), device=source.device).long()
        # assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[
            :num_to_mask]].squeeze(1)

        source_length = source.size(0)
        # TODO why?
        # assert source_length - 1 not in indices
        to_keep = torch.ones(
            source_length,
            dtype=torch.bool,
            device=source.device)

        is_word_start = is_word_start.long()
        # acts as a long length, so spans don't go over the end of doc
        is_word_start[-1] = 10e5
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            # random ratio disabled
            # source[indices[mask_random]] = torch.randint(
            #     1, len(self.vocab), size=(mask_random.sum(),))

        # if self.mask_span_distribution is not None:
        # assert len(lengths.size()) == 1
        # assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            # assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1

            # mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = self.mask_idx
                # random ratio disabled
                # source[indices[mask_random]] = torch.randint(
                #     1, len(self.vocab), size=(mask_random.sum(),))
        # else:
        #     # A bit faster when all lengths are 1
        #     while indices.size(0) > 0:
        #         uncompleted = is_word_start[indices + 1] == 0
        #         indices = indices[uncompleted] + 1
        #         mask_random = mask_random[uncompleted]
        #         if self.replace_length != -1:
        #             # delete token
        #             to_keep[indices] = 0
        #         else:
        #             # keep index, but replace it with [MASK]
        #             source[indices] = self.mask_idx
        #             source[indices[mask_random]] = torch.randint(
        #                 1, len(self.vocab), size=(mask_random.sum(),))

        #         assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(
                source, num_inserts / source.size(0))

        # aeq(source.eq(self.pad_idx).long().sum(), 0)
        final_length = source.size(0)
        return source, final_length

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = tokens.size(0)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(
            size=(
                num_tokens + n,
            ),
            dtype=torch.bool,
            device=tokens.device)
        noise_mask[noise_indices] = 1
        result = torch.ones([n + len(tokens)],
                            dtype=torch.long,
                            device=tokens.device) * -1

        # random ratio disabled
        # num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices] = self.mask_idx
        # result[noise_indices[:num_random]] = torch.randint(
        #    low=1, high=len(self.vocab), size=(num_random,))

        result[~noise_mask] = tokens

        # assert (result >= 0).all()
        return result


class MultiNoise(NoiseBaseForAPE):
    NOISES = {
        "sen_shuffling": SenShufflingNoise,
        "infilling": InfillingNoise,
        "mask": MaskNoiseForAPE
    }

    def __init__(self, noises={}, probs={}, **kwargs):
        assert ("src" in noises and "mt" in noises) \
               and ("src" in probs and "mt" in probs)
        assert len(noises["src"]) == len(probs["src"]) \
               and len(noises["mt"]) == len(probs["mt"])
        super().__init__(probs, **kwargs)

        self.noises = {}
        self.noises["src"] = MultiNoise.make_noise_list(
            noises["src"], probs["src"], **kwargs)
        self.noises["mt"] = MultiNoise.make_noise_list(
            noises["mt"], probs["mt"], **kwargs)

    @classmethod
    def make_noise_list(cls, noises=[], probs=[], **kwargs):
        l = []
        for i, n in enumerate(noises):
            noise_cls = cls.NOISES.get(n)
            if n is None:
                raise ValueError("Unknown noise function '%s'" % n)
            else:
                noise = noise_cls(probs[i], **kwargs)
                l.append(noise)
        return l

    def noise_source(self, source, length=None, **kwargs):
        for inp in {source, length}:
            assert type(inp) == dict \
                   and ("src" in inp and "mt" in inp)
        for noise in self.noises["src"]:
            source["src"], length["src"] = noise.noise_source(
                source["src"], length=length["src"], **kwargs)
        for noise in self.noises["mt"]:
            source["mt"], length["mt"] = noise.noise_source(
                source["mt"], length=length["mt"], **kwargs)
        return source, length

    def noise_batch(self, batch):
        src, src_len = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, [None] * batch.src.size(1))
        # noise_skip = batch.noise_skip
        # aeq(len(batch.noise_skip) == src.size(1))
        mt, mt_len = batch.mt if isinstance(batch.mt, tuple) \
            else (batch.mt, [None] * batch.mt.size(1))

        # src is [src_len x bs x feats]
        src_skipped = src[:self.skip_first, :, :]
        src = src[self.skip_first:]

        # mt is [mt_len * bs * feats]
        mt_skipped = mt[:self.skip_first, :, :]
        mt = mt[self.skip_first:]
        
        for i in range(src.size(1)):
            if hasattr(batch, 'corpus_id'):
                corpus_id = batch.corpus_id[i]
                if corpus_id.item() not in self.ids_to_noise:
                    continue
            src_tkn, mt_tkn = src[:, i, 0], mt[:, i, 0]
            src_mask = src_tkn.ne(self.pad_idx)
            mt_mask = mt_tkn.ne(self.pad_idx)

            masked_src = src_tkn[src_mask]
            noisy_src, noisy_src_len = self.noise_source(
                masked_src, length=src_len[i])
            
            masked_mt = mt_tkn[mt_mask]
            noisy_inp, noisy_inp_len = self.noise_source(
                {"src": masked_src, "mt": masked_mt},
                {"src": src_len[i], "mt": mt_len[i]})
            noisy_src, noisy_mt = noisy_inp["src"], noisy_inp["mt"]
            noisy_src_len = noisy_inp_len["src"]
            noisy_mt_len = noisy_inp_len["mt"]

            src_len[i], mt_len[i] = noisy_src_len, noisy_mt_len

            # src might increase length
            # so we need to resize the whole tensor
            src_delta = noisy_src_len - (src.size(0) - self.skip_first)
            if src_delta > 0:
                src_pad = torch.ones([src_delta],
                                     device=src.device,
                                     dtype=src.dtype)
                src_pad *= self.pad_idx
                src_pad = src_pad.unsqueeze(1).expand(-1, 15).unsqueeze(2)

                src = torch.cat([src, src_pad])
            src[:noisy_src.size(0), i, 0] = noisy_src

            # mt might increase length
            # so we need to resize the whole tensor
            mt_delta = noisy_mt_len - (mt.size(0) - self.skip_first)
            if mt_delta > 0:
                mt_pad = torch.ones([mt_delta],
                                    device=mt.device,
                                    dtype=mt.dtype)
                mt_pad *= self.pad_idx
                mt_pad = mt_pad.unsqueeze(1).expand(-1, 15).unsqueeze(2)

                mt = torch.cat([mt, mt_pad])
            mt[:noisy_mt.size(0), i, 0] = noisy_mt

        src = torch.cat([src_skipped['src'], src])
        mt = torch.cat([mt_skipped, mt])

        # remove useless pad
        src_max_len, mt_max_len = src_len.max(), mt_len.max()
        src, mt = src[:src_max_len, :, :], mt[:mt_max_len, :, :]

        batch.src = src, src_len
        batch.mt = mt, mt_len
        return batch
