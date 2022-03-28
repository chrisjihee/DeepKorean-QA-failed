import collections
import contextlib
import json
import os
from typing import List
from typing import Optional, Union
from urllib.request import urlopen

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer, load_vocab, whitespace_tokenize
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType


class KorbertTokenizer(BertTokenizer):
    """
    Construct a BERT tokenizer for morpheme-analized data.
    """

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            analyzer_netloc=None,
            tokenize_online=False,
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
    ):
        super(BertTokenizer, self).__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            if tokenize_online and analyzer_netloc:
                self.online_tokenizer = OnlineTokenizer(
                    netloc=analyzer_netloc,
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )
            else:
                self.offline_tokenizer = OfflineTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )
        self.wordpiece_tokenizer = KorbertWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def tokenize(self, morps: TextInput, **kwargs):
        sub_tokens = []
        for token in self.offline_tokenizer.tokenize(morps):
            if token not in self.all_special_tokens:
                token += '_'
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)
        return sub_tokens

    def tokenize_online(self, text: TextInput, **kwargs):
        split_tokens = []
        split_offsets = []
        tokens, offsets = self.online_tokenizer.tokenize_with_offset(text)
        for token, offset in zip(tokens, offsets):
            start, end = offset
            token += '_'
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                if token == sub_token:
                    split_offsets.append(offset)
                else:
                    sub_end = min(start + len(sub_token), end)
                    split_offsets.append((start, sub_end))
                    start = sub_end
                split_tokens.append(sub_token)
        # print(f"(tokens, offsets)={list(zip(split_tokens, split_offsets))}")
        return split_tokens, split_offsets

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
    ) -> BatchEncoding:
        ##rev##
        def get_input_ids_with_extra(text):
            if isinstance(text, str):
                tokens, offsets = self.tokenize_online(text, **kwargs)
                return self.convert_tokens_to_ids(tokens), offsets, None
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    all_tokens, all_offsets, all_word_ids = \
                        zip(*[(ts, os, [(i,)] * len(ts)) for i, (ts, os) in
                              enumerate(self.tokenize_online(t, is_split_into_words=True, **kwargs) for t in text)])
                    all_tokens = sum(all_tokens, [])
                    all_offsets = sum(all_offsets, [])
                    all_word_ids = sum(all_word_ids, [])
                    return self.convert_tokens_to_ids(all_tokens), all_offsets, all_word_ids
                else:
                    return self.convert_tokens_to_ids(text), None, None
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text, None, None
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ##rev##
        first_ids, first_offsets, first_word_ids = get_input_ids_with_extra(text)
        second_ids, second_offsets, second_word_ids = get_input_ids_with_extra(text_pair) if text_pair is not None else (None, None, None)

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            offsets=first_offsets,  ##rev##
            pair_offsets=second_offsets,  ##rev##
            word_ids=first_word_ids,  ##rev##
            pair_word_ids=second_word_ids,  ##rev##
            return_offsets_mapping=return_offsets_mapping,  ##rev##
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
            **kwargs,  ##rev##
        )

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            offsets=None,  ##rev##
            pair_offsets=None,  ##rev##
            word_ids=None,  ##rev##
            pair_word_ids=None,  ##rev##
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
    ) -> BatchEncoding:
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        overflowing_offsets = None
        overflowing_word_ids = None
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            ##rev##
            if offsets is not None:
                offsets, pair_offsets, overflowing_offsets = self.truncate_sequences(
                    offsets,
                    pair_ids=pair_offsets,
                    num_tokens_to_remove=total_len - max_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )
            ##rev##
            if word_ids is not None:
                word_ids, pair_word_ids, overflowing_word_ids = self.truncate_sequences(
                    word_ids,
                    pair_ids=pair_word_ids,
                    num_tokens_to_remove=total_len - max_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length
            if offsets and overflowing_offsets:
                encoded_inputs["overflowing_offsets"] = overflowing_offsets  ##rev##
            if word_ids and overflowing_word_ids:
                encoded_inputs["overflowing_word_ids"] = overflowing_word_ids  ##rev##

        # Add special tokens
        no_offset_tokens = {self.cls_token_id, self.sep_token_id, self.pad_token_id}  ##rev##
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            if offsets is not None:  ##rev##
                offset_mapping = [x if x not in no_offset_tokens else (0, 0) for x in self.build_inputs_with_special_tokens(offsets, pair_offsets)]
            if word_ids is not None:  ##rev##
                sequence_word_ids = [x if x not in no_offset_tokens else None for x in self.build_inputs_with_special_tokens(word_ids, pair_word_ids)]
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            if offsets is not None:  ##rev##
                offset_mapping = offsets + pair_offsets if pair else offsets
            if word_ids is not None:  ##rev##
                sequence_word_ids = word_ids + pair_word_ids if pair else word_ids

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if word_ids is not None:  ##rev##
            encoded_inputs["word_ids"] = sequence_word_ids
        if return_offsets_mapping and offsets is not None:  ##rev##
            encoded_inputs["offset_mapping"] = offset_mapping
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs


class OfflineTokenizer(BasicTokenizer):
    """
    Constructs a BasicTokenizer that will run space splitting.
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

    def tokenize(self, text, never_split=None):
        return super().tokenize(text, never_split=never_split)

    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if char == ' ':
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]


class OnlineTokenizer(BasicTokenizer):
    """
    Constructs a BasicTokenizer that will run online pos tagging..
    """

    def __init__(self, netloc: str, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )
        self.tagger = MLTagger(netloc=netloc)

    def tokenize(self, text, never_split=None):
        morps = self.tagger.tag(text=text)
        return super().tokenize(' '.join(morps), never_split=never_split)

    def tokenize_with_offset(self, text, never_split=None):
        morps, offsets = self.tagger.tag_with_offset(text=text)
        return super().tokenize(' '.join(morps), never_split=never_split), offsets

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if char == ' ':
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class MLTagger:  # jihee.ryu @ 2021-09-22
    def __init__(self, netloc: str):
        self.netloc = netloc
        self.api = f"http://{self.netloc}/interface/lm_interface"

    def do_lang(self, text: str):
        param = {"argument": {"analyzer_types": ["MORPH"], "text": text}}
        try:
            with contextlib.closing(urlopen(self.api, json.dumps(param).encode())) as res:
                return json.loads(res.read().decode())['return_object']['json']
        except:
            print("\n" + "=" * 120)
            print(f'[error] Can not connect to WiseAPI[{self.api}]')
            print("=" * 120 + "\n")
            exit(1)

    def tag(self, text: str):
        ndoc = self.do_lang(text)
        morps = [f"{m['lemma']}/{m['type']}" for s in ndoc['sentence'] for m in s['morp']]
        return morps

    def tag_with_offset(self, text: str):
        ndoc = self.do_lang(text)
        morps = [f"{m['lemma']}/{m['type']}" for s in ndoc['sentence'] for m in s['morp']]
        byte_starts = [int(m['position']) for s in ndoc['sentence'] for m in s['morp']]
        text_bytes = text.encode()
        try:
            char_starts = [len(text_bytes[0:x].decode()) for x in byte_starts]
        except:
            print("\n" + "=" * 120)
            print(f'[error] Can NOT decode to specific byte')
            print("=" * 120 + "\n")
            with open("ndoc.json", 'w') as out:
                json.dump(ndoc, out, ensure_ascii=False, indent="  ")
            print(f'text={text}')
            print(f'morps={morps}')
            print(f'byte_starts={byte_starts}')
            for x in byte_starts:
                print(f'text_bytes[0:{x}]={text_bytes[0:x].decode()}')
            exit(1)

        char_starts.append(len(text))
        offsets = list()
        for i, morp in enumerate(morps):
            char_start = char_starts[i]
            char_end = char_starts[i + 1]
            if text[char_end - 1].isspace():
                char_end = char_end - 1
            if char_end < char_start:
                char_end = char_start
            offset = (char_start, char_end)
            offsets.append(offset)
        # print(f"text={text}")
        # print(f"(morps, offsets)={list(zip(morps, offsets))}")
        return morps, offsets


class KorbertWordpieceTokenizer(WordpieceTokenizer):
    """Runs WordPiece tokenization without '##'."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        super().__init__(
            vocab=vocab,
            unk_token=unk_token,
            max_input_chars_per_word=max_input_chars_per_word,
        )

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


if __name__ == "__main__":
    plain = "[CLS] 한국어 사전학습 모델을 공유합니다. [SEP] 지금까지 금해졌다."
    morps = "[CLS] 한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF [SEP] 지금/NNG 까지/JX 금하/VV 어/EC 지/VX 었/EP 다/EF ./SF"
    print(f"plain={plain}")
    print(f"morps={morps}")

    tokenizer1A = AutoTokenizer.from_pretrained(
        "pretrained/KoELECTRA-Base-v3",
        max_len=512,
        use_fast=True,
    )
    tokenizer1B = BertTokenizer(
        vocab_file="pretrained/KoELECTRA-Base-v3/vocab.txt",
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )
    tokenizer2A = KorbertTokenizer.from_pretrained(
        "pretrained/KorBERT-Base-morp",
        max_len=512,
        use_fast=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )
    tokenizer2B = KorbertTokenizer(
        vocab_file="pretrained/KorBERT-Base-morp/vocab.txt",
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )

    print(f"tokens from plain={tokenizer1A.tokenize(plain)}")
    print(f"tokens from plain={tokenizer1B.tokenize(plain)}")
    print(f"tokens from morps={tokenizer2A.tokenize(morps)}")
    print(f"tokens from morps={tokenizer2B.tokenize(morps)}")

    tt = "[CLS] 한국어 사전학습 모델을 공유합니다. [SEP] 지금까지 금해졌다."
    print(f"text={tt}")

    tagger = MLTagger(netloc="129.254.164.137:19001")
    ms = tagger.tag(tt)
    print(f"morps={ms}")
    print(f"morps={' '.join(ms)}")

    tokenizer = KorbertTokenizer(
        vocab_file="pretrained/KorBERT-Base-morp/vocab.txt",
        analyzer_netloc="129.254.164.137:19001",
        tokenize_online=True,
        tokenize_chinese_chars=False,
        do_lower_case=False,
        never_split=None,
        strip_accents=None,
    )
    ts, os = tokenizer.tokenize_online(tt)
    print(f"ts={ts}")
    print(f"os={os}")

    print('\t'.join(["token", "offset", "substr"]))
    for a, b in zip(ts, os):
        print('\t'.join([a, str(b), tt[b[0]:b[1]]]))
