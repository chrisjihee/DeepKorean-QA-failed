import collections
import contextlib
import json
import os
from urllib.request import urlopen

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer, load_vocab, whitespace_tokenize
from transformers.tokenization_utils_base import TextInput


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

    def tokenize_with_offset(self, text: TextInput, **kwargs):
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
    ts, os = tokenizer.tokenize_with_offset(tt)
    print(f"ts={ts}")
    print(f"os={os}")

    print('\t'.join(["token", "offset", "substr"]))
    for a, b in zip(ts, os):
        print('\t'.join([a, str(b), tt[b[0]:b[1]]]))
