import re
from typing import List

from ftfy import fix_text
from wordsegment import load as ws_load, segment as ws_segment


SOS, EOS, PAD, UNK = range(4)
ALLOWED_PUNCT = ".,;:!?'\"-"

ALLOWED_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    + ALLOWED_PUNCT
)
ALLOWED_SET = set(ALLOWED_CHARS)


def ascii_filter(text):
    return "".join(ch if ch in ALLOWED_SET else "" for ch in text)

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)*"       #words with optional hyphens
    r"|\d+(?:\.\d+)?"                 #numbers
    r"|[.,;:!?'\"-]"                  #punctuation
)


class TextPreprocessor:
    def __init__(self):
        ws_load()

    def _de_stutter(self, token):
        """
        Remove stuttered patterns like:
        "h-e-l-l-o" → "o"
        """
        parts = token.split("-")
        if len(parts) >= 3 and all(len(p) == 1 for p in parts[:-1]):
            return parts[-1]
        return token

    def _segment_core(self, token):
        """
        Segment long words into components using `wordsegment`.
        Only applies to alphabetic words of length >= 6.
        """
        if len(token) >= 6 and token.isalpha() and not token.isupper():
            return ws_segment(token.lower())
        return [token]

    def clean_text(self, text):
        """
        Clean and preprocess a string.
        Steps:
          1. Fix odd unicode (mojibake)
          2. Keep only allowed ASCII chars
          3. Normalize whitespace
          4. De-stutter words
          5. Word segmentation
          6. Remove unwanted spaces before punctuation
        """
        #fix unicode
        t = fix_text(text)

        #ASCII-filter
        t = ascii_filter(t)

        #normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            return ""

        #de-stutter, segmentation
        out = []
        for tok in t.split():
            tok = self._de_stutter(tok)

        t = " ".join(out)

        #remove space before punctuation
        t = re.sub(r"\s+([.,;:!?'\"-])", r"\1", t)

        return t

    def __call__(self, text):
        return self.clean_text(text)

def tokenize_ascii(text):
    """
    Tokenize ASCII text into words, numbers, and punctuation.
    """
    return TOKEN_PATTERN.findall(text)

def encode_tokens(tokens, word_to_idx):
    return [SOS] + [word_to_idx.get(t, UNK) for t in tokens] + [EOS]
