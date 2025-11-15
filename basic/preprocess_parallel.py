import os
import pickle
from collections import Counter
from typing import List, Dict, Iterable, Tuple
import concurrent.futures as cf
from functools import partial

import numpy as np
import pandas as pd
from ftfy import fix_text
from wordsegment import load as ws_load, segment as ws_segment
import re
from text_preprocessor import TextPreprocessor, tokenize_ascii, encode_tokens

SOS, EOS, PAD, UNK = range(4)

SPECIAL_TOKENS = { "<sos>": SOS, "<eos>": EOS, "<pad>": PAD, "<unk>": UNK, }

ALLOWED_PUNCT = ".,;:!?\'\"-"
ALLOWED_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    + ALLOWED_PUNCT
)
ALLOWED_SET = set(ALLOWED_CHARS)

#Translation table to keep ASCII characters only
translation = {i: 32 for i in range(256)}  # map disallowed to space
for ch in ALLOWED_CHARS:
    translation[ord(ch)] = ord(ch)

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)*"     # hyphenated words
    r"|\d+(?:\.\d+)?"               # numbers
    r"|[.,;:!?'\"-]"                # punctuation
)

SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?'\"-])")


#Multi threaded workers for efficiency

def _init_worker():
    global PRE
    PRE = TextPreprocessor()


#build vocabulary
def build_vocab_on_texts(texts, lowercase= True):
    global PRE
    cnt = Counter()

    for raw in texts:
        t = PRE(raw)
        if lowercase:
            t = t.lower()
        cnt.update(tokenize_ascii(t))
    return cnt

def _encode_lengths(texts, word_to_idx):
    global PRE
    total = 0
    for raw in texts:
        t = PRE(raw).lower()
        toks = tokenize_ascii(t)
        total += (1 + len(toks) + 1)  # SOS + tokens + EOS
    return total

def _encode_block(texts, word_to_idx):
    global PRE
    out = []
    for raw in texts:
        t = PRE(raw).lower()
        toks = tokenize_ascii(t)
        out.append(SOS)
        for tok in toks:
            out.append(word_to_idx.get(tok, UNK))
        out.append(EOS)
    return out


#reacding dataset from CSV files downloaded from kaggle as huggigface was blocked in my development environment

def iter_csv_text_chunks(path, chunksize = 50000, text_col = "text"):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if text_col not in chunk.columns:
            raise ValueError(
                f"Column '{text_col}' not found in {path}. Columns: {list(chunk.columns)}"
            )
        yield chunk[text_col].astype(str).tolist()



def build_vocab_from_csv( train_csv, workers = None, chunksize = 50000, ) :
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    #count ds len
    with cf.ProcessPoolExecutor( max_workers=workers, initializer=_init_worker, initargs=()) as ex:
        futures = []
        for texts in iter_csv_text_chunks(train_csv, chunksize=chunksize):
            futures.append(ex.submit(build_vocab_on_texts, texts, True))

        global_cnt = Counter()
        for fut in cf.as_completed(futures):
            global_cnt.update(fut.result())

    #assign IDs
    word_to_idx = {}
    next_id = 4  # reserve 0..3 for special tokens

    for tok, _ in global_cnt.most_common():
        if tok not in word_to_idx:
            word_to_idx[tok] = next_id
            next_id += 1

    # Add special tokens
    for s, idx in SPECIAL_TOKENS.items():
        word_to_idx[s] = idx

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    return word_to_idx, idx_to_word

#save ds as memmap for streamed loading
def encode_csv_to_memmap(
    csv_path: str,
    word_to_idx: Dict[str, int],
    out_pickle: str,
    out_memmap: str,
    workers: int = None,
    chunksize: int = 50_000,
):
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    #compute total length
    total_len = 0
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(False,)
    ) as ex:
        futures = []
        for texts in iter_csv_text_chunks(csv_path, chunksize=chunksize):
            futures.append(ex.submit(_encode_lengths, texts, word_to_idx))
        for fut in cf.as_completed(futures):
            total_len += fut.result()

    #allocate memmap
    arr = np.memmap(out_memmap, dtype=np.int32, mode="w+", shape=(total_len,))
    pos = 0
    ids_list = []

    with cf.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(False,)
    ) as ex:

        futures = []
        chunk_cache = []

        for texts in iter_csv_text_chunks(csv_path, chunksize=chunksize):
            chunk_cache.append(texts)
            futures.append(ex.submit(_encode_block, texts, word_to_idx))

        # Maintain order
        for texts, fut in zip(chunk_cache, futures):
            flat_ids = fut.result()
            L = len(flat_ids)
            arr[pos:pos+L] = np.asarray(flat_ids, dtype=np.int32)
            pos += L

            ids_list.append(None)

    arr.flush()

    # save meta data in pickle
    meta = { 
        "memmap": out_memmap,
        "num_sequences": None,  # not computing per-sample
        "total_tokens": int(total_len),
    }
    with open(out_pickle, "wb") as f:
        pickle.dump(meta, f)

    return int(total_len), -1

def main():

    train_csv = "train.csv"
    val_csv ="validation.csv"
    workers = None
    chunksize = 50000

    print("Building vocab...")
    word_to_idx, idx_to_word = build_vocab_from_csv(train_csv, workers=workers, chunksize=chunksize)

    with open(f"word_to_idx.pkl", "wb") as f:
        pickle.dump(word_to_idx, f)
    with open(f"idx_to_word.pkl", "wb") as f:
        pickle.dump(idx_to_word, f)

    print("Encoding training set...")
    train_total, _ = encode_csv_to_memmap( train_csv, word_to_idx,  out_pickle=f"train_meta.pkl",
        out_memmap=f"train_tokens.int32", workers=workers, chunksize=chunksize,)

    print("Encoding validation set...")
    val_total, _ = encode_csv_to_memmap(val_csv, word_to_idx, out_pickle=f"val_meta.pkl",
        out_memmap=f"val_tokens.int32", workers=workers, chunksize=chunksize, )

    print("Done.")
    print(f"Vocab size: {len(word_to_idx)}")
    print(f"Train total tokens: {train_total}")
    print(f"Val total tokens:   {val_total}")
    print("Memmaps written as int32 token streams.")


if __name__ == "__main__":
    main()
