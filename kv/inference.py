import random
import json
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import evaluate 
import pickle
from nltk.translate.bleu_score import sentence_bleu

from text_preprocessor import TextPreprocessor, tokenize_ascii, encode_tokens
from time import perf_counter
from basic_transformer_optimized import BasicTransformerLM


RANDOM_SEED = 4174

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#Load vocab
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

#special tokens and hyperparms
SOS, EOS, PAD, UNK = range(4)

vocab_size = len(word_to_idx)
seq_len = 64
embed_dim = 300
num_heads = 6
num_layers = 4
vocab_size = 52130  

#load model
checkpoint = torch.load("best_model.pt", map_location=DEVICE)
raw_sd = checkpoint["model"]

#due to DDP/compiled models : remove possible "_orig_mod." prefix from keys
clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

model = BasicTransformerLM( embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, 
                            seq_len=seq_len, vocab_size=vocab_size).to(DEVICE)
model.load_state_dict(clean_sd)
model.eval()

def load_texts_from_csv(path: str):
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist()


def sample_examples(csv_file, count_samples):
    val_texts = load_texts_from_csv(csv_file)
    data_len = len(val_texts)

    indices = np.random.randint(0, data_len, size=count_samples).astype(int)
    samples = [val_texts[i] for i in indices]

    preprocessor = TextPreprocessor()
    ids_list = []

    for raw in samples:
        t = preprocessor(raw).lower()
        toks = tokenize_ascii(t)
        ids = encode_tokens(toks, word_to_idx)
        ids_list.append(ids)

    return ids_list


def decode_tokens(ids):
    words = []
    for i in ids:
        if i in idx_to_word:
            words.append(idx_to_word[i])
        else:
            words.append("<unk>")
    return " ".join(words)


#generation helpers
def generate_greedy(model, prompt_ids, max_new = 50, temperature = 1.0, eos_id = EOS):
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_new):
        with torch.no_grad():
            logits = model(x)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        x = torch.cat([x, next_id], dim=1)

        if int(next_id.item()) == eos_id:
            break

    return x.squeeze(0).tolist()



# KV-cached decoding
@torch.no_grad()
def generate_kv(model, prompt_ids, max_new: int = 50, eos_id: int = EOS):

    model.eval()

    #Run a full forward on the prompt to initialize cache position
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    _, T = x.shape

    #initialize empty per-layer cache
    cache = {
        "layers": [None for _ in model.blocks],  # KV cache for each layer
        "pos": T,
    }

    logits = model(x)
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    out_ids = prompt_ids[:] + [next_token]

    #terative autoregressive decode using forward_kv
    for _ in range(max_new - 1):
        xt = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        logits, cache = model.forward_kv(xt, cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        out_ids.append(next_token)

        if next_token == eos_id:
            break

    return out_ids



def perplexity(model, input_ids, target_ids):
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)  # (1, T, V)

    # match lengths
    T = min(logits.shape[1], y.shape[1])
    logits = logits[:, :T, :]
    y = y[:, :T]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=PAD,
        reduction="mean",
    )

    return torch.exp(loss).item()


# ---------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------
def compute_bleu(pred: str, ref: str):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    return sentence_bleu([ref_tokens], pred_tokens)


# ---------------------------------------------------------------------
# Evaluation for a given decoding function
# ---------------------------------------------------------------------
def eval_generation(decode_fn, name: str, prompts):
    runtimes = []
    bleus = []
    tps = []
    records = []

    for i, (prompt, target) in enumerate(prompts, 1):
        start = perf_counter()
        out_ids = decode_fn(prompt)
        elapsed = perf_counter() - start

        # Separate generated continuation
        gen_cont = out_ids[len(prompt):]
        tokens_generated = max(1, len(gen_cont))

        gen_text = decode_tokens(gen_cont)
        ref_text = decode_tokens(target)

        b = compute_bleu(gen_text, ref_text)

        toks_per_sec = tokens_generated / max(1e-9, elapsed)

        print(
            f"[{name}] sample {i}: {tokens_generated} tokens in "
            f"{elapsed:.3f}s ({toks_per_sec:.1f} tok/s), BLEU={b:.4f}"
        )

        runtimes.append(elapsed)
        bleus.append(b)
        tps.append(toks_per_sec)

        records.append(
            {
                "sample": i,
                "decoder": name,
                "tokens_generated": tokens_generated,
                "elapsed_sec": elapsed,
                "tokens_per_sec": toks_per_sec,
                "bleu": b,
                "generated": gen_text,
                "reference": ref_text,
            }
        )

    summary = {
        "decoder": name,
        "avg_tokens_per_sec": sum(tps) / len(tps),
        "avg_bleu": sum(bleus) / len(bleus),
    }

    return records, summary


def get_sample(examples, idx):
    x = examples[idx]
    prompt = x[:5]         # first 5 tokens as prompt
    target = x[5:55]       # next 50 tokens as continuation
    return prompt, target



def main():
    num_examples = 5

    examples = sample_examples("validation.csv", num_examples)
    prompts = [get_sample(examples, idx) for idx in range(num_examples)]

    summaries = []
    records_all = []

    # Greedy decoding
    rec, summ = eval_generation(
        decode_fn=lambda prompt: generate_greedy(model, prompt, max_new=50),
        name="greedy_search",
        prompts=prompts,
    )
    records_all += rec
    summaries.append(summ)

    #generate with KV-cached decoding
    rec, summ = eval_generation(
        decode_fn=lambda prompt: generate_kv(model, prompt, max_new=50),
        name="kv_cache",
        prompts=prompts,
    )
    records_all += rec
    summaries.append(summ)

    print("\nAverages:")
    for s in summaries:
        print(
            f"{s['decoder']:>18} , avg tok/s: {s['avg_tokens_per_sec']:.1f} "
            f", avg BLEU: {s['avg_bleu']:.4f}"
        )



if __name__ == "__main__":
    main()
