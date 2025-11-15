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

model = BasicTransformerLM( embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, seq_len=seq_len, vocab_size=vocab_size).to(DEVICE)
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


def generate_beam_search( model, prompt_ids, max_new = 50, temperature = 1.0, k = 5, eos_id = EOS, length_penalty = 0.0, ):
    model.eval()

    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    beams = [(prompt, 0.0, False)]
    completed = []

    for _ in range(max_new):
        new_beams = []

        for seq, lp, finished in beams:
            if finished:
                new_beams.append((seq, lp, True))
                continue

            with torch.no_grad():
                logits = model(seq)[:, -1, :] / max(temperature, 1e-6)
                logprobs = F.log_softmax(logits, dim=-1)
                topk_logprobs, topk_ids = torch.topk(logprobs, k, dim=-1)

            for j in range(k):
                token = topk_ids[0, j].view(1, 1)
                new_seq = torch.cat([seq, token], dim=1)
                new_lp = lp + float(topk_logprobs[0, j].item())
                finished_flag = int(token.item()) == eos_id

                if length_penalty > 0:
                    L = new_seq.shape[1]
                    norm_lp = new_lp / (L ** length_penalty)
                else:
                    norm_lp = new_lp

                new_beams.append((new_seq, norm_lp, finished_flag))

        #keep top-k by score
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:k]

        #move finished beams to completed
        still_alive = []
        for seq, score, fin in beams:
            if fin:
                completed.append((seq, score, True))
            else:
                still_alive.append((seq, score, False))
        beams = still_alive

        #stop if enough finished and none alive
        if len(completed) >= k and not beams:
            break

    #pick best completed. if nothing finished, pick best alive 
    if completed:
        completed.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
    else:
        beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = beams[0][0]

    return best_seq.squeeze(0).tolist()


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

def compute_bleu(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    return sentence_bleu([ref_tokens], pred_tokens)

#Evaluation loop
def eval_generation(decode_fn, name, prompts):
    runtimes = []
    bleus = []
    tps = []
    records = []

    for i, (prompt, target) in enumerate(prompts, 1):
        start = perf_counter()
        out_ids = decode_fn(prompt)
        elapsed = perf_counter() - start

        # separate generated continuation
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


def plot_attention(model, token_ids, sample_name = "sample"):
    model.eval()
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    #tokens for first 20 positions
    ts = decode_tokens(token_ids[:20])
    tokens = [str(t) for t in ts.split()]

    with torch.no_grad():
        heads_weights = model(x, return_weights=True)

    print(
        "heads_weights shapes:",
        len(heads_weights),
        len(heads_weights[0]),
        heads_weights[0][0].shape if len(heads_weights[0]) > 0 else None,
    )

    for l, att_l in enumerate(heads_weights):
        for h, att in enumerate(att_l):
            att_map = att[0].cpu().numpy()  # (T, T)

            plt.figure(figsize=(6, 6))
            plt.imshow(att_map[:20, :20], cmap="viridis")
            plt.title(f"{sample_name} Layer {l} Head {h} Attention")
            plt.xlabel("Key positions")
            plt.ylabel("Query positions")
            plt.colorbar()

            plt.xticks(range(min(20, len(tokens))), tokens[:20], fontsize=8, rotation=85)
            plt.yticks(range(min(20, len(tokens))), tokens[:20], fontsize=8)

            plt.tight_layout()
            fname = f"attention_{sample_name}_layer{l}_head{h}.png"
            plt.savefig(fname)
            plt.close()

            print(f"Saved {fname}")


def main():
    # evaluate on random examples
    num_examples = 5
    examples = sample_examples("validation.csv", num_examples)
    prompts = [get_sample(examples, idx) for idx in range(num_examples)]

    records_all = []
    summaries = []

    # Greedy decoding
    rec, summ = eval_generation(
        decode_fn=lambda prompt: generate_greedy(model, prompt, max_new=50),
        name="greedy_search",
        prompts=prompts,
    )
    records_all += rec
    summaries.append(summ)

    # Beam search k=5
    rec, summ = eval_generation(
        decode_fn=lambda prompt: generate_beam_search(model, prompt, max_new=50, k=5),
        name="beam_search_k5",
        prompts=prompts,
    )
    records_all += rec
    summaries.append(summ)

    # Beam search k=10
    rec, summ = eval_generation(
        decode_fn=lambda prompt: generate_beam_search(model, prompt, max_new=50, k=10),
        name="beam_search_k10",
        prompts=prompts,
    )
    records_all += rec
    summaries.append(summ)

    print("\n== Averages ==")
    for s in summaries:
        print(
            f"{s['decoder']:>18} | avg tok/s: {s['avg_tokens_per_sec']:.1f} | "
            f"avg BLEU: {s['avg_bleu']:.4f}"
        )

    # Save results
    df = pd.DataFrame(records_all)
    df_summary = pd.DataFrame(summaries)

    df.to_csv("beam_eval_per_sample.csv", index=False)
    df_summary.to_csv("beam_eval_summary.csv", index=False)

    with open("beam_eval_per_sample.json", "w") as f:
        json.dump(records_all, f, indent=2)

    with open("beam_eval_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(
        "Saved: beam_eval_per_sample.csv, beam_eval_summary.csv, "
        "beam_eval_per_sample.json, beam_eval_summary.json"
    )

    # Simple plots: tokens/sec and BLEU averages
    plt.figure(figsize=(6, 4))
    plt.bar(df_summary["decoder"], df_summary["avg_tokens_per_sec"])
    plt.ylabel("Avg tokens / sec")
    plt.title("Decoding speed")
    plt.tight_layout()
    plt.savefig("beam_speed.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(df_summary["decoder"], df_summary["avg_bleu"])
    plt.ylabel("Avg BLEU")
    plt.title("Output quality")
    plt.tight_layout()
    plt.savefig("beam_bleu.png")
    plt.show()

    # Attention visualizations for 3 validation samples
    print("\nVisualizing attention for 3 validation samples...\n")
    for idx in range(3):
        x = examples[idx]
        token_ids = np.zeros(50, dtype=int)
        token_ids[: min(20, len(x))] = x[:20]
        token_ids[20:] = PAD
        plot_attention(model, token_ids.tolist(), sample_name=f"idx{idx}")

    print("\nAttention visualizations saved as PNG files.")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
