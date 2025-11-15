import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

#special tokens
SOS, EOS, PAD, UNK = range(4)
PAD_IDX = PAD

#hyperparameters
batch_size = 64 * 10  # 640
seq_len = 64
embed_dim = 300
num_heads = 6
num_layers = 4
lr = 3e-4
epochs = 25
vocab_size = 52130
stride = 32


class TinyStoriesDataset(Dataset):
    def __init__(self, mmap_path, seq_len=64, stride=None, dtype=np.int32):
        self.arr = np.memmap(mmap_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        # number of windows
        self.n = max(0, (len(self.arr) - (seq_len + 1)) // self.stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.stride
        end = start + self.seq_len + 1
        chunk = np.array(self.arr[start:end], copy=True)

        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional embedding layer.
    """

    def __init__(self, max_len, embed_dim):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)           # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )                                                           # (embed_dim/2,)
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        return x + self.pe[:, : x.size(1), :]


class LayerNorm(nn.Module):
    """
    Simple LayerNorm (last-dim).
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class AttentionHead(nn.Module):
    """
    Single-head causal self-attention.
    """

    def __init__(self, embed_dim, head_dim, max_len=seq_len):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)

        # causal mask (T, T)
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, _ = x.size()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if kv_cache is not None:
            # Concatenate along sequence dimension
            k = torch.cat([kv_cache["k"], k], dim=1)
            v = torch.cat([kv_cache["v"], v], dim=1)

        # q: (B, T, H), k: (B, T_total, H)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        # causal mask: T x T_total
        T_total = k.size(1)
        causal_mask = self.mask[:T, :T_total]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        new_cache = {"k": k, "v": v}

        return out, att, new_cache


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with KV cache.
    """

    def __init__(self, embed_dim, num_heads, max_len=seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, self.head_dim, max_len) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_weights=False, cache=None):
        if cache is None:
            cache = [None] * self.num_heads

        outs = []
        att_w = [] if return_weights else None
        new_cache = []

        for head, head_cache in zip(self.heads, cache):
            out, wt, updated_cache = head(x, head_cache)
            outs.append(out)
            new_cache.append(updated_cache)
            if return_weights:
                att_w.append(wt)

        out = self.proj(torch.cat(outs, dim=-1))

        if return_weights:
            return out, att_w, new_cache
        return out, None, new_cache


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x, cache=None, return_weights=False):

        att_out, att_w, new_cache = self.attn(x, return_weights=return_weights, cache=cache)
        x = x + self.ln1(att_out)
        x = x + self.ln2(self.ff(x))
        return x, att_w, new_cache


class BasicTransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(seq_len, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, return_weights=False):
        x = self.embed(x)          # (B, T, D)
        x = self.pos(x)            # add positional encoding

        att_all_layers = [] if return_weights else None
        dummy_cache = [None] * len(self.blocks)

        for blk, layer_cache in zip(self.blocks, dummy_cache):
            x, att_w, _ = blk(x, cache=layer_cache, return_weights=return_weights)
            if return_weights:
                att_all_layers.append(att_w)

        x = self.ln(x)
        logits = self.out(x)

        if return_weights:
            return logits, att_all_layers
        return logits

    def forward_kv(self, x, cache):
        x = self.embed(x)
        x = self.pos(x)

        if cache is None or "layers" not in cache:
            cache = {"layers": [None] * len(self.blocks)}

        new_layers = []

        for blk, layer_cache in zip(self.blocks, cache["layers"]):
            x, _, updated_layer_cache = blk(x, cache=layer_cache, return_weights=False)
            new_layers.append(updated_layer_cache)

        new_cache = {"layers": new_layers}

        x = self.ln(x)
        logits = self.out(x)
        return logits, new_cache


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        total += loss.item()
        batches += 1

    val_loss = total / max(1, batches)
    ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
    return val_loss, ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = TinyStoriesDataset(
        "train_tokens.int32", seq_len=seq_len, stride=stride
    )
    val_ds = TinyStoriesDataset(
        "val_tokens.int32", seq_len=seq_len, stride=stride
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Model
    model = BasicTransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
    ).to(device)

    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        start_time = time.time()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches += 1

        avg_train = total_loss / max(1, batches)
        train_ppl = math.exp(avg_train) if avg_train < 20 else float("inf")

        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{now}] Epoch {epoch}/{epochs} "
            f"train_loss={avg_train:.4f} train_ppl={train_ppl:.2f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}",
            flush=True,
        )

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                "best_model.pt",
            )

        print(
            f"Epoch time: {time.time() - start_time:.2f} sec",
            flush=True,
        )


if __name__ == "__main__":
    main()
