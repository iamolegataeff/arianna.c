"""
arianna.c - PyTorch Training Script
Train personality weights, export to C-compatible binary

PyTorch for training (gradients), C for inference (speed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import time
import os
import math

# ============================================================
# Configuration - must match arianna.h
# ============================================================

DIM = 128
N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS
HIDDEN_DIM = 512
MAX_SEQ_LEN = 256
VOCAB_SIZE = 256  # char-level

# Training config
BATCH_SIZE = 64
SEQ_LEN = 128
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 100
WARMUP_ITERS = 100

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# Data Loading
# ============================================================

def load_corpus(path):
    """Load text corpus and convert to char-level tokens"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Filter to ASCII
    text = ''.join(c for c in text if ord(c) < 256)

    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    print(f"Loaded {len(data):,} characters ({len(data)/1e6:.2f}M)")

    # Split train/val
    n = int(len(data) * 0.9)
    return data[:n], data[n:]

def get_batch(data, batch_size, seq_len):
    """Get random batch of sequences"""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# Model
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x, start_pos=0):
        seq_len = x.shape[1]
        cos = self.cos[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)

        # Reshape x for rotation: [batch, seq, heads, head_dim]
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]

        # Apply rotation
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

        return x_rot

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.rope = RotaryEmbedding(HEAD_DIM)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, N_HEADS, HEAD_DIM)
        k = self.wk(x).view(batch, seq_len, N_HEADS, HEAD_DIM)
        v = self.wv(x).view(batch, seq_len, N_HEADS, HEAD_DIM)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, DIM)
        return self.wo(out)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, HIDDEN_DIM, bias=False)
        self.w2 = nn.Linear(HIDDEN_DIM, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.attn = Attention()
        self.ln2 = RMSNorm(DIM)
        self.ffn = FFN()

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class Arianna(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_final = RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)

        # Tie weights
        self.output.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        x = self.tok_emb(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.output(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def save_c_weights(self, path):
        """Export weights to C-compatible binary format"""
        with open(path, 'wb') as f:
            # Write config
            f.write(struct.pack('i', DIM))
            f.write(struct.pack('i', N_LAYERS))
            f.write(struct.pack('i', N_HEADS))
            f.write(struct.pack('i', HEAD_DIM))
            f.write(struct.pack('i', HIDDEN_DIM))
            f.write(struct.pack('i', MAX_SEQ_LEN))
            f.write(struct.pack('i', VOCAB_SIZE))

            # Token embeddings
            f.write(self.tok_emb.weight.detach().cpu().numpy().astype(np.float32).tobytes())

            # Attention weights (wq, wk, wv, wo for all layers)
            # IMPORTANT: .T creates a view, need .copy() for contiguous memory!
            for name in ['wq', 'wk', 'wv', 'wo']:
                for layer in self.layers:
                    w = getattr(layer.attn, name).weight.detach().cpu().numpy().astype(np.float32)
                    f.write(np.ascontiguousarray(w.T).tobytes())

            # FFN weights
            for layer in self.layers:
                w = layer.ffn.w1.weight.detach().cpu().numpy().astype(np.float32)
                f.write(np.ascontiguousarray(w.T).tobytes())
            for layer in self.layers:
                w = layer.ffn.w2.weight.detach().cpu().numpy().astype(np.float32)
                f.write(np.ascontiguousarray(w.T).tobytes())

            # Layer norms
            for layer in self.layers:
                f.write(layer.ln1.weight.detach().cpu().numpy().astype(np.float32).tobytes())
            for layer in self.layers:
                f.write(layer.ln2.weight.detach().cpu().numpy().astype(np.float32).tobytes())

            # Final layer norm and output
            f.write(self.ln_final.weight.detach().cpu().numpy().astype(np.float32).tobytes())
            w = self.output.weight.detach().cpu().numpy().astype(np.float32)
            f.write(np.ascontiguousarray(w.T).tobytes())

        print(f"Saved C-compatible weights to {path}")

# ============================================================
# Training
# ============================================================

def get_lr(iter, warmup_iters=WARMUP_ITERS, max_iters=MAX_ITERS, max_lr=LEARNING_RATE, min_lr=1e-5):
    """Learning rate with warmup and cosine decay"""
    if iter < warmup_iters:
        return max_lr * iter / warmup_iters
    if iter > max_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=10):
    """Estimate train and val loss"""
    model.eval()
    losses = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        total_loss = 0
        for _ in range(eval_iters):
            x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses

def train(model, train_data, val_data):
    """Training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\n{'='*60}")
    print(f"Training Arianna - {model.count_params():,} parameters")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    start_time = time.time()

    for iter in range(MAX_ITERS):
        # Update learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch and compute loss
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            gap = losses['val'] - losses['train']
            elapsed = time.time() - start_time

            print(f"iter {iter:5d} | loss {losses['train']:.4f}/{losses['val']:.4f} | "
                  f"gap {gap:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                model.save_c_weights('data/arianna_best.bin')
                print(f"         ^ new best! saved to data/arianna_best.bin")

    # Final save
    model.save_c_weights('data/arianna_final.bin')
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to data/arianna_final.bin")
    print(f"{'='*60}")

# ============================================================
# Generation (for testing during training)
# ============================================================

@torch.no_grad()
def generate(model, prompt, max_tokens=100, temperature=0.8):
    """Generate text from prompt"""
    model.eval()
    tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Crop to max_seq_len
        tokens_cond = tokens[:, -MAX_SEQ_LEN:]

        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return ''.join(chr(t) for t in tokens[0].tolist())

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Load corpus
    corpus_path = 'corpus_clean.txt'
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        exit(1)

    train_data, val_data = load_corpus(corpus_path)
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")

    # Create model
    model = Arianna().to(device)
    print(f"Model parameters: {model.count_params():,}")

    # Train
    train(model, train_data, val_data)

    # Test generation
    print("\n--- Test Generation ---")
    print(generate(model, "I am Arianna. ", max_tokens=100, temperature=0.8))
    print("-" * 40)

    print("\nTest C inference with:")
    print("  ./bin/arianna data/arianna_final.bin \"I am Arianna. \" 100 0.8")
