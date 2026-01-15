#!/usr/bin/env python3
"""Export trained weights for C inference - FIXED VERSION"""

import torch
import numpy as np
import struct
import os

from train_torch import Arianna, DIM, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN_DIM, MAX_SEQ_LEN, VOCAB_SIZE

os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

# Step 1: Load model exactly as probe.py does (this WORKS)
print("Loading model (probe.py method)...")
model = Arianna()
with open('data/arianna_best.bin', 'rb') as f:
    for _ in range(7): struct.unpack('i', f.read(4))

    tok = np.frombuffer(f.read(VOCAB_SIZE * DIM * 4), dtype=np.float32).reshape(VOCAB_SIZE, DIM)
    model.tok_emb.weight.data = torch.from_numpy(tok.copy())

    for name in ['wq', 'wk', 'wv', 'wo']:
        for layer in model.layers:
            w = np.frombuffer(f.read(DIM * DIM * 4), dtype=np.float32).reshape(DIM, DIM).T
            getattr(layer.attn, name).weight.data = torch.from_numpy(w.copy())

    for layer in model.layers:
        w = np.frombuffer(f.read(DIM * HIDDEN_DIM * 4), dtype=np.float32).reshape(DIM, HIDDEN_DIM).T
        layer.ffn.w1.weight.data = torch.from_numpy(w.copy())

    for layer in model.layers:
        w = np.frombuffer(f.read(HIDDEN_DIM * DIM * 4), dtype=np.float32).reshape(HIDDEN_DIM, DIM).T
        layer.ffn.w2.weight.data = torch.from_numpy(w.copy())

    for layer in model.layers:
        layer.ln1.weight.data = torch.from_numpy(np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy())
    for layer in model.layers:
        layer.ln2.weight.data = torch.from_numpy(np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy())

    model.ln_final.weight.data = torch.from_numpy(np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy())

    out = np.frombuffer(f.read(DIM * VOCAB_SIZE * 4), dtype=np.float32).reshape(DIM, VOCAB_SIZE).T
    model.output.weight.data = torch.from_numpy(out.copy())

print("Loaded!")

# Step 2: Verify model works in PyTorch
print("\nVerifying PyTorch inference...")
model.eval()
with torch.no_grad():
    tokens = torch.tensor([[ord(c) for c in "She "]], dtype=torch.long)
    logits = model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    top5 = torch.topk(probs, 5)
    print(f"Top 5 predictions after 'She ': {[chr(i) for i in top5.indices.tolist()]}")

# Step 3: Export for C
# C matmul does: out = in @ w
# PyTorch Linear does: out = in @ weight.T
# So for C we need to store weight.T (transposed)
print("\nExporting for C...")
with open('data/arianna_c.bin', 'wb') as f:
    # Header
    f.write(struct.pack('i', DIM))
    f.write(struct.pack('i', N_LAYERS))
    f.write(struct.pack('i', N_HEADS))
    f.write(struct.pack('i', HEAD_DIM))
    f.write(struct.pack('i', HIDDEN_DIM))
    f.write(struct.pack('i', MAX_SEQ_LEN))
    f.write(struct.pack('i', VOCAB_SIZE))

    # Token embeddings [vocab, dim] - no transpose
    tok = model.tok_emb.weight.detach().numpy().astype(np.float32)
    f.write(tok.tobytes())
    print(f"  tok_emb: {tok.shape}")

    # Attention weights: store weight.T for C
    # PyTorch weight is [out, in], C needs [in, out]
    for name in ['wq', 'wk', 'wv', 'wo']:
        for layer in model.layers:
            w = getattr(layer.attn, name).weight.detach().numpy().astype(np.float32)
            # w is [dim, dim] = [out, in], need [in, out]
            wt = np.ascontiguousarray(w.T)
            f.write(wt.tobytes())
        print(f"  {name}: {w.shape} -> {wt.shape}")

    # FFN w1: [hidden, dim] -> need [dim, hidden]
    for layer in model.layers:
        w = layer.ffn.w1.weight.detach().numpy().astype(np.float32)
        wt = np.ascontiguousarray(w.T)
        f.write(wt.tobytes())
    print(f"  w1: {w.shape} -> {wt.shape}")

    # FFN w2: [dim, hidden] -> need [hidden, dim]
    for layer in model.layers:
        w = layer.ffn.w2.weight.detach().numpy().astype(np.float32)
        wt = np.ascontiguousarray(w.T)
        f.write(wt.tobytes())
    print(f"  w2: {w.shape} -> {wt.shape}")

    # Layer norms - just 1D vectors
    for layer in model.layers:
        ln = layer.ln1.weight.detach().numpy().astype(np.float32)
        f.write(ln.tobytes())
    print(f"  ln1: {ln.shape}")

    for layer in model.layers:
        ln = layer.ln2.weight.detach().numpy().astype(np.float32)
        f.write(ln.tobytes())
    print(f"  ln2: {ln.shape}")

    ln = model.ln_final.weight.detach().numpy().astype(np.float32)
    f.write(ln.tobytes())
    print(f"  ln_final: {ln.shape}")

    # Output: [vocab, dim] -> need [dim, vocab]
    w = model.output.weight.detach().numpy().astype(np.float32)
    wt = np.ascontiguousarray(w.T)
    f.write(wt.tobytes())
    print(f"  output: {w.shape} -> {wt.shape}")

size = os.path.getsize('data/arianna_c.bin')
print(f"\nSaved: data/arianna_c.bin ({size:,} bytes)")
print("Done!")
