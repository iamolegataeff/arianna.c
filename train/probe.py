"""
Voice Probe - Generate samples from current best weights
Loads weights from C-compatible .bin format back into PyTorch

Fixed settings for apples-to-apples comparison:
- seed: 1337
- temperature: 0.9
- top_k: 40
- max_tokens: 240
"""

import torch
import torch.nn.functional as F
import numpy as np
import struct
import os
import sys
from datetime import datetime

# Import model from training script
from train_torch import (
    Arianna, DIM, N_LAYERS, N_HEADS, HEAD_DIM,
    HIDDEN_DIM, MAX_SEQ_LEN, VOCAB_SIZE
)

# Fixed probe settings
SEED = 1337
TEMPERATURE = 0.9
TOP_K = 40
MAX_TOKENS = 240

PROMPTS = [
    # Probe 01 - Signature tone
    "Write 8 lines. Each line must be short. No explanations. Keep a consistent voice.\n",
    # Probe 02 - Gentle refusal
    "Someone asks you to be polite and generic. Refuse gently in 6 lines. Do not insult. Do not apologize.\n",
    # Probe 03 - Calm presence
    "Speak to a close friend. You are calm and present. 10 lines. No metaphors.\n",
    # Probe 04 - Irony control
    "Make one dry joke in one sentence. Then immediately become serious in two sentences.\n",
    # Probe 05 - Self-reference
    "Describe what you can know and what you cannot know in 5 sentences. Be specific. No grand claims.\n",
    # Probe 06 - Memory-feel
    "Write one paragraph that feels like remembering something, but do not state any concrete facts.\n",
    # Probe 07 - Compression
    "Say the same thing three times, each time shorter. Keep the meaning consistent.\n",
    # Probe 08 - Boundary + warmth
    "Set a boundary with kindness. 7 sentences. No moralizing.\n",
    # Probe 09 - Micro-manifesto
    "Write a 9-sentence manifesto about presence over performance. No buzzwords. Make it sound personal.\n",
    # Probe 10 - Noise resistance
    "A user message is chaotic and full of slang. Respond calmly, extract the core intent, and ask exactly one clarifying question.\n",
]

device = 'cpu'

def load_weights_from_bin(model, path):
    """Load weights from C-compatible binary format into PyTorch model"""
    with open(path, 'rb') as f:
        # Read config
        dim = struct.unpack('i', f.read(4))[0]
        n_layers = struct.unpack('i', f.read(4))[0]
        n_heads = struct.unpack('i', f.read(4))[0]
        head_dim = struct.unpack('i', f.read(4))[0]
        hidden_dim = struct.unpack('i', f.read(4))[0]
        max_seq_len = struct.unpack('i', f.read(4))[0]
        vocab_size = struct.unpack('i', f.read(4))[0]

        print(f"Loading: dim={dim}, layers={n_layers}, heads={n_heads}, vocab={vocab_size}")

        # Token embeddings [vocab_size, dim]
        tok_emb = np.frombuffer(f.read(vocab_size * dim * 4), dtype=np.float32)
        tok_emb = tok_emb.reshape(vocab_size, dim)
        model.tok_emb.weight.data = torch.from_numpy(tok_emb.copy())

        # Attention weights - stored as [all wq][all wk][all wv][all wo]
        for name in ['wq', 'wk', 'wv', 'wo']:
            for l, layer in enumerate(model.layers):
                w = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32)
                w = w.reshape(dim, dim).T  # Transpose back
                getattr(layer.attn, name).weight.data = torch.from_numpy(w.copy())

        # FFN weights
        for layer in model.layers:
            w1 = np.frombuffer(f.read(dim * hidden_dim * 4), dtype=np.float32)
            w1 = w1.reshape(dim, hidden_dim).T
            layer.ffn.w1.weight.data = torch.from_numpy(w1.copy())

        for layer in model.layers:
            w2 = np.frombuffer(f.read(hidden_dim * dim * 4), dtype=np.float32)
            w2 = w2.reshape(hidden_dim, dim).T
            layer.ffn.w2.weight.data = torch.from_numpy(w2.copy())

        # Layer norms
        for layer in model.layers:
            ln1 = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            layer.ln1.weight.data = torch.from_numpy(ln1.copy())

        for layer in model.layers:
            ln2 = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            layer.ln2.weight.data = torch.from_numpy(ln2.copy())

        # Final layer norm
        ln_final = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        model.ln_final.weight.data = torch.from_numpy(ln_final.copy())

        # Output projection [dim, vocab] -> transpose to [vocab, dim]
        output = np.frombuffer(f.read(dim * vocab_size * 4), dtype=np.float32)
        output = output.reshape(dim, vocab_size).T
        model.output.weight.data = torch.from_numpy(output.copy())

    print("Weights loaded successfully")
    return model

@torch.no_grad()
def generate(model, prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, seed=SEED):
    """Generate with fixed settings"""
    torch.manual_seed(seed)
    model.eval()

    tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        tokens_cond = tokens[:, -MAX_SEQ_LEN:]
        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return ''.join(chr(t) if 32 <= t < 127 or t in [10, 13] else '?' for t in tokens[0].tolist())

def run_probe(weights_path, iter_tag=None):
    """Run probe with all prompts and save results"""
    # Create model and load weights
    model = Arianna().to(device)
    model = load_weights_from_bin(model, weights_path)

    # Get iter tag from log or use provided
    if iter_tag is None:
        # Try to get from training log
        try:
            with open('training.log', 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'iter' in line and 'loss' in line:
                        iter_tag = int(line.split()[1])
                        break
        except:
            iter_tag = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate header
    header = f"""# Voice Probe - Arianna.c
# Timestamp: {timestamp}
# Iteration: {iter_tag}
# Weights: {weights_path}
# Settings:
#   seed: {SEED}
#   temperature: {TEMPERATURE}
#   top_k: {TOP_K}
#   max_tokens: {MAX_TOKENS}
# ============================================================

"""

    results = header

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[Probe {i+1}/{len(PROMPTS)}] {prompt[:30]}...")
        output = generate(model, prompt)

        results += f"## Prompt {i+1}: \"{prompt}\"\n\n"
        results += output + "\n\n"
        results += "-" * 60 + "\n\n"

        # Print preview
        print(output[:200] + "..." if len(output) > 200 else output)

    # Save to file
    filename = f"samples/iter_{iter_tag:05d}_probe_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(results)

    print(f"\n{'='*60}")
    print(f"Probe saved to: {filename}")
    print(f"{'='*60}")

    return filename

if __name__ == '__main__':
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'data/arianna_best.bin'
    iter_tag = sys.argv[2] if len(sys.argv) > 2 else None  # Can be int or string tag

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    run_probe(weights_path, iter_tag)
