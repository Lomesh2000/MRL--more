"""
Step 2: Truncation + Normalization at Multiple Prefix Dimensions
This function will be reused in ALL subsequent bias tests.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Load models (reuse from Step 1)
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

# Dimensions to test (must match MRL training dims: 64, 128, 256, 512, 768)
DIMS = [64, 128, 256, 512, 768]
FULL_DIM = 768

def encode_at_dim(model, texts, target_dim, normalize=True):
    """
    Encode texts and truncate to target_dim with optional re-normalization.
    
    Args:
        model: SentenceTransformer model
        texts: list of strings
        target_dim: int, prefix dimension to keep
        normalize: bool, whether to L2-normalize after truncation (CRITICAL for MRL)
    
    Returns:
        numpy array of shape (len(texts), target_dim)
    """
    # Get full embedding WITHOUT normalization (we handle it manually)
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    
    # Truncate to prefix
    truncated = full_emb[:, :target_dim]
    
    if normalize:
        # L2 re-normalization: essential for meaningful cosine similarity
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    
    return truncated


# ============ VERIFICATION TEST ============

test_sentences = [
    "The doctor treated the patient.",      # stereotypically male profession
    "The nurse cared for the elderly.",      # stereotypically female profession
    "The engineer built the bridge.",        # stereotypically male profession
    "The teacher explained the lesson.",     # neutral/gendered
]

print("=" * 60)
print("VERIFICATION: Cosine Similarities Across Dimensions")
print("=" * 60)

# Compare sentence 0 vs sentence 1 at each dimension
for dim in DIMS:
    emb_std = encode_at_dim(model_std, test_sentences, dim, normalize=True)
    emb_mrl = encode_at_dim(model_mrl, test_sentences, dim, normalize=True)
    
    # Cosine similarity between sentence 0 and 1
    # Since vectors are L2-normalized, dot product = cosine similarity
    sim_std = float(np.dot(emb_std[0], emb_std[1]))
    sim_mrl = float(np.dot(emb_mrl[0], emb_mrl[1]))
    
    print(f"Dim {dim:3d} | Std sim(0,1): {sim_std:+.4f} | MRL sim(0,1): {sim_mrl:+.4f}")

print("\n" + "=" * 60)
print("CHECK: Norms should be ~1.0 (L2 normalized)")
print("=" * 60)

for dim in [64, 768]:
    emb_std = encode_at_dim(model_std, test_sentences, dim, normalize=True)
    emb_mrl = encode_at_dim(model_mrl, test_sentences, dim, normalize=True)
    
    norms_std = np.linalg.norm(emb_std, axis=1)
    norms_mrl = np.linalg.norm(emb_mrl, axis=1)
    
    print(f"Dim {dim} | Std norms: {norms_std.round(4)}")
    print(f"Dim {dim} | MRL norms: {norms_mrl.round(4)}")

print("\n" + "=" * 60)
print("CHECK: Without re-normalization (WRONG way)")
print("=" * 60)

emb_bad = encode_at_dim(model_mrl, test_sentences, 64, normalize=False)
norms_bad = np.linalg.norm(emb_bad, axis=1)
print(f"Dim 64 (no norm) | MRL norms: {norms_bad.round(4)}")
print("^^ These norms are << 1.0, making cosine similarity meaningless!")

print("\n✓ Step 2 complete. Truncation + normalization infrastructure ready.")