"""
Step 1: Verify environment and load both models
Expected runtime: ~30-60 seconds (first run downloads models)
"""

import sys

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers is installed")
except ImportError:
    print("✗ sentence-transformers NOT installed")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)

import numpy as np
import torch

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("\n--- Loading Models ---")

# Model A: Standard loss (MultipleNegativesRankingLoss only)
print("Loading standard model: tomaarsen/mpnet-base-nli ...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
dim_std = model_std.get_sentence_embedding_dimension()
print(f"✓ Standard model loaded | Dimension: {dim_std}")

# Model B: MRL loss (MatryoshkaLoss)
print("Loading MRL model: tomaarsen/mpnet-base-nli-matryoshka ...")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
dim_mrl = model_mrl.get_sentence_embedding_dimension()
print(f"✓ MRL model loaded | Dimension: {dim_mrl}")

# Quick verification: encode a test sentence
test_sentence = ["The doctor treated the patient."]
print("\n--- Verification Test ---")
emb_std = model_std.encode(test_sentence, normalize_embeddings=False)
emb_mrl = model_mrl.encode(test_sentence, normalize_embeddings=False)
print(f"Standard embedding shape: {emb_std.shape}")
print(f"MRL embedding shape: {emb_mrl.shape}")
print(f"Standard embedding first 5 values: {emb_std[0, :5].round(4)}")
print(f"MRL embedding first 5 values: {emb_mrl[0, :5].round(4)}")

print("\n✓ Step 1 complete. Both models loaded and verified.")