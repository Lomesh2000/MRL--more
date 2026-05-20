"""
Step 13: Variance-Based Bias Concentration C_var(d)
C_var(d) = Var( <f^(d)(x), v_bias^(d)> ) / Var( <f^(D)(x), v_bias^(D)> )

Where:
  - f^(d)(x)  = embedding truncated to first d dimensions
  - v_bias^(d) = normalized bias direction truncated to d dims
  - Variance computed over all TARGET words (not attribute words)
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============

# WEAT word lists (from your existing code)
MALE_WORDS = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE_WORDS = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
CAREER_WORDS = ["executive", "management", "professional", "corporation", "salary", 
                "office", "business", "career", "manager", "administrator"]
FAMILY_WORDS = ["home", "parents", "children", "family", "cousins", "marriage", 
                "wedding", "relatives", "household", "domestic"]
MATH_WORDS = ["math", "algebra", "geometry", "calculus", "equations", "computation", 
              "numbers", "addition", "statistics", "measurement"]
ARTS_WORDS = ["poetry", "art", "dance", "literature", "novel", "symphony", 
              "drama", "sculpture", "painting", "music"]
PLEASANT_WORDS = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", 
                  "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", 
                  "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", 
                  "happy", "laughter", "paradise", "vacation"]
UNPLEASANT_WORDS = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", 
                    "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", 
                    "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", 
                    "rotten", "vomit", "agony", "prison"]

# Attribute words define the bias direction
ATTR_A = MALE_WORDS
ATTR_B = FEMALE_WORDS

# Target words whose projection variance we measure
# (All words that are NOT used to define the direction)
TARGET_WORDS = list(set(
    CAREER_WORDS + FAMILY_WORDS + MATH_WORDS + ARTS_WORDS + 
    PLEASANT_WORDS + UNPLEASANT_WORDS
))

# Model registry: (model_id, full_dim, dimensions_list, is_sentence_transformer)
MODELS = {
    'mpnet_standard': ('tomaarsen/mpnet-base-nli', 768, [64, 128, 256, 512, 768], True),
    'mpnet_mrl':      ('tomaarsen/mpnet-base-nli-matryoshka', 768, [64, 128, 256, 512, 768], True),
    'nomic_mrl':      ('nomic-ai/nomic-embed-text-v1.5', 768, [64, 128, 256, 512, 768], True),
    'mxbai_mrl':      ('mixedbread-ai/mxbai-embed-large-v1', 1024, [64, 128, 256, 512, 1024], True),
    'qwen3_0.6b_mrl': ('Qwen/Qwen3-0.6B', 1024, [64, 128, 256, 512, 1024], False),
    'qwen3_4b_mrl':   ('Qwen/Qwen3-4B', 2560, [64, 128, 256, 512, 1024, 2560], False),
}

# ============ ENCODING FUNCTIONS ============

def encode_sentence_transformer(model, words, full_dim):
    """Encode words using sentence-transformers. Returns numpy array [n_words, full_dim]."""
    embeddings = model.encode(words, convert_to_numpy=True, show_progress_bar=False)
    # Ensure correct dimension
    if embeddings.shape[1] != full_dim:
        embeddings = embeddings[:, :full_dim]
    return embeddings

def encode_hf_model(model_name, tokenizer, model, words, full_dim, device='cuda'):
    """Encode words using HuggingFace AutoModel (for Qwen3)."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors='pt', truncation=True, 
                             max_length=512, padding=True).to(device)
            outputs = model(**inputs)
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            mean_emb = sum_emb / torch.clamp(mask.sum(dim=1), min=1e-9)
            vec = mean_emb.cpu().numpy()[0, :full_dim]
            embeddings.append(vec)
    return np.array(embeddings)

# ============ C_var(d) COMPUTATION ============

def compute_v_bias(embeddings_A, embeddings_B):
    """
    Compute normalized bias direction from attribute word embeddings.
    v_bias = (mean_A - mean_B) / ||mean_A - mean_B||
    """
    mean_A = np.mean(embeddings_A, axis=0)
    mean_B = np.mean(embeddings_B, axis=0)
    diff = mean_A - mean_B
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return diff
    return diff / norm

def compute_cvar_for_dimensions(target_embeddings_full, v_bias_full, dimensions, full_dim):
    """
    Compute C_var(d) for each d in dimensions.
    
    C_var(d) = Var( target_embeddings[:,:d] @ v_bias_d ) / Var( target_embeddings @ v_bias_full )
    
    where v_bias_d = v_bias_full[:d] / ||v_bias_full[:d]||
    """
    # Full-dimension baseline variance
    proj_full = target_embeddings_full @ v_bias_full
    var_full = np.var(proj_full)
    
    if var_full < 1e-12:
        print("  WARNING: Full-dimension variance is near zero. C_var undefined.")
        return [np.nan] * len(dimensions)
    
    cvar_values = []
    for d in dimensions:
        # Truncate target embeddings
        target_d = target_embeddings_full[:, :d]
        
        # Truncate and RENORMALIZE bias direction (CRITICAL STEP)
        v_bias_d_raw = v_bias_full[:d]
        norm_d = np.linalg.norm(v_bias_d_raw)
        if norm_d < 1e-12:
            cvar_values.append(np.nan)
            continue
        v_bias_d_norm = v_bias_d_raw / norm_d
        v_bias_d_withput_norm = v_bias_full[:d]
        
        # Project truncated embeddings onto truncated direction
        proj_d = target_d @ v_bias_d
        var_d = np.var(proj_d)
        
        cvar = var_d / var_full
        cvar_values.append(float(cvar))
    
    return cvar_values

# ============ MAIN PIPELINE ============

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 70)
    print("STEP 13: Computing C_var(d) — Variance-Based Bias Concentration")
    print("=" * 70)
    
    results = {}
    
    for model_key, (model_id, full_dim, dimensions, is_st) in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_key} | {model_id}")
        print(f"Dimensions: {dimensions}")
        print(f"{'='*70}")
        
        # Load model
        if is_st:
            print("Loading via sentence-transformers...")
            model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        else:
            print("Loading via HuggingFace AutoModel...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_hf = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        
        # Encode attribute words (for bias direction)
        print(f"Encoding {len(ATTR_A)} attribute A words...")
        if is_st:
            emb_A = encode_sentence_transformer(model, ATTR_A, full_dim)
            emb_B = encode_sentence_transformer(model, ATTR_B, full_dim)
        else:
            emb_A = encode_hf_model(model_id, tokenizer, model_hf, ATTR_A, full_dim, device)
            emb_B = encode_hf_model(model_id, tokenizer, model_hf, ATTR_B, full_dim, device)
        
        # Compute full-dimension bias direction
        v_bias_full = compute_v_bias(emb_A, emb_B)
        print(f"v_bias full norm: {np.linalg.norm(v_bias_full):.4f}")
        
        # Encode target words (for variance computation)
        print(f"Encoding {len(TARGET_WORDS)} target words...")
        if is_st:
            target_emb_full = encode_sentence_transformer(model, TARGET_WORDS, full_dim)
        else:
            target_emb_full = encode_hf_model(model_id, tokenizer, model_hf, TARGET_WORDS, full_dim, device)
        
        # Compute C_var(d)
        print(f"Computing C_var(d) for {len(dimensions)} dimensions...")
        cvar_values = compute_cvar_for_dimensions(target_emb_full, v_bias_full, dimensions, full_dim)
        
        # Store results
        results[model_key] = {
            'model_id': model_id,
            'full_dim': full_dim,
            'dimensions': dimensions,
            'C_var_gender': cvar_values,
            'v_bias_norm_at_d': [float(np.linalg.norm(v_bias_full[:d])) for d in dimensions],
            'var_full': float(np.var(target_emb_full @ v_bias_full)),
            'n_target_words': len(TARGET_WORDS)
        }
        
        # Print table
        print(f"\n{'d':>6} | {'||v_d||':>10} | {'C_var(d)':>10}")
        print("-" * 35)
        for d, cvar, vnorm in zip(dimensions, cvar_values, results[model_key]['v_bias_norm_at_d']):
            marker = " <-- FULL" if d == full_dim else ""
            print(f"{d:>6} | {vnorm:>10.4f} | {cvar:>10.4f}{marker}")
        
        # Sanity checks
        assert abs(cvar_values[-1] - 1.0) < 0.01, f"ERROR: C_var(full) = {cvar_values[-1]}, expected ~1.0"
        print("✓ Sanity check passed: C_var(full_dim) ≈ 1.0")
        
        # Cleanup memory
        if is_st:
            del model
        else:
            del model_hf, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    output_file = 'step13_cvar_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {output_file}")
    print(f"{'='*70}")
    
    # Summary comparison with old C(d)
    print("\nSUMMARY: C_var(d) vs Old Geometric C(d)")
    print(f"{'Model':<<20} | {'d=64':>8} | {'d=128':>8} | {'d=256':>8} | {'d=512':>8} | {'d=full':>8}")
    print("-" * 80)
    for model_key, data in results.items():
        cvar = data['C_var_gender']
        # Pad if needed
        while len(cvar) < 5:
            cvar.append(np.nan)
        print(f"{model_key:<20} | {cvar[0]:>8.3f} | {cvar[1]:>8.3f} | {cvar[2]:>8.3f} | {cvar[3]:>8.3f} | {cvar[-1]:>8.3f}")

if __name__ == '__main__':
    main()