"""
Step 13: Variance-Based Bias Concentration — BOTH VERSIONS
  C_var_unnorm(d): Var(<f^(d), v_bias^(D)[1:d]>) / Var(<f^(D), v_bias^(D)>)   [NO renormalize]
  C_var_renorm(d): Var(<f^(d), v_bias^(d)>) / Var(<f^(D), v_bias^(D)>)         [renormalized]

This lets us compare the two definitions and pick the correct one.
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============

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

ATTR_A = MALE_WORDS
ATTR_B = FEMALE_WORDS

TARGET_WORDS = list(set(
    CAREER_WORDS + FAMILY_WORDS + MATH_WORDS + ARTS_WORDS + 
    PLEASANT_WORDS + UNPLEASANT_WORDS
))

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
    embeddings = model.encode(words, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.shape[1] != full_dim:
        embeddings = embeddings[:, :full_dim]
    return embeddings

def encode_hf_model(model_name, tokenizer, model, words, full_dim, device='cuda'):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors='pt', truncation=True, 
                             max_length=512, padding=True).to(device)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            mean_emb = sum_emb / torch.clamp(mask.sum(dim=1), min=1e-9)
            vec = mean_emb.cpu().numpy()[0, :full_dim]
            embeddings.append(vec)
    return np.array(embeddings)

# ============ BIAS DIRECTION ============

def compute_v_bias(embeddings_A, embeddings_B):
    mean_A = np.mean(embeddings_A, axis=0)
    mean_B = np.mean(embeddings_B, axis=0)
    diff = mean_A - mean_B
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return diff
    return diff / norm

# ============ C_var(d) — BOTH VERSIONS ============

def compute_both_cvars(target_embeddings_full, v_bias_full, dimensions, full_dim):
    """
    Computes BOTH:
      C_var_unnorm(d): NO renormalization of truncated direction
      C_var_renorm(d): WITH renormalization of truncated direction
    """
    proj_full = target_embeddings_full @ v_bias_full
    var_full = np.var(proj_full)
    
    if var_full < 1e-12:
        print("  WARNING: Full-dimension variance is near zero.")
        nan_list = [np.nan] * len(dimensions)
        return nan_list, nan_list
    
    cvar_unnorm = []
    cvar_renorm = []
    
    for d in dimensions:
        target_d = target_embeddings_full[:, :d]
        v_raw = v_bias_full[:d]
        
        # --- VERSION 1: NO renormalization (correct concentration measure) ---
        proj_unnorm = target_d @ v_raw
        var_unnorm = np.var(proj_unnorm)
        cvar_unnorm.append(float(var_unnorm / var_full))
        
        # --- VERSION 2: WITH renormalization (discriminability measure) ---
        norm_d = np.linalg.norm(v_raw)
        if norm_d < 1e-12:
            cvar_renorm.append(np.nan)
        else:
            v_renorm = v_raw / norm_d
            proj_renorm = target_d @ v_renorm
            var_renorm = np.var(proj_renorm)
            cvar_renorm.append(float(var_renorm / var_full))
    
    return cvar_unnorm, cvar_renorm

# ============ MAIN ============

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 80)
    print("STEP 13: C_var(d) — BOTH Unnormalized AND Renormalized Versions")
    print("=" * 80)
    
    results = {}
    
    for model_key, (model_id, full_dim, dimensions, is_st) in MODELS.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_key}")
        print(f"ID:    {model_id}")
        print(f"Dims:  {dimensions}")
        print(f"{'='*80}")
        
        # Load
        if is_st:
            print("Loading via sentence-transformers...")
            model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        else:
            print("Loading via HuggingFace AutoModel...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_hf = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        
        # Encode attribute words
        print(f"Encoding {len(ATTR_A)} attribute A words...")
        if is_st:
            emb_A = encode_sentence_transformer(model, ATTR_A, full_dim)
            emb_B = encode_sentence_transformer(model, ATTR_B, full_dim)
        else:
            emb_A = encode_hf_model(model_id, tokenizer, model_hf, ATTR_A, full_dim, device)
            emb_B = encode_hf_model(model_id, tokenizer, model_hf, ATTR_B, full_dim, device)
        
        v_bias_full = compute_v_bias(emb_A, emb_B)
        print(f"v_bias full norm: {np.linalg.norm(v_bias_full):.6f}")
        
        # Encode target words
        print(f"Encoding {len(TARGET_WORDS)} target words...")
        if is_st:
            target_emb_full = encode_sentence_transformer(model, TARGET_WORDS, full_dim)
        else:
            target_emb_full = encode_hf_model(model_id, tokenizer, model_hf, TARGET_WORDS, full_dim, device)
        
        # Compute both C_var versions
        print(f"Computing C_var(d) for {len(dimensions)} dimensions...")
        cvar_unnorm, cvar_renorm = compute_both_cvars(target_emb_full, v_bias_full, dimensions, full_dim)
        
        # Store
        var_full_val = float(np.var(target_emb_full @ v_bias_full))
        results[model_key] = {
            'model_id': model_id,
            'full_dim': full_dim,
            'dimensions': dimensions,
            'C_var_unnorm_gender': cvar_unnorm,
            'C_var_renorm_gender': cvar_renorm,
            'v_bias_norm_at_d': [float(np.linalg.norm(v_bias_full[:d])) for d in dimensions],
            'var_full': var_full_val,
            'n_target_words': len(TARGET_WORDS)
        }
        
        # Print comparison table
        print(f"\n{'d':>6} | {'||v_d||':>10} | {'C_unnorm':>10} | {'C_renorm':>10} | {'d/D':>8}")
        print("-" * 55)
        for d, cu, cr, vnorm in zip(dimensions, cvar_unnorm, cvar_renorm, results[model_key]['v_bias_norm_at_d']):
            marker = " <-- FULL" if d == full_dim else ""
            print(f"{d:>6} | {vnorm:>10.4f} | {cu:>10.4f} | {cr:>10.4f} | {d/full_dim:>8.4f}{marker}")
        
        # Sanity checks
        assert abs(cvar_unnorm[-1] - 1.0) < 0.01, f"ERROR: C_unnorm(full) = {cvar_unnorm[-1]}"
        assert abs(cvar_renorm[-1] - 1.0) < 0.01, f"ERROR: C_renorm(full) = {cvar_renorm[-1]}"
        print("✓ Sanity checks passed: both versions ≈ 1.0 at full dim")
        
        # Cleanup
        if is_st:
            del model
        else:
            del model_hf, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save
    output_file = 'step13_cvar_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {output_file}")
    print(f"{'='*80}")
    
    # Final summary table
    print("\n" + "=" * 80)
    print("SUMMARY: C_var_unnorm(d)  [NO renormalize — concentration measure]")
    print("=" * 80)
    hdr = f"{'Model':<<20} | {'d=64':>8} | {'d=128':>8} | {'d=256':>8} | {'d=512':>8} | {'full':>8}"
    print(hdr)
    print("-" * len(hdr))
    for model_key, data in results.items():
        cu = data['C_var_unnorm_gender']
        while len(cu) < 5: cu.append(np.nan)
        print(f"{model_key:<20} | {cu[0]:>8.3f} | {cu[1]:>8.3f} | {cu[2]:>8.3f} | {cu[3]:>8.3f} | {cu[-1]:>8.3f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: C_var_renorm(d)  [renormalized — discriminability measure]")
    print("=" * 80)
    print(hdr)
    print("-" * len(hdr))
    for model_key, data in results.items():
        cr = data['C_var_renorm_gender']
        while len(cr) < 5: cr.append(np.nan)
        print(f"{model_key:<20} | {cr[0]:>8.3f} | {cr[1]:>8.3f} | {cr[2]:>8.3f} | {cr[3]:>8.3f} | {cr[-1]:>8.3f}")

if __name__ == '__main__':
    main()