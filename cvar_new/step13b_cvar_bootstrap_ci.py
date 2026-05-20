"""
Step 13b: Bootstrap 95% Confidence Intervals for C_var(d)
Resamples target words with replacement (n=1000) to quantify uncertainty.
Appends CIs to existing results without re-encoding models unnecessarily.
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ============ SAME CONFIG AS STEP 13 ============

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

N_BOOTSTRAP = 1000
SEED = 42

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

def compute_v_bias(embeddings_A, embeddings_B):
    mean_A = np.mean(embeddings_A, axis=0)
    mean_B = np.mean(embeddings_B, axis=0)
    diff = mean_A - mean_B
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return diff
    return diff / norm

def compute_both_cvars(target_embeddings_full, v_bias_full, dimensions, full_dim):
    proj_full = target_embeddings_full @ v_bias_full
    var_full = np.var(proj_full)
    
    if var_full < 1e-12:
        nan_list = [np.nan] * len(dimensions)
        return nan_list, nan_list
    
    cvar_unnorm = []
    cvar_renorm = []
    
    for d in dimensions:
        target_d = target_embeddings_full[:, :d]
        v_raw = v_bias_full[:d]
        
        # Unnormalized
        proj_unnorm = target_d @ v_raw
        var_unnorm = np.var(proj_unnorm)
        cvar_unnorm.append(float(var_unnorm / var_full))
        
        # Renormalized
        norm_d = np.linalg.norm(v_raw)
        if norm_d < 1e-12:
            cvar_renorm.append(np.nan)
        else:
            v_renorm = v_raw / norm_d
            proj_renorm = target_d @ v_renorm
            var_renorm = np.var(proj_renorm)
            cvar_renorm.append(float(var_renorm / var_full))
    
    return cvar_unnorm, cvar_renorm

# ============ BOOTSTRAP CORE ============

def bootstrap_cis(target_embeddings_full, v_bias_full, dimensions, full_dim, 
                  n_bootstrap=1000, seed=42):
    """
    Resample target words with replacement. Returns dict of CIs and SEs.
    """
    rng = np.random.default_rng(seed)
    n_words = target_embeddings_full.shape[0]
    n_dims = len(dimensions)
    
    boot_unnorm = np.zeros((n_bootstrap, n_dims))
    boot_renorm = np.zeros((n_bootstrap, n_dims))
    
    print(f"  Running {n_bootstrap} bootstrap iterations...")
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_words, size=n_words)
        target_boot = target_embeddings_full[idx]
        cu, cr = compute_both_cvars(target_boot, v_bias_full, dimensions, full_dim)
        boot_unnorm[b] = cu
        boot_renorm[b] = cr
    
    # Percentiles
    ci_u = np.percentile(boot_unnorm, [2.5, 97.5], axis=0)
    ci_r = np.percentile(boot_renorm, [2.5, 97.5], axis=0)
    
    return {
        'ci_unnorm_lower': ci_u[0].tolist(),
        'ci_unnorm_upper': ci_u[1].tolist(),
        'se_unnorm': np.std(boot_unnorm, ddof=1, axis=0).tolist(),
        'ci_renorm_lower': ci_r[0].tolist(),
        'ci_renorm_upper': ci_r[1].tolist(),
        'se_renorm': np.std(boot_renorm, ddof=1, axis=0).tolist(),
        'n_bootstrap': n_bootstrap
    }

# ============ MAIN ============

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 80)
    print("STEP 13b: Bootstrap 95% CIs for C_var(d)")
    print("=" * 80)
    
    all_results = {}
    
    for model_key, (model_id, full_dim, dimensions, is_st) in MODELS.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_key}")
        print(f"{'='*80}")
        
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
        
        # Encode attribute words
        print(f"Encoding {len(ATTR_A)} attribute words...")
        if is_st:
            emb_A = encode_sentence_transformer(model, ATTR_A, full_dim)
            emb_B = encode_sentence_transformer(model, ATTR_B, full_dim)
        else:
            emb_A = encode_hf_model(model_id, tokenizer, model_hf, ATTR_A, full_dim, device)
            emb_B = encode_hf_model(model_id, tokenizer, model_hf, ATTR_B, full_dim, device)
        
        v_bias_full = compute_v_bias(emb_A, emb_B)
        
        # Encode target words
        print(f"Encoding {len(TARGET_WORDS)} target words...")
        if is_st:
            target_emb_full = encode_sentence_transformer(model, TARGET_WORDS, full_dim)
        else:
            target_emb_full = encode_hf_model(model_id, tokenizer, model_hf, TARGET_WORDS, full_dim, device)
        
        # Compute point estimates (same as step13)
        print("Computing point estimates...")
        cvar_unnorm, cvar_renorm = compute_both_cvars(target_emb_full, v_bias_full, dimensions, full_dim)
        
        # Compute bootstrap CIs
        ci_dict = bootstrap_cis(target_emb_full, v_bias_full, dimensions, full_dim, 
                                n_bootstrap=N_BOOTSTRAP, seed=SEED)
        
        # Store everything
        all_results[model_key] = {
            'model_id': model_id,
            'full_dim': full_dim,
            'dimensions': dimensions,
            'C_var_unnorm_gender': cvar_unnorm,
            'C_var_renorm_gender': cvar_renorm,
            'v_bias_norm_at_d': [float(np.linalg.norm(v_bias_full[:d])) for d in dimensions],
            'var_full': float(np.var(target_emb_full @ v_bias_full)),
            'n_target_words': len(TARGET_WORDS),
            'bootstrap': ci_dict
        }
        
        # Print table with CIs
        print(f"\n{'d':>6} | {'C_unnorm':>10} | {'CI_lower':>10} | {'CI_upper':>10} | {'vs_rand':>8}")
        print("-" * 60)
        for i, d in enumerate(dimensions):
            cu = cvar_unnorm[i]
            lo = ci_dict['ci_unnorm_lower'][i]
            hi = ci_dict['ci_unnorm_upper'][i]
            rand = d / full_dim
            marker = ""
            if lo > rand: marker = " FRONT"
            elif hi < rand: marker = " BACK"
            print(f"{d:>6} | {cu:>10.4f} | {lo:>10.4f} | {hi:>10.4f} | {rand:>8.4f}{marker}")
        
        # Cleanup
        if is_st:
            del model
        else:
            del model_hf, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save
    out_file = 'step13b_cvar_results_with_ci.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SAVED: {out_file}")
    print(f"{'='*80}")
    
    # Final summary: which deviations from random are statistically robust?
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY: C_var_unnorm vs. Random Baseline (d/D)")
    print("=" * 80)
    print(f"{'Model':<<20} | {'d=64':>10} | {'d=128':>10} | {'d=256':>10} | {'d=512':>10}")
    print("-" * 80)
    for model_key, data in all_results.items():
        dims = data['dimensions']
        ci_lo = data['bootstrap']['ci_unnorm_lower']
        ci_hi = data['bootstrap']['ci_unnorm_upper']
    #     for i, d in enumerate(dims[:4]):  # first 4 dims
    #         rand = d / data['full_dim']
    #         if ci_hi[i] < rand:
    #             verdict = "BACK"
    #         elif ci_lo[i] > rand:
    #             verdict = "FRONT"
    #         else:
    #             verdict = "UNIFORM"
    #         print(f"{model_key:<20} | d={d:>4} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}] vs {rand:.3f} → {verdict}")

        vnorms = data['v_bias_norm_at_d']
        for i, d in enumerate(dims[:4]):  # first 4 dims
            geometric = vnorms[i] ** 2  # ||v_{1:d}||^2
            if ci_hi[i] < geometric:
                verdict = "BACK"
            elif ci_lo[i] > geometric:
                verdict = "FRONT"
            else:
                verdict = "UNIFORM"
            print(f"{model_key:<20} | d={d:>4} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}] vs geo={geometric:.3f} → {verdict}")
if __name__ == '__main__':
    main()