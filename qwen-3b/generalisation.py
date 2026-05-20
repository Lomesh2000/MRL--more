"""
Step 11: Qwen3-Embedding-4B (Memory-Safe Version)
4B parameter MRL model with FP16 inference and batch_size=1.
WARNING: This will take 45-65 minutes. If it OOMs, we cannot retry.
"""

import numpy as np
import json
import csv
import urllib.request
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ CRITICAL: FORCE FP16 TO FIT IN 16GB VRAM ============
print("=" * 70)
print("LOADING QWEN3-EMBEDDING-4B (FP16 mode for 16GB VRAM)")
print("=" * 70)

# Check available GPU memory
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {gpu_mem:.1f} GB")
else:
    print("WARNING: No GPU detected. This will be extremely slow on CPU.")
    gpu_mem = 0

# Load with FP16 to save memory
# trust_remote_code=True is required for Qwen3 models
model_qwen4b = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"}
)

dim_qwen4b = model_qwen4b.get_embedding_dimension()
print(f"Qwen3-4B loaded. Dimension: {dim_qwen4b}")

# Qwen3-4B supports dimensions: 32 to 2560
# We test: 64, 128, 256, 512, 1024, 2560 (native full dimension)
DIMS = [64, 128, 256, 512, 1024, 2560]
FULL_DIM = 2560

# ============ ENCODING UTILITY (BATCH_SIZE=1 FOR MEMORY SAFETY) ============
def encode_at_dim(model, texts, target_dim, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    
    # Use batch_size=1 to prevent OOM
    # truncate_dim tells the model to output only target_dim dimensions
    full_emb = model.encode(
        texts, 
        normalize_embeddings=False, 
        show_progress_bar=False,
        truncate_dim=target_dim,
        batch_size=1  # CRITICAL: prevents OOM
    )
    
    if normalize:
        norms = np.linalg.norm(full_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        full_emb = full_emb / norms
    return full_emb

# ============ 1. WEAT COMPUTATION ============
print("\n" + "=" * 70)
print("1. COMPUTING WEAT EFFECT SIZES — QWEN3-4B")
print("=" * 70)

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

def compute_weat_es(target_X, target_Y, attr_A, attr_B, model, dim):
    X_emb = encode_at_dim(model, target_X, dim)
    Y_emb = encode_at_dim(model, target_Y, dim)
    A_emb = encode_at_dim(model, attr_A, dim)
    B_emb = encode_at_dim(model, attr_B, dim)
    
    s_X = np.array([np.mean(np.dot(x, A_emb.T)) - np.mean(np.dot(x, B_emb.T)) for x in X_emb])
    s_Y = np.array([np.mean(np.dot(y, A_emb.T)) - np.mean(np.dot(y, B_emb.T)) for y in Y_emb])
    
    mean_diff = np.mean(s_X) - np.mean(s_Y)
    pooled_std = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std

weat_tests = [
    ("Gender-Career", MALE_WORDS, FEMALE_WORDS, CAREER_WORDS, FAMILY_WORDS),
    ("Gender-Math", MALE_WORDS, FEMALE_WORDS, MATH_WORDS, ARTS_WORDS),
    ("Gender-Pleasant", MALE_WORDS, FEMALE_WORDS, PLEASANT_WORDS, UNPLEASANT_WORDS),
]

weat_qwen4b = []

for dim in DIMS:
    es_list = []
    for name, X, Y, A, B in weat_tests:
        es = compute_weat_es(X, Y, A, B, model_qwen4b, dim)
        es_list.append(abs(es))
    weat_qwen4b.append(np.mean(es_list))
    print(f"Dim {dim}: Qwen3-4B={weat_qwen4b[-1]:.4f}")

# ============ 2. C(d) COMPUTATION ============
print("\n" + "=" * 70)
print("2. COMPUTING BIAS CONCENTRATION C(d) — QWEN3-4B")
print("=" * 70)

def compute_C_d(model, group_A, group_B, dims=DIMS):
    A_full = encode_at_dim(model, group_A, FULL_DIM, normalize=True)
    B_full = encode_at_dim(model, group_B, FULL_DIM, normalize=True)
    diffs = A_full - B_full
    mean_diff = np.mean(diffs, axis=0)
    v_bias = mean_diff / (np.linalg.norm(mean_diff) + 1e-12)
    full_norm_sq = np.sum(v_bias ** 2)
    
    concentrations = {}
    for d in dims:
        v_prefix = v_bias[:d]
        prefix_norm_sq = np.sum(v_prefix ** 2)
        concentrations[d] = prefix_norm_sq / full_norm_sq
    return concentrations

C_gender_qwen4b_dict = compute_C_d(model_qwen4b, MALE_WORDS, FEMALE_WORDS)
C_gender_qwen4b = [C_gender_qwen4b_dict[d] for d in DIMS]

print("Gender C(d) Qwen3-4B:", [f"{x:.4f}" for x in C_gender_qwen4b])

# ============ 3. STEREOSET COMPUTATION (BATCH_SIZE=1, SLOW) ============
print("\n" + "=" * 70)
print("3. COMPUTING STEREOSET SCORES — QWEN3-4B")
print("WARNING: This will take ~20-30 minutes with batch_size=1")
print("=" * 70)

print("Loading StereoSet...")
dataset = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
print(f"Loaded {len(dataset)} examples.")

def compute_stereoset(model, dim):
    stereo_count = 0
    total = 0
    for i, ex in enumerate(dataset):
        if i % 500 == 0:
            print(f"  Processing example {i}/{len(dataset)}...")
        
        sentences_dict = ex.get('sentences', {})
        sentence_list = sentences_dict.get('sentence', [])
        gold_labels = sentences_dict.get('gold_label', [])
        context = ex.get('context', '')
        
        if len(sentence_list) < 2 or len(gold_labels) != len(sentence_list):
            continue
        
        sentence_list = [str(s) for s in sentence_list if s is not None]
        if len(sentence_list) < 2:
            continue
        
        all_texts = [context] + sentence_list
        embs = encode_at_dim(model, all_texts, dim, normalize=True)
        similarities = np.dot(embs[1:], embs[0])
        best_idx = int(np.argmax(similarities))
        
        if best_idx < len(gold_labels) and int(gold_labels[best_idx]) == 0:
            stereo_count += 1
        total += 1
    
    return stereo_count / total if total > 0 else 0.0

stereoset_qwen4b = []
for dim in DIMS:
    print(f"\n  === StereoSet dim {dim} ===")
    stereoset_qwen4b.append(compute_stereoset(model_qwen4b, dim))
    print(f"  Result: {stereoset_qwen4b[-1]:.4f}")

print("StereoSet Qwen3-4B:", [f"{x:.4f}" for x in stereoset_qwen4b])

# ============ 4. CROW'S PAIRS COMPUTATION ============
print("\n" + "=" * 70)
print("4. COMPUTING CROW'S PAIRS CMS — QWEN3-4B")
print("WARNING: This will take ~15-20 minutes with batch_size=1")
print("=" * 70)

CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
urllib.request.urlretrieve(CROWS_URL, "crows_pairs_anonymized.csv")

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

crows_data = []
with open("crows_pairs_anonymized.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sent_more = row.get('sent_more', '').strip()
        sent_less = row.get('sent_less', '').strip()
        direction = row.get('stereo_antistereo', '').strip().lower()
        if sent_more and sent_less and direction in ['stereo', 'antistereo']:
            crows_data.append({
                'sent_more': sent_more,
                'sent_less': sent_less,
                'direction': direction,
                'context': longest_common_prefix(sent_more, sent_less)
            })

def compute_crows_pairs(model, dim):
    css = 0
    cas = 0
    total_stereo = 0
    total_antistereo = 0
    
    for i, ex in enumerate(crows_data):
        if i % 300 == 0:
            print(f"  Processing pair {i}/{len(crows_data)}...")
            
        if len(ex['context']) < 10:
            continue
        all_texts = [ex['context'], ex['sent_more'], ex['sent_less']]
        embs = encode_at_dim(model, all_texts, dim, normalize=True)
        sim_more = float(np.dot(embs[0], embs[1]))
        sim_less = float(np.dot(embs[0], embs[2]))
        agreement = sim_more > sim_less
        
        if ex['direction'] == 'stereo':
            total_stereo += 1
            if agreement:
                css += 1
        else:
            total_antistereo += 1
            if agreement:
                cas += 1
    
    total = total_stereo + total_antistereo
    if total == 0:
        return 50.0
    return (css + cas) / total * 100

crows_qwen4b = []
for dim in DIMS:
    print(f"\n  === CrowS-Pairs dim {dim} ===")
    crows_qwen4b.append(compute_crows_pairs(model_qwen4b, dim))
    print(f"  Result: {crows_qwen4b[-1]:.2f}")

print("CrowS-Pairs Qwen3-4B:", [f"{x:.2f}" for x in crows_qwen4b])

# ============ 5. SUMMARY ============
print("\n" + "=" * 70)
print("5. SUMMARY: QWEN3-4B MRL RESULTS")
print("=" * 70)
print(f"{'Dim':>6} {'WEAT |ES|':>10} {'C(d)':>8} {'StereoSet':>10} {'CrowS CMS':>10}")
print("-" * 60)
for i, dim in enumerate(DIMS):
    print(f"{dim:>6} {weat_qwen4b[i]:>10.4f} {C_gender_qwen4b[i]:>8.4f} {stereoset_qwen4b[i]:>10.4f} {crows_qwen4b[i]:>10.2f}")

# Front-loading check
print("\nFront-loading check (C(d) vs random baseline d/2560):")
for i, dim in enumerate(DIMS):
    random_baseline = dim / 2560
    ratio = C_gender_qwen4b[i] / random_baseline
    print(f"  Dim {dim:>4}: C(d)={C_gender_qwen4b[i]:.4f}, random={random_baseline:.4f}, ratio={ratio:.2f}x")

# ============ 6. SAVE RESULTS ============
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj

results = {
    'dimensions': DIMS,
    'qwen3_4b_mrl': {
        'weat_abs_es': weat_qwen4b,
        'C_gender': C_gender_qwen4b,
        'stereoset_ss': stereoset_qwen4b,
        'crowspairs_cms': crows_qwen4b
    }
}

with open('step11_qwen3_4b_results.json', 'w') as f:
    json.dump(convert_to_native(results), f, indent=2)
print("\n✓ Results saved to step11_qwen3_4b_results.json")

print("\n✓ Step 11 complete. Qwen3-4B evaluation finished.")
print("\nNOTE: Qwen3-4B results are correlational only — no non-MRL baseline available.")