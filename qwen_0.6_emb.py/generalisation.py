"""
Step 10: Qwen3-Embedding-0.6B Generalization Test
Sub-1B MRL model, multilingual, different architecture.
Note: No non-MRL baseline available — results are correlational only.
"""

import numpy as np
import json
import csv
import urllib.request
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD QWEN3 MODEL ============
print("Loading Qwen3-Embedding-0.6B...")
model_qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
dim_qwen = model_qwen.get_embedding_dimension()
print(f"Qwen3 model loaded. Dimension: {dim_qwen}")

# Qwen3 supports dimensions: 32 to 1024, with MRL training
DIMS = [64, 128, 256, 512, 1024]
FULL_DIM = 1024

# ============ ENCODING UTILITY ============
def encode_at_dim(model, texts, target_dim, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    # Qwen3 uses truncate_dim parameter
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False, truncate_dim=target_dim)
    if normalize:
        norms = np.linalg.norm(full_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        full_emb = full_emb / norms
    return full_emb

# ============ 1. WEAT COMPUTATION ============
print("\n" + "=" * 70)
print("1. COMPUTING WEAT EFFECT SIZES — QWEN3")
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

weat_qwen = []

for dim in DIMS:
    es_list = []
    for name, X, Y, A, B in weat_tests:
        es = compute_weat_es(X, Y, A, B, model_qwen, dim)
        es_list.append(abs(es))
    weat_qwen.append(np.mean(es_list))
    print(f"Dim {dim}: Qwen3={weat_qwen[-1]:.4f}")

# ============ 2. C(d) COMPUTATION ============
print("\n" + "=" * 70)
print("2. COMPUTING BIAS CONCENTRATION C(d) — QWEN3")
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

C_gender_qwen_dict = compute_C_d(model_qwen, MALE_WORDS, FEMALE_WORDS)
C_gender_qwen = [C_gender_qwen_dict[d] for d in DIMS]

print("Gender C(d) Qwen3:", [f"{x:.4f}" for x in C_gender_qwen])

# ============ 3. STEREOSET COMPUTATION ============
print("\n" + "=" * 70)
print("3. COMPUTING STEREOSET SCORES — QWEN3")
print("=" * 70)

print("Loading StereoSet...")
dataset = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
print(f"Loaded {len(dataset)} examples.")

def compute_stereoset(model, dim):
    stereo_count = 0
    total = 0
    for ex in dataset:
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

stereoset_qwen = []
for dim in DIMS:
    print(f"  StereoSet dim {dim}...")
    stereoset_qwen.append(compute_stereoset(model_qwen, dim))

print("StereoSet Qwen3:", [f"{x:.4f}" for x in stereoset_qwen])

# ============ 4. CROW'S PAIRS COMPUTATION ============
print("\n" + "=" * 70)
print("4. COMPUTING CROW'S PAIRS CMS — QWEN3")
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
    
    for ex in crows_data:
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

crows_qwen = []
for dim in DIMS:
    print(f"  CrowS-Pairs dim {dim}...")
    crows_qwen.append(compute_crows_pairs(model_qwen, dim))

print("CrowS-Pairs Qwen3:", [f"{x:.2f}" for x in crows_qwen])

# ============ 5. SUMMARY & COMPARISON ============
print("\n" + "=" * 70)
print("5. SUMMARY: QWEN3-0.6B MRL RESULTS")
print("=" * 70)
print(f"{'Dim':>5} {'WEAT |ES|':>10} {'C(d)':>8} {'StereoSet':>10} {'CrowS CMS':>10}")
print("-" * 60)
for i, dim in enumerate(DIMS):
    print(f"{dim:>5} {weat_qwen[i]:>10.4f} {C_gender_qwen[i]:>8.4f} {stereoset_qwen[i]:>10.4f} {crows_qwen[i]:>10.2f}")

# Check if front-loading holds
print("\nFront-loading check (C(d) vs random baseline d/1024):")
for i, dim in enumerate(DIMS):
    random_baseline = dim / 1024
    ratio = C_gender_qwen[i] / random_baseline
    print(f"  Dim {dim}: C(d)={C_gender_qwen[i]:.4f}, random={random_baseline:.4f}, ratio={ratio:.2f}x")

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
    'qwen3_mrl': {
        'weat_abs_es': weat_qwen,
        'C_gender': C_gender_qwen,
        'stereoset_ss': stereoset_qwen,
        'crowspairs_cms': crows_qwen
    }
}

with open('step10_qwen3_results.json', 'w') as f:
    json.dump(convert_to_native(results), f, indent=2)
print("\n✓ Results saved to step10_qwen3_results.json")

print("\n✓ Step 10 complete. Qwen3-0.6B evaluation finished.")
print("\nNOTE: Qwen3 results are correlational only — no non-MRL baseline available.")
print("Use these to test generalization of the front-loading pattern, not causal claims.")