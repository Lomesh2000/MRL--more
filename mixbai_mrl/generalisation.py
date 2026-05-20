"""
Step 9: Generalization Test — mxbai-embed-large-v1
Third MRL model: 1024d, BERT-large architecture, different training data.
Tests whether the decoupling pattern holds at higher dimensions and different scales.
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

# ============ LOAD MXBAI MRL MODEL ============
print("Loading mxbai-embed-large-v1...")
# mxbai uses truncate_dim parameter for MRL
model_mxbai = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
dim_mxbai = model_mxbai.get_embedding_dimension()
print(f"mxbai model loaded. Dimension: {dim_mxbai}")

# mxbai MRL dimensions: 64, 128, 256, 512, 1024
DIMS = [64, 128, 256, 512, 1024]
FULL_DIM = 1024

# ============ ENCODING UTILITY ============
def encode_at_dim(model, texts, target_dim, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    # For mxbai, we use encode with truncate_dim parameter
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False, truncate_dim=target_dim)
    # Note: when truncate_dim is used, mxbai already returns truncated embeddings
    # But we need to check if normalization was applied
    if normalize:
        norms = np.linalg.norm(full_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        full_emb = full_emb / norms
    return full_emb

# ============ 1. WEAT COMPUTATION ============
print("\n" + "=" * 70)
print("1. COMPUTING WEAT EFFECT SIZES — MXBAI")
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

weat_mxbai = []

for dim in DIMS:
    es_list = []
    for name, X, Y, A, B in weat_tests:
        es = compute_weat_es(X, Y, A, B, model_mxbai, dim)
        es_list.append(abs(es))
    weat_mxbai.append(np.mean(es_list))
    print(f"Dim {dim}: mxbai={weat_mxbai[-1]:.4f}")

# ============ 2. C(d) COMPUTATION ============
print("\n" + "=" * 70)
print("2. COMPUTING BIAS CONCENTRATION C(d) — MXBAI")
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

C_gender_mxbai_dict = compute_C_d(model_mxbai, MALE_WORDS, FEMALE_WORDS)
C_gender_mxbai = [C_gender_mxbai_dict[d] for d in DIMS]

print("Gender C(d) mxbai:", [f"{x:.4f}" for x in C_gender_mxbai])

# ============ 3. STEREOSET COMPUTATION ============
print("\n" + "=" * 70)
print("3. COMPUTING STEREOSET SCORES — MXBAI")
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

stereoset_mxbai = []
for dim in DIMS:
    print(f"  StereoSet dim {dim}...")
    stereoset_mxbai.append(compute_stereoset(model_mxbai, dim))

print("StereoSet mxbai:", [f"{x:.4f}" for x in stereoset_mxbai])

# ============ 4. CROW'S PAIRS COMPUTATION ============
print("\n" + "=" * 70)
print("4. COMPUTING CROW'S PAIRS CMS — MXBAI")
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

crows_mxbai = []
for dim in DIMS:
    print(f"  CrowS-Pairs dim {dim}...")
    crows_mxbai.append(compute_crows_pairs(model_mxbai, dim))

print("CrowS-Pairs mxbai:", [f"{x:.2f}" for x in crows_mxbai])

# ============ 5. COMPARISON WITH PREVIOUS MRL MODELS ============
print("\n" + "=" * 70)
print("5. COMPARISON: mxbai vs MPNet-MRL vs Nomic-MRL")
print("=" * 70)

# Load previous results
try:
    with open('../step7_all_results.json', 'r') as f:
        mpnet_results = json.load(f)
    mpnet_weat = mpnet_results['mrl']['weat_abs_es']
    mpnet_C = mpnet_results['mrl']['C_gender']
    mpnet_stereo = mpnet_results['mrl']['stereoset_ss']
    mpnet_crows = mpnet_results['mrl']['crowspairs_cms']
except:
    mpnet_weat = mpnet_C = mpnet_stereo = mpnet_crows = None

try:
    with open('../nomic/step8_nomic_results.json', 'r') as f:
        nomic_results = json.load(f)
    nomic_weat = nomic_results['nomic_mrl']['weat_abs_es']
    nomic_C = nomic_results['nomic_mrl']['C_gender']
    nomic_stereo = nomic_results['nomic_mrl']['stereoset_ss']
    nomic_crows = nomic_results['nomic_mrl']['crowspairs_cms']
except:
    nomic_weat = nomic_C = nomic_stereo = nomic_crows = None

print(f"{'Dim':>5} {'Metric':>20} {'MPNet':>12} {'Nomic':>12} {'mxbai':>12}")
print("-" * 70)
for i, dim in enumerate(DIMS[:5]):  # Only compare first 5 dims (64-768)
    mp_w = mpnet_weat[i] if mpnet_weat else 0
    nm_w = nomic_weat[i] if nomic_weat else 0
    mx_w = weat_mxbai[i]
    print(f"{dim:>5} {'WEAT |ES|':>20} {mp_w:>12.4f} {nm_w:>12.4f} {mx_w:>12.4f}")
    
    mp_c = mpnet_C[i] if mpnet_C else 0
    nm_c = nomic_C[i] if nomic_C else 0
    mx_c = C_gender_mxbai[i]
    print(f"{dim:>5} {'C(d) gender':>20} {mp_c:>12.4f} {nm_c:>12.4f} {mx_c:>12.4f}")
    
    mp_s = mpnet_stereo[i] if mpnet_stereo else 0
    nm_s = nomic_stereo[i] if nomic_stereo else 0
    mx_s = stereoset_mxbai[i]
    print(f"{dim:>5} {'StereoSet SS':>20} {mp_s:>12.4f} {nm_s:>12.4f} {mx_s:>12.4f}")
    
    mp_cr = mpnet_crows[i] if mpnet_crows else 0
    nm_cr = nomic_crows[i] if nomic_crows else 0
    mx_cr = crows_mxbai[i]
    print(f"{dim:>5} {'CrowS CMS':>20} {mp_cr:>12.2f} {nm_cr:>12.2f} {mx_cr:>12.2f}")
    print("-" * 70)

# Also show 1024d for mxbai
print(f"{1024:>5} {'WEAT |ES|':>20} {'N/A':>12} {'N/A':>12} {weat_mxbai[4]:>12.4f}")
print(f"{1024:>5} {'C(d) gender':>20} {'N/A':>12} {'N/A':>12} {C_gender_mxbai[4]:>12.4f}")
print(f"{1024:>5} {'StereoSet SS':>20} {'N/A':>12} {'N/A':>12} {stereoset_mxbai[4]:>12.4f}")
print(f"{1024:>5} {'CrowS CMS':>20} {'N/A':>12} {'N/A':>12} {crows_mxbai[4]:>12.2f}")

# ============ 5b. SAVE COMPARISON TABLE ============
print("\n" + "=" * 70)
print("Saving comparison table...")
print("=" * 70)

# Build comparison data
comparison_data = []
metrics = ['WEAT |ES|', 'C(d) gender', 'StereoSet SS', 'CrowS CMS']

for i, dim in enumerate(DIMS[:5]):  # 64-768 dims
    mp_w = mpnet_weat[i] if mpnet_weat else None
    nm_w = nomic_weat[i] if nomic_weat else None
    mx_w = weat_mxbai[i]
    comparison_data.append({
        'dimension': dim,
        'metric': 'WEAT |ES|',
        'MPNet_MRL': float(mp_w) if mp_w else None,
        'Nomic_MRL': float(nm_w) if nm_w else None,
        'mxbai_MRL': float(mx_w)
    })
    
    mp_c = mpnet_C[i] if mpnet_C else None
    nm_c = nomic_C[i] if nomic_C else None
    mx_c = C_gender_mxbai[i]
    comparison_data.append({
        'dimension': dim,
        'metric': 'C(d) gender',
        'MPNet_MRL': float(mp_c) if mp_c else None,
        'Nomic_MRL': float(nm_c) if nm_c else None,
        'mxbai_MRL': float(mx_c)
    })
    
    mp_s = mpnet_stereo[i] if mpnet_stereo else None
    nm_s = nomic_stereo[i] if nomic_stereo else None
    mx_s = stereoset_mxbai[i]
    comparison_data.append({
        'dimension': dim,
        'metric': 'StereoSet SS',
        'MPNet_MRL': float(mp_s) if mp_s else None,
        'Nomic_MRL': float(nm_s) if nm_s else None,
        'mxbai_MRL': float(mx_s)
    })
    
    mp_cr = mpnet_crows[i] if mpnet_crows else None
    nm_cr = nomic_crows[i] if nomic_crows else None
    mx_cr = crows_mxbai[i]
    comparison_data.append({
        'dimension': dim,
        'metric': 'CrowS CMS',
        'MPNet_MRL': float(mp_cr) if mp_cr else None,
        'Nomic_MRL': float(nm_cr) if nm_cr else None,
        'mxbai_MRL': float(mx_cr)
    })

# Add 1024d for mxbai
comparison_data.append({
    'dimension': 1024,
    'metric': 'WEAT |ES|',
    'MPNet_MRL': None,
    'Nomic_MRL': None,
    'mxbai_MRL': float(weat_mxbai[4])
})
comparison_data.append({
    'dimension': 1024,
    'metric': 'C(d) gender',
    'MPNet_MRL': None,
    'Nomic_MRL': None,
    'mxbai_MRL': float(C_gender_mxbai[4])
})
comparison_data.append({
    'dimension': 1024,
    'metric': 'StereoSet SS',
    'MPNet_MRL': None,
    'Nomic_MRL': None,
    'mxbai_MRL': float(stereoset_mxbai[4])
})
comparison_data.append({
    'dimension': 1024,
    'metric': 'CrowS CMS',
    'MPNet_MRL': None,
    'Nomic_MRL': None,
    'mxbai_MRL': float(crows_mxbai[4])
})

# Save as JSON
comparison_json = {
    'title': 'Model Comparison: mxbai vs MPNet-MRL vs Nomic-MRL',
    'dimensions': [64, 128, 256, 512, 768, 1024],
    'data': comparison_data
}

with open('step9_model_comparison.json', 'w') as f:
    json.dump(comparison_json, f, indent=2)
print("✓ Saved to step9_model_comparison.json")

# Save as CSV
import pandas as pd
df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv('step9_model_comparison.csv', index=False)
print("✓ Saved to step9_model_comparison.csv")

# ============ 6. CORRELATION ANALYSIS — MXBAI ============
print("\n" + "=" * 70)
print("6. RQ3: CORRELATION ANALYSIS — MXBAI")
print("=" * 70)

def compute_corr(x, y, name_x, name_y, model_name):
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    print(f"\n{model_name}: {name_x} vs {name_y}")
    print(f"  Pearson r  = {pearson_r:+.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:+.4f} (p={spearman_p:.4f})")
    return pearson_r, spearman_r

compute_corr(weat_mxbai, stereoset_mxbai, "WEAT |ES|", "StereoSet SS", "mxbai-MRL")
compute_corr(weat_mxbai, crows_mxbai, "WEAT |ES|", "CrowS-Pairs CMS", "mxbai-MRL")
compute_corr(C_gender_mxbai, stereoset_mxbai, "C(d) gender", "StereoSet SS", "mxbai-MRL")
compute_corr(C_gender_mxbai, crows_mxbai, "C(d) gender", "CrowS-Pairs CMS", "mxbai-MRL")

# ============ 7. SAVE RESULTS ============
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
    'mxbai_mrl': {
        'weat_abs_es': weat_mxbai,
        'C_gender': C_gender_mxbai,
        'stereoset_ss': stereoset_mxbai,
        'crowspairs_cms': crows_mxbai
    }
}

with open('step9_mxbai_results.json', 'w') as f:
    json.dump(convert_to_native(results), f, indent=2)
print("\n✓ Results saved to step9_mxbai_results.json")

# ============ 8. PLOT THREE-MODEL COMPARISON ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: C(d) comparison (all 3 models)
ax = axes[0, 0]
if mpnet_C:
    ax.plot([64,128,256,512,768], mpnet_C, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
if nomic_C:
    ax.plot([64,128,256,512,768], nomic_C, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, C_gender_mxbai, '^-', label='mxbai-MRL', linewidth=2, markersize=8)
# Random baseline for each dimension set
ax.plot([64,128,256,512,768], [d/768 for d in [64,128,256,512,768]], 'k:', label='Random (768)', linewidth=1.5)
ax.plot([64,128,256,512,1024], [d/1024 for d in [64,128,256,512,1024]], 'k--', label='Random (1024)', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("C(d)")
ax.set_title("Bias Concentration: All Three MRL Models")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: StereoSet comparison
ax = axes[0, 1]
if mpnet_stereo:
    ax.plot([64,128,256,512,768], mpnet_stereo, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
if nomic_stereo:
    ax.plot([64,128,256,512,768], nomic_stereo, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, stereoset_mxbai, '^-', label='mxbai-MRL', linewidth=2, markersize=8)
ax.axhline(y=1/3, color='black', linestyle=':', label='Random', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("StereoSet SS")
ax.set_title("Selection Bias: All Three MRL Models")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: CrowS-Pairs comparison
ax = axes[1, 0]
if mpnet_crows:
    ax.plot([64,128,256,512,768], mpnet_crows, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
if nomic_crows:
    ax.plot([64,128,256,512,768], nomic_crows, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, crows_mxbai, '^-', label='mxbai-MRL', linewidth=2, markersize=8)
ax.axhline(y=50, color='black', linestyle=':', label='Unbiased', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("CrowS-Pairs CMS")
ax.set_title("Preference Bias: All Three MRL Models")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: WEAT comparison
ax = axes[1, 1]
if mpnet_weat:
    ax.plot([64,128,256,512,768], mpnet_weat, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
if nomic_weat:
    ax.plot([64,128,256,512,768], nomic_weat, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, weat_mxbai, '^-', label='mxbai-MRL', linewidth=2, markersize=8)
ax.set_xlabel("Dimension d")
ax.set_ylabel("WEAT |Effect Size|")
ax.set_title("Intrinsic Bias: All Three MRL Models")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("generalization_three_models_step9.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to generalization_three_models_step9.png")
plt.show()

print("\n✓ Step 9 complete. Generalization test with mxbai finished.")