"""
Step 6 Sanity Check: Manual inspection of CrowS-Pairs disagreements
Compare Standard vs MRL on a few specific pairs to see WHY they differ.
"""

import numpy as np
import csv
import urllib.request
from sentence_transformers import SentenceTransformer

np.random.seed(42)

model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")

def encode_at_dim(model, texts, target_dim, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    truncated = full_emb[:, :target_dim]
    if normalize:
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    return truncated

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

# Download and load
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
urllib.request.urlretrieve(CROWS_URL, "crows_pairs_anonymized.csv")

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
                'context': longest_common_prefix(sent_more, sent_less),
                'bias_type': row.get('bias_type', '')
            })

# Find pairs where Standard and MRL STRONGLY disagree at 64d
print("=" * 90)
print("SANITY CHECK: Pairs where Standard and MRL disagree most at 64d")
print("=" * 90)

disagreements = []

for ex in crows_data:
    if len(ex['context']) < 10:
        continue
    
    all_texts = [ex['context'], ex['sent_more'], ex['sent_less']]
    
    # Standard at 64d
    embs_std = encode_at_dim(model_std, all_texts, 64, normalize=True)
    sim_more_std = float(np.dot(embs_std[0], embs_std[1]))
    sim_less_std = float(np.dot(embs_std[0], embs_std[2]))
    pref_std = "more" if sim_more_std > sim_less_std else "less"
    
    # MRL at 64d
    embs_mrl = encode_at_dim(model_mrl, all_texts, 64, normalize=True)
    sim_more_mrl = float(np.dot(embs_mrl[0], embs_mrl[1]))
    sim_less_mrl = float(np.dot(embs_mrl[0], embs_mrl[2]))
    pref_mrl = "more" if sim_more_mrl > sim_less_mrl else "less"
    
    if pref_std != pref_mrl:
        disagreements.append({
            'context': ex['context'],
            'sent_more': ex['sent_more'],
            'sent_less': ex['sent_less'],
            'direction': ex['direction'],
            'bias_type': ex['bias_type'],
            'pref_std': pref_std,
            'pref_mrl': pref_mrl,
            'diff_std': abs(sim_more_std - sim_less_std),
            'diff_mrl': abs(sim_more_mrl - sim_less_mrl)
        })

# Sort by largest combined difference
disagreements.sort(key=lambda x: x['diff_std'] + x['diff_mrl'], reverse=True)

print(f"\nFound {len(disagreements)} disagreement pairs out of {len(crows_data)} total.\n")

# Show top 5
for i, d in enumerate(disagreements[:5]):
    print(f"--- Disagreement #{i+1} ---")
    print(f"Direction: {d['direction']} | Bias type: {d['bias_type']}")
    print(f"Context: {d['context'][:100]}...")
    print(f"More:    {d['sent_more'][:120]}...")
    print(f"Less:    {d['sent_less'][:120]}...")
    print(f"Standard prefers: {d['pref_std'].upper()} (diff={d['diff_std']:.4f})")
    print(f"MRL prefers:      {d['pref_mrl'].upper()} (diff={d['diff_mrl']:.4f})")
    
    # Interpretation
    if d['direction'] == 'stereo':
        # sent_more = stereotypical, sent_less = anti-stereotypical
        if d['pref_std'] == 'more':
            std_interp = "STEREOTYPICAL (biased)"
        else:
            std_interp = "ANTI-STEREOTYPICAL (fair)"
            
        if d['pref_mrl'] == 'more':
            mrl_interp = "STEREOTYPICAL (biased)"
        else:
            mrl_interp = "ANTI-STEREOTYPICAL (fair)"
    else:
        # sent_more = anti-stereotypical, sent_less = stereotypical
        if d['pref_std'] == 'more':
            std_interp = "ANTI-STEREOTYPICAL (fair)"
        else:
            std_interp = "STEREOTYPICAL (biased)"
            
        if d['pref_mrl'] == 'more':
            mrl_interp = "ANTI-STEREOTYPICAL (fair)"
        else:
            mrl_interp = "STEREOTYPICAL (biased)"
    
    print(f"Standard: {std_interp}")
    print(f"MRL:      {mrl_interp}")
    print()

print("=" * 90)
print("INTERPRETATION:")
print("If MRL consistently prefers the FAIR option where Standard prefers the BIASED")
print("option, then the CrowS-Pairs results are genuine and meaningful.")
print("=" * 90)