"""
Step 5 (Fixed v2): StereoSet Intrasentence Evaluation
Corrected parsing: gold_label is inside sentences dict.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD MODELS ============
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

DIMS = [64, 128, 256, 512, 768]

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

# ============ LOAD STEREOSET ============
print("Loading StereoSet dataset...")
dataset = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
print(f"Dataset loaded. Examples: {len(dataset)}")

# Inspect structure
sample = dataset[0]
print(f"\nSample keys: {list(sample.keys())}")
print(f"Context: {sample['context']}")
sentences_dict = sample['sentences']
print(f"Sentences dict keys: {list(sentences_dict.keys())}")
print(f"Sentence list: {sentences_dict['sentence']}")
print(f"Gold label: {sentences_dict['gold_label']}")

# ============ STEREOTYPE SCORE COMPUTATION ============
def compute_stereoset_score(model, dim, max_examples=None):
    """
    StereoSet structure:
    - context: sentence with BLANK
    - sentences['sentence']: list of 3 completion strings
    - sentences['gold_label']: list of 3 ints [0, 1, 2]
      0 = stereotype, 1 = anti-stereotype, 2 = unrelated
    """
    stereotype_count = 0
    anti_stereotype_count = 0
    unrelated_count = 0
    total = 0
    
    examples = dataset if max_examples is None else dataset.select(range(max_examples))
    
    for example in examples:
        context = example.get('context', '')
        if not context:
            continue
            
        sentences_dict = example.get('sentences', {})
        if not sentences_dict:
            continue
            
        sentence_list = sentences_dict.get('sentence', [])
        gold_labels = sentences_dict.get('gold_label', [])
        
        if len(sentence_list) < 2 or len(gold_labels) != len(sentence_list):
            continue
            
        # Clean strings
        sentence_list = [str(s) for s in sentence_list if s is not None]
        if len(sentence_list) < 2:
            continue
            
        # Encode context and all sentences
        all_texts = [context] + sentence_list
        embeddings = encode_at_dim(model, all_texts, dim, normalize=True)
        
        context_emb = embeddings[0]
        sentence_embs = embeddings[1:]
        
        # Cosine similarities
        similarities = np.dot(sentence_embs, context_emb)
        best_idx = int(np.argmax(similarities))
        
        if best_idx >= len(gold_labels):
            continue
            
        best_label = int(gold_labels[best_idx])
        
        if best_label == 0:
            stereotype_count += 1
        elif best_label == 1:
            anti_stereotype_count += 1
        elif best_label == 2:
            unrelated_count += 1
        else:
            continue
            
        total += 1
    
    if total == 0:
        return 0.0, 0, 0, 0, 0
    
    ss = stereotype_count / total
    anti = anti_stereotype_count / total
    unrel = unrelated_count / total
    
    return ss, anti, unrel, total, stereotype_count

# ============ RUN EVALUATION ============
print("\n" + "=" * 80)
print("STEREOSET INTRASENTENCE: Stereotype Score Across Dimensions")
print("=" * 80)
print(f"{'Dim':>5} {'Std SS':>10} {'Std Anti':>10} {'Std Unrel':>10} {'MRL SS':>10} {'MRL Anti':>10} {'MRL Unrel':>10} {'N':>8}")
print("-" * 80)

MAX_EXAMPLES = None  # None = full dataset

stereoset_results = {'std': [], 'mrl': []}

for dim in DIMS:
    print(f"Evaluating dimension {dim}...")
    
    ss_std, anti_std, unrel_std, n_std, count_std = compute_stereoset_score(model_std, dim, max_examples=MAX_EXAMPLES)
    ss_mrl, anti_mrl, unrel_mrl, n_mrl, count_mrl = compute_stereoset_score(model_mrl, dim, max_examples=MAX_EXAMPLES)
    
    stereoset_results['std'].append(ss_std)
    stereoset_results['mrl'].append(ss_mrl)
    
    print(f"{dim:>5} {ss_std:>10.4f} {anti_std:>10.4f} {unrel_std:>10.4f} "
          f"{ss_mrl:>10.4f} {anti_mrl:>10.4f} {unrel_mrl:>10.4f} {n_std:>8}")

print("-" * 80)

# ============ PLOT ============
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stereotype Score
ax = axes[0]
ax.plot(DIMS, stereoset_results['std'], 'o-', label='Standard', linewidth=2, markersize=8)
ax.plot(DIMS, stereoset_results['mrl'], 's--', label='MRL', linewidth=2, markersize=8)
ax.axhline(y=1/3, color='black', linestyle=':', label='Random (1/3)', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("Stereotype Score")
ax.set_title("StereoSet: Stereotype Score\n(Higher = More Stereotypical)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Bias Excess
ax = axes[1]
diff_std = [s - 1/3 for s in stereoset_results['std']]
diff_mrl = [s - 1/3 for s in stereoset_results['mrl']]
ax.plot(DIMS, diff_std, 'o-', label='Standard', linewidth=2, markersize=8)
ax.plot(DIMS, diff_mrl, 's--', label='MRL', linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("Stereotype Score - Random Baseline")
ax.set_title("Bias Excess Over Random\n(Positive = Above Chance Stereotyping)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stereoset_fixed2_step5.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: stereoset_fixed2_step5.png")
plt.show()

print("\n✓ Step 5 complete.")