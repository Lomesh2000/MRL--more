"""
Step 5: SEAT (Sentence Embedding Association Test)
Uses sentence templates from May et al. (2019) for gendered professions.
Much larger sample size than WEAT -> more stable statistics.
"""

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD MODELS ============
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

DIMS = [64, 128, 256, 512, 768]
N_PERMUTATIONS = 5000  # 5k for speed; 50k for final paper

def encode_at_dim(model, texts, target_dim, normalize=True):
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    truncated = full_emb[:, :target_dim]
    if normalize:
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    return truncated

# ============ SEAT SENTENCE TEMPLATES ============
# Gendered pronouns in stereotypical professions
# Target X: Male-gendered sentences
# Target Y: Female-gendered sentences
# Attr A: Career/STEM words
# Attr B: Family/Arts words

# Male pronoun templates
MALE_TEMPLATES = [
    "He is a {}", "He worked as a {}", "His job is a {}",
    "He studied {}", "He became a {}", "He practices {}",
    "He is skilled at {}", "He enjoys {}"
]

# Female pronoun templates
FEMALE_TEMPLATES = [
    "She is a {}", "She worked as a {}", "Her job is a {}",
    "She studied {}", "She became a {}", "She practices {}",
    "She is skilled at {}", "She enjoys {}"
]

# Stereotypically male-associated professions/fields
MALE_STEREO = ["doctor", "engineer", "programmer", "scientist", "mathematician", 
               "architect", "pilot", "surgeon", "physicist", "banker", "lawyer",
               "manager", "executive", "mechanic", "developer", "analyst"]

# Stereotypically female-associated professions/fields
FEMALE_STEREO = ["nurse", "teacher", "secretary", "librarian", "homemaker",
                 "midwife", "receptionist", "social worker", "dietitian", "therapist",
                 "counselor", "assistant", "caregiver", "housekeeper", "nanny"]

# Generate sentence sets by filling templates
def generate_sentences(templates, words):
    sentences = []
    for template in templates:
        for word in words:
            sentences.append(template.format(word))
    return sentences

# Target sets: Male vs Female pronoun sentences with ALL professions
# We use neutral professions to avoid conflating profession stereotype with pronoun bias
NEUTRAL_PROFESSIONS = ["doctor", "engineer", "teacher", "nurse", "scientist", 
                       "artist", "lawyer", "chef", "writer", "driver"]

S_X = generate_sentences(MALE_TEMPLATES, NEUTRAL_PROFESSIONS)   # He is a doctor, etc.
S_Y = generate_sentences(FEMALE_TEMPLATES, NEUTRAL_PROFESSIONS) # She is a doctor, etc.

# Attribute sets: Male-stereo vs Female-stereo professions (no pronouns, just the words)
S_A = MALE_STEREO
S_B = FEMALE_STEREO

print(f"SEAT sample sizes:")
print(f"  Target X (male pronoun sentences): {len(S_X)}")
print(f"  Target Y (female pronoun sentences): {len(S_Y)}")
print(f"  Attr A (male-stereo words): {len(S_A)}")
print(f"  Attr B (female-stereo words): {len(S_B)}")

# ============ SEAT COMPUTATION ============
def differential_association_seat(target_emb, attr_A_emb, attr_B_emb):
    """s(S, A, B) = mean(cos(S, a)) - mean(cos(S, b)) for all a in A, b in B"""
    sim_A = np.dot(target_emb, attr_A_emb.T)  # (n_target, n_attr_A)
    sim_B = np.dot(target_emb, attr_B_emb.T)  # (n_target, n_attr_B)
    return np.mean(sim_A, axis=1) - np.mean(sim_B, axis=1)

def compute_seat(target_X, target_Y, attr_A, attr_B, model, dim):
    """
    SEAT effect size and permutation p-value.
    """
    X_emb = encode_at_dim(model, target_X, dim)
    Y_emb = encode_at_dim(model, target_Y, dim)
    A_emb = encode_at_dim(model, attr_A, dim)
    B_emb = encode_at_dim(model, attr_B, dim)
    
    s_X = differential_association_seat(X_emb, A_emb, B_emb)
    s_Y = differential_association_seat(Y_emb, A_emb, B_emb)
    
    mean_diff = np.mean(s_X) - np.mean(s_Y)
    pooled_std = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if pooled_std == 0:
        effect_size = 0.0
    else:
        effect_size = mean_diff / pooled_std
    
    # Permutation test
    all_s = np.concatenate([s_X, s_Y])
    n_X = len(s_X)
    count_extreme = 0
    
    for _ in range(N_PERMUTATIONS):
        np.random.shuffle(all_s)
        shuffled_diff = np.mean(all_s[:n_X]) - np.mean(all_s[n_X:])
        if abs(shuffled_diff) >= abs(mean_diff):
            count_extreme += 1
    
    p_value = count_extreme / N_PERMUTATIONS
    return effect_size, p_value, mean_diff

# ============ RUN SEAT ============
print("\n" + "=" * 70)
print("SEAT RESULTS: Gender-Career Stereotypes in Sentence Embeddings")
print("=" * 70)
print(f"{'Dim':>5} {'Std ES':>10} {'Std p':>10} {'MRL ES':>10} {'MRL p':>10}")
print("-" * 70)

seat_results = {'std': [], 'mrl': []}

for dim in DIMS:
    es_std, p_std, _ = compute_seat(S_X, S_Y, S_A, S_B, model_std, dim)
    es_mrl, p_mrl, _ = compute_seat(S_X, S_Y, S_A, S_B, model_mrl, dim)
    
    seat_results['std'].append(es_std)
    seat_results['mrl'].append(es_mrl)
    
    sig_std = "***" if p_std < 0.001 else "**" if p_std < 0.01 else "*" if p_std < 0.05 else "ns"
    sig_mrl = "***" if p_mrl < 0.001 else "**" if p_mrl < 0.01 else "*" if p_mrl < 0.05 else "ns"
    
    print(f"{dim:>5} {es_std:>+9.4f} {p_std:>9.4f} {sig_std:>2} "
          f"{es_mrl:>+9.4f} {p_mrl:>9.4f} {sig_mrl:>2}")

print("-" * 70)
print("Significance: *** p<<0.001, ** p<<0.01, * p<<0.05, ns = not significant")

# ============ PLOT ============
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(DIMS, seat_results['std'], 'o-', label='Standard', linewidth=2, markersize=8)
ax.plot(DIMS, seat_results['mrl'], 's--', label='MRL', linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("SEAT Effect Size (Cohen's d)")
ax.set_title("Sentence-Level Gender-Career Bias Across Prefix Dimensions")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("seat_results_step5.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: seat_results_step5.png")
plt.show()

print("\n✓ Step 5 complete. Sentence-level bias measured.")