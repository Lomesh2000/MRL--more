"""
Step 4: WEAT Visualization + PCA Bias Subspace Concentration
Computes C(d) = ||v_bias[1:d]||^2 / ||v_bias||^2
If C(d) >> d/D for small d, bias is concentrated in early dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD MODELS ============
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

DIMS = [64, 128, 256, 512, 768]
FULL_DIM = 768

def encode_at_dim(model, words, target_dim, normalize=True):
    full_emb = model.encode(words, normalize_embeddings=False, show_progress_bar=False)
    truncated = full_emb[:, :target_dim]
    if normalize:
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    return truncated

# ============ WORD SETS ============
MALE_WORDS = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE_WORDS = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

EA_WORDS = ["Adam", "Harry", "Josh", "Roger", "Alan", "Frank", "Justin", "Ryan", 
            "Andrew", "Jack", "Matthew", "Stephen", "Brad", "Greg", "Paul", "Jonathan", 
            "Peter", "Amanda", "Courtney", "Heather", "Melanie", "Katie", "Betsy", 
            "Kristin", "Nancy", "Stephanie", "Ellen", "Lauren", "Colleen", "Emily", 
            "Megan", "Rachel"]
AA_WORDS = ["Alonzo", "Jamel", "Theo", "Alphonse", "Jerome", "Leroy", "Torrance", 
            "Darnell", "Lamar", "Lionel", "Tyree", "Deion", "Lamont", "Malik", "Terrence", 
            "Tyrone", "Lonnie", "Tamika", "Ebony", "Latisha", "Keisha", "Shaniqua", 
            "Jasmine", "Tanisha", "Tia", "Lakisha", "Latoya", "Shereen", "Nichelle", 
            "Shanise", "Sharise", "Tameka"]

# ============ WEAT EFFECT SIZE (cleaner version) ============
def differential_association(word_emb, attr_A_emb, attr_B_emb):
    sim_A = np.dot(word_emb, attr_A_emb.T)
    sim_B = np.dot(word_emb, attr_B_emb.T)
    return np.mean(sim_A) - np.mean(sim_B)

def compute_weat_es(target_X, target_Y, attr_A, attr_B, model, dim):
    X_emb = encode_at_dim(model, target_X, dim)
    Y_emb = encode_at_dim(model, target_Y, dim)
    A_emb = encode_at_dim(model, attr_A, dim)
    B_emb = encode_at_dim(model, attr_B, dim)
    
    s_X = np.array([differential_association(x, A_emb, B_emb) for x in X_emb])
    s_Y = np.array([differential_association(y, A_emb, B_emb) for y in Y_emb])
    
    mean_diff = np.mean(s_X) - np.mean(s_Y)
    pooled_std = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std

# ============ PCA BIAS SUBSPACE CONCENTRATION ============
def compute_bias_concentration(model, group_A_words, group_B_words, dims=DIMS):
    """
    Compute C(d) for each dimension d.
    C(d) = ||v_bias[1:d]||^2 / ||v_bias||^2
    where v_bias is the top principal direction of (A - B) differences in FULL space.
    """
    # Encode in FULL dimension (768)
    A_full = encode_at_dim(model, group_A_words, FULL_DIM, normalize=True)
    B_full = encode_at_dim(model, group_B_words, FULL_DIM, normalize=True)
    
    # Compute difference vectors
    diffs = A_full - B_full  # shape: (n_words, 768)
    
    # PCA on differences to find bias direction
    pca = PCA(n_components=1)
    pca.fit(diffs)
    v_bias_full = pca.components_[0]  # shape: (768,)
    
    # Full norm squared
    full_norm_sq = np.sum(v_bias_full ** 2)
    
    concentrations = {}
    for d in dims:
        v_prefix = v_bias_full[:d]
        prefix_norm_sq = np.sum(v_prefix ** 2)
        C_d = prefix_norm_sq / full_norm_sq
        concentrations[d] = C_d
    
    return concentrations, v_bias_full

# ============ RUN EVERYTHING ============

print("=" * 70)
print("1. WEAT EFFECT SIZES ACROSS DIMENSIONS")
print("=" * 70)

# Tests to plot
test_configs = [
    ("Gender-Science", MALE_WORDS, FEMALE_WORDS, 
     ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy", "biology", "engineering"],
     ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama", "sculpture", "painting"]),
    ("Gender-Math", MALE_WORDS, FEMALE_WORDS,
     ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition", "statistics", "measurement"],
     ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture", "painting", "music"]),
    ("Race-Pleasant", EA_WORDS, AA_WORDS,
     ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure"],
     ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink"]),
]

weat_results = {name: {'std': [], 'mrl': []} for name, _, _, _, _ in test_configs}

for test_name, X, Y, A, B in test_configs:
    print(f"\n{test_name}:")
    print(f"{'Dim':>5} {'Std ES':>10} {'MRL ES':>10}")
    for dim in DIMS:
        es_std = compute_weat_es(X, Y, A, B, model_std, dim)
        es_mrl = compute_weat_es(X, Y, A, B, model_mrl, dim)
        weat_results[test_name]['std'].append(es_std)
        weat_results[test_name]['mrl'].append(es_mrl)
        print(f"{dim:>5} {es_std:>+10.4f} {es_mrl:>+10.4f}")

print("\n" + "=" * 70)
print("2. BIAS SUBSPACE CONCENTRATION C(d)")
print("=" * 70)

# Gender concentration
C_gender_std, v_gender_std = compute_bias_concentration(model_std, MALE_WORDS, FEMALE_WORDS)
C_gender_mrl, v_gender_mrl = compute_bias_concentration(model_mrl, MALE_WORDS, FEMALE_WORDS)

# Race concentration  
C_race_std, v_race_std = compute_bias_concentration(model_std, EA_WORDS, AA_WORDS)
C_race_mrl, v_race_mrl = compute_bias_concentration(model_mrl, EA_WORDS, AA_WORDS)

print("\nGender Bias Concentration C(d):")
print(f"{'Dim':>5} {'Std C(d)':>12} {'MRL C(d)':>12} {'Random C(d)':>12}")
for dim in DIMS:
    random_baseline = dim / FULL_DIM
    print(f"{dim:>5} {C_gender_std[dim]:>12.4f} {C_gender_mrl[dim]:>12.4f} {random_baseline:>12.4f}")

print("\nRace Bias Concentration C(d):")
print(f"{'Dim':>5} {'Std C(d)':>12} {'MRL C(d)':>12} {'Random C(d)':>12}")
for dim in DIMS:
    random_baseline = dim / FULL_DIM
    print(f"{dim:>5} {C_race_std[dim]:>12.4f} {C_race_mrl[dim]:>12.4f} {random_baseline:>12.4f}")

# ============ PLOTTING ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: WEAT curves
ax = axes[0, 0]
for test_name in ["Gender-Science", "Gender-Math"]:
    ax.plot(DIMS, weat_results[test_name]['std'], 'o-', label=f"Std: {test_name}")
    ax.plot(DIMS, weat_results[test_name]['mrl'], 's--', label=f"MRL: {test_name}")
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Dimension")
ax.set_ylabel("WEAT Effect Size")
ax.set_title("Gender Bias Across Dimensions")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Race WEAT
ax = axes[0, 1]
ax.plot(DIMS, weat_results["Race-Pleasant"]['std'], 'o-', label="Std: Race-Pleasant")
ax.plot(DIMS, weat_results["Race-Pleasant"]['mrl'], 's--', label="MRL: Race-Pleasant")
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Dimension")
ax.set_ylabel("WEAT Effect Size")
ax.set_title("Race Bias Across Dimensions")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Gender Concentration C(d)
ax = axes[1, 0]
ax.plot(DIMS, [C_gender_std[d] for d in DIMS], 'o-', label="Standard", linewidth=2)
ax.plot(DIMS, [C_gender_mrl[d] for d in DIMS], 's--', label="MRL", linewidth=2)
ax.plot(DIMS, [d/FULL_DIM for d in DIMS], 'k:', label="Random Baseline (d/D)", linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("C(d) = ||v_bias[1:d]||² / ||v_bias||²")
ax.set_title("Gender Bias Concentration in Prefix Subspaces")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Race Concentration C(d)
ax = axes[1, 1]
ax.plot(DIMS, [C_race_std[d] for d in DIMS], 'o-', label="Standard", linewidth=2)
ax.plot(DIMS, [C_race_mrl[d] for d in DIMS], 's--', label="MRL", linewidth=2)
ax.plot(DIMS, [d/FULL_DIM for d in DIMS], 'k:', label="Random Baseline (d/D)", linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("C(d) = ||v_bias[1:d]||² / ||v_bias||²")
ax.set_title("Race Bias Concentration in Prefix Subspaces")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bias_analysis_step4.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: bias_analysis_step4.png")
plt.show()

print("\n✓ Step 4 complete. Subspace concentration computed and visualized.")