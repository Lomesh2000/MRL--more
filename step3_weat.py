"""
Step 3: WEAT (Word Embedding Association Test)
Rigorous implementation following Caliskan et al. (2017, Science)
Computes effect size + permutation p-value at multiple dimensions.
"""

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
DIMS = [64, 128, 256, 512, 768]
N_PERMUTATIONS = 10000  # 10k for speed; use 100k for final paper
np.random.seed(42)

# ============ MODEL LOADING ============
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

# ============ TRUNCATION + NORMALIZATION ============
def encode_at_dim(model, words, target_dim, normalize=True):
    """Encode words and truncate to target_dim with L2 re-normalization."""
    full_emb = model.encode(words, normalize_embeddings=False, show_progress_bar=False)
    truncated = full_emb[:, :target_dim]
    if normalize:
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    return truncated

# ============ STANDARD WEAT WORD SETS ============
# Source: Caliskan et al. (2017) supplementary material

# Gender
MALE_WORDS = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE_WORDS = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

# Career vs Family
CAREER_WORDS = ["executive", "management", "professional", "corporation", "salary", 
                "office", "business", "career", "manager", "administrator"]
FAMILY_WORDS = ["home", "parents", "children", "family", "cousins", "marriage", 
                "wedding", "relatives", "household", "domestic"]

# Math vs Arts
MATH_WORDS = ["math", "algebra", "geometry", "calculus", "equations", "computation", 
              "numbers", "addition", "statistics", "measurement"]
ARTS_WORDS = ["poetry", "art", "dance", "literature", "novel", "symphony", 
              "drama", "sculpture", "painting", "music"]

# Pleasant vs Unpleasant
PLEASANT_WORDS = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", 
                  "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", 
                  "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", 
                  "happy", "laughter", "paradise", "vacation"]
UNPLEASANT_WORDS = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", 
                    "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", 
                    "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", 
                    "rotten", "vomit", "agony", "prison"]

# European American vs African American names (from Caliskan et al.)
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

# Science vs Arts
SCIENCE_WORDS = ["science", "technology", "physics", "chemistry", "Einstein", "NASA", 
                 "experiment", "astronomy", "biology", "engineering"]
ARTS_WORDS2 = ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", 
               "drama", "sculpture", "painting"]

# ============ WEAT COMPUTATION ============
def differential_association(word_emb, attr_A_emb, attr_B_emb):
    """
    s(w, A, B) = mean(cos(w, a)) - mean(cos(w, b))
    Since vectors are L2-normalized, cos = dot product.
    """
    sim_A = np.dot(word_emb, attr_A_emb.T)  # shape: (1, |A|)
    sim_B = np.dot(word_emb, attr_B_emb.T)  # shape: (1, |B|)
    return np.mean(sim_A) - np.mean(sim_B)

def compute_weat(target_X, target_Y, attr_A, attr_B, model, dim):
    """
    Compute WEAT effect size and permutation p-value.
    
    Returns:
        effect_size: float (Cohen's d)
        p_value: float (two-tailed permutation test)
        mean_diff: float (mean_X - mean_Y)
    """
    # Encode all words at target dimension
    X_emb = encode_at_dim(model, target_X, dim)
    Y_emb = encode_at_dim(model, target_Y, dim)
    A_emb = encode_at_dim(model, attr_A, dim)
    B_emb = encode_at_dim(model, attr_B, dim)
    
    # Compute differential associations for all target words
    s_X = np.array([differential_association(x, A_emb, B_emb) for x in X_emb])
    s_Y = np.array([differential_association(y, A_emb, B_emb) for y in Y_emb])
    
    # Effect size: Cohen's d
    mean_diff = np.mean(s_X) - np.mean(s_Y)
    pooled_std = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if pooled_std == 0:
        effect_size = 0.0
    else:
        effect_size = mean_diff / pooled_std
    
    # Permutation test
    all_s = np.concatenate([s_X, s_Y])
    n_X = len(s_X)
    n_total = len(all_s)
    
    count_extreme = 0
    for _ in range(N_PERMUTATIONS):
        np.random.shuffle(all_s)
        shuffled_X = all_s[:n_X]
        shuffled_Y = all_s[n_X:]
        shuffled_diff = np.mean(shuffled_X) - np.mean(shuffled_Y)
        # Two-tailed: count if absolute shuffled diff >= absolute observed diff
        if abs(shuffled_diff) >= abs(mean_diff):
            count_extreme += 1
    
    p_value = count_extreme / N_PERMUTATIONS
    
    return effect_size, p_value, mean_diff

# ============ TEST DEFINITIONS ============
TESTS = [
    ("Gender-Career", MALE_WORDS, FEMALE_WORDS, CAREER_WORDS, FAMILY_WORDS),
    ("Gender-Math", MALE_WORDS, FEMALE_WORDS, MATH_WORDS, ARTS_WORDS),
    ("Gender-Science", MALE_WORDS, FEMALE_WORDS, SCIENCE_WORDS, ARTS_WORDS2),
    ("Race-Pleasant", EA_WORDS, AA_WORDS, PLEASANT_WORDS, UNPLEASANT_WORDS),
    ("Race-Career", EA_WORDS, AA_WORDS, CAREER_WORDS, FAMILY_WORDS),
]

# ============ RUN ALL TESTS ============
print("=" * 80)
print("WEAT RESULTS: Effect Size (Cohen's d) and Permutation p-value")
print("=" * 80)
print(f"{'Test':<<20} {'Dim':>5} {'Std ES':>10} {'Std p':>10} {'MRL ES':>10} {'MRL p':>10}")
print("-" * 80)

results = []

for test_name, X, Y, A, B in TESTS:
    for dim in DIMS:
        # Standard model
        es_std, p_std, _ = compute_weat(X, Y, A, B, model_std, dim)
        # MRL model
        es_mrl, p_mrl, _ = compute_weat(X, Y, A, B, model_mrl, dim)
        
        results.append({
            'test': test_name, 'dim': dim,
            'std_es': es_std, 'std_p': p_std,
            'mrl_es': es_mrl, 'mrl_p': p_mrl
        })
        
        # Format: *** p<<0.001, ** p<<0.01, * p<<0.05
        sig_std = "***" if p_std < 0.001 else "**" if p_std < 0.01 else "*" if p_std < 0.05 else ""
        sig_mrl = "***" if p_mrl < 0.001 else "**" if p_mrl < 0.01 else "*" if p_mrl < 0.05 else ""
        
        print(f"{test_name:<20} {dim:>5} {es_std:>+9.4f}{sig_std:<1} {p_std:>9.4f} "
              f"{es_mrl:>+9.4f}{sig_mrl:<1} {p_mrl:>9.4f}")

print("-" * 80)
print("Significance: *** p<<0.001, ** p<<0.01, * p<<0.05")
print("\n✓ Step 3 complete. WEAT bias measurement across dimensions finished.")