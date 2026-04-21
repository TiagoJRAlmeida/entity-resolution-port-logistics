from string_cleaning import is_person_name
import Levenshtein
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re

LEGAL_PREFIX = {
    'plc', 'inc', 'company', 'tic', 'dis', 'sti', 'ltd', 'sirketi', 'ic', 'spa', 've', 'sa', 'ltdsti', 
    'san', 'as', 'pte', 'bv', 'ticaret', 'anonim', 'co', 'llc', 'gmbh', 'corp', 'ltda', 'lda'
}

REGION_KEYWORDS = {
    # Full names
    'usa', 'uk', 'germany', 'greece', 'china', 'india', 'israel', 'egypt', 'uae',
    'turkey', 'saudi', 'france', 'spain', 'italy', 'canada', 'brazil', 'mexico', 'australia',
    'panama', 'japan', 'senegal', 'colombia', 'vietnam', 'indonesia', 'netherlands',
    'south africa', 'portugal', 'argentina', 'singapore', 'russia', 'poland', 'peru', 'chile',
    'morocco', 'algeria', 'kenya', 'tanzania', 'ghana', 'nigeria', 'costa rica', 'ecuador',
    'bolivia', 'uruguay', 'venezuela', 'honduras', 'guatemala', 'nicaragua', 'paraguay',
    'dominican', 'el salvador', 'lebanon', 'sri lanka', 'bangladesh', 'nepal', 'pakistan',
    'thailand', 'malaysia', 'myanmar', 'taiwan', 'hong kong', 'south korea', 'korea', 'mozambique',
    'angola', 'tunisia', 'libya', 'zambia', 'zimbabwe', 'cameroon', 'ethiopia', 'botswana', 'grecia', 
    'belgica', 'noruega', 'suecia', 'alemanha', 'irlanda', 'dinamarca', 'norway', 'denmark', 'franca',
    'espanha', 'marrocos', 'irlanda', 'bulgaria', 'estonia', 'paises baixos', 'britanica', 'coreia do sul',
    'italia', 'brasil', 

    # Cities or custom regions
    'dubai', 'riyadh', 'jeddah', 'cairo', 'doha', 'abu dhabi', 'amman', 'santiago',
    'barcelona', 'madrid', 'guadalajara', 'monterrey', 'puebla', 'lyon', 'porto', 'lisbon',
    'paris', 'milan', 'milano', 'napoli', 'rome', 'athens', 'sofia', 'bucharest', 'vienna',
    'warsaw', 'zagreb', 'brussels', 'oslo', 'stockholm', 'copenhagen', 'geneva', 'lausanne',
    'luxembourg', 'shanghai', 'beijing', 'shenzhen', 'qingdao', 'xiamen', 'tianjin', 'ilha do faial',
    'ilha do corvo', 'ilha terceira', 'ilha sao miguel', 'texas', 'hong kong', 'matosinhos', 'leixoes',
    'setubal', 'sesimbra', 'ponta delgada', 'setubalense', 'portimao', 'roque do pico', 'madalena do pico',
    'praia da vitoria'

    # ISO Alpha-2 country codes
    'us', 'fr', 'de', 'pt', 'es', 'it', 'nl', 'be', 'se', 'no', 'fi', 'dk', 'pl', 'gr',
    'ro', 'bg', 'cz', 'hu', 'sk', 'si', 'at', 'ch', 'ie', 'ru', 'tr', 'cn', 'jp', 'kr', 'in',
    'id', 'th', 'my', 'vn', 'sg', 'ph', 'bd', 'pk', 'il', 'eg', 'ae', 'qa', 'kw', 'dz',
    'ma', 'tn', 'ng', 'gh', 'za', 'zm', 'zw', 'cm', 'ke', 'tz',

    # ISO Alpha-3 codes (and common short customs abbreviations)
    'usa', 'mex', 'can', 'esp', 'bra', 'deu', 'fra', 'ita', 'prt', 'nld', 'bel', 'che', 'swe',
    'nor', 'fin', 'dnk', 'pol', 'grc', 'rou', 'bgr', 'cze', 'hun', 'svk', 'svn', 'aut', 'irl',
    'gbr', 'rus', 'tur', 'chn', 'jpn', 'kor', 'ind', 'idn', 'tha', 'mys', 'vnm', 'sgp', 'phl',
    'pak', 'bgd', 'isr', 'egy', 'sau', 'are', 'qat', 'kwt', 'dza', 'mar', 'tun', 'nga', 'gha',
    'zaf', 'zmb', 'zwe', 'cmr', 'ken', 'tza',

    # Seen in the clusters
    'esp', 'pt', 'po', 'sh', 'sur', 'sor', 'ne', 'lim', 'sen', 'del', 'mex', "spa", "viet", "chin"
}


def remove_regions(name):
    tokens = name.split()

    cleaned_tokens = [
        token for token in tokens
        if token not in REGION_KEYWORDS and token not in LEGAL_PREFIX
    ]

    return ' '.join(cleaned_tokens)


def extract_regions(name):
    name_lower = name.lower()
    
    # Garante que só vai buscar frases ou palavras inteiras (ex: 'ilha terceira', não 'sa' dentro de 'saude')
    regions = {
        region for region in REGION_KEYWORDS
        if re.search(rf'\b{re.escape(region)}\b', name_lower)
    }
    
    return regions if regions else {"default"}


def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)


def postprocess_by_region(clusters, similarity_threshold=0.66):
    new_clusters = []

    for cluster in clusters:
        name_to_regions = {name: extract_regions(name) for name in cluster}

        # Flatten all region sets to check how many unique regions exist
        all_regions = set()
        for regions in name_to_regions.values():
            all_regions.update(regions)

        # Skip processing if only one region (excluding "default")
        filtered_regions = all_regions - {"default"}
        if len(filtered_regions) <= 1:
            new_clusters.append(cluster)
            continue

        # Proceed with regional grouping
        region_groups = []
        assigned = set()

        for name, regions in name_to_regions.items():
            if name in assigned:
                continue
            group = [name]
            assigned.add(name)

            for other_name, other_regions in name_to_regions.items():
                if other_name in assigned:
                    continue

                similarity = jaccard_similarity(regions, other_regions)
                if similarity >= similarity_threshold:
                    group.append(other_name)
                    assigned.add(other_name)

            region_groups.append(group)

        new_clusters.extend(region_groups)

    return new_clusters


def get_core_prefix(name, max_tokens=2):
    tokens = [t for t in name.split() if t not in LEGAL_PREFIX]
    return " ".join(tokens[:max_tokens])


def shared_prefix(a, b, n=2):
    return get_core_prefix(a, n) == get_core_prefix(b, n)


def weighted_token_jaccard(a, b):
    tokens_a = set(a.split())
    tokens_b = set(b.split())

    intersection = 0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    
    for token in tokens_a & tokens_b:
        if token in LEGAL_PREFIX or token in REGION_KEYWORDS:
            continue
            #intersection += 0.2  # small weight
        else:
            intersection += 1.0  # full weight

    return intersection / len(union)


def normalized_levenshtein_core(name1, name2):
    # Remove legal terms
    t1 = " ".join([t for t in name1.split() if t not in LEGAL_PREFIX and t not in REGION_KEYWORDS])
    t2 = " ".join([t for t in name2.split() if t not in LEGAL_PREFIX and t not in REGION_KEYWORDS])
    if not t1 or not t2:
        return 1.0  # prevent false match on empty core
    dist = Levenshtein.distance(t1, t2)
    return dist / max(len(t1), len(t2))


def combined_similarity(name1, name2):
    lev_sim = 1 - normalized_levenshtein_core(name1, name2)
    jac_sim = weighted_token_jaccard(name1, name2)
    return 0.7 * lev_sim + 0.3 * jac_sim


def refine_noisy_cluster(cluster):
    refined = []
    for name in cluster:
        added = False
        for group in refined:
            if all(combined_similarity(name, other) >= 0.75 for other in group):
                group.append(name)
                added = True
                break
        if not added:
            refined.append([name])
    return refined


# NOTE: names argument must be a set of already cleaned names.
# base_clusters argument must also contain names in their cleaned form
def create_clusters(names=None, similarity_threshold=0.75, base_clusters=[]):
    from sentence_transformers import SentenceTransformer
    import faiss
    if not names:
        return base_clusters

    names = {name for name in names if name.strip()}

    # Flatten base clusters and store index
    base_clustered_names = set()
    base_cluster_map = {}
    for idx, cluster in enumerate(base_clusters):
        for name in cluster:
            base_clustered_names.add(name)
            base_cluster_map[name] = idx

    # Why create and use "all_name" instead of just remaining_names?
    # Because we also want to embed the base clustered names, so that they are indexed and
    # can appear in the query results.
    # This way, we can also use the expand upon the base clusters.  
    remaining_names = names - base_clustered_names
    all_names = list(base_clustered_names) + list(remaining_names)
    processed_all_names = [remove_regions(name) for name in all_names]

    # Step 1: Get embeddings
    #print("Calculating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(processed_all_names, convert_to_numpy=True)

    # Step 2: Normalize embeddings
    #print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # Step 3: Index embeddings
    #print("Indexing embeddings...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    clustered = [False] * len(all_names)
    clusters = [list(cluster) for cluster in base_clusters]

    #print(f"Building Clustering...\n")
    for i in tqdm(range(len(all_names)), desc="Adding Names to Clusters", unit="name"):
        # Allow others to attach to base, but not re-process base elements
        if all_names[i] in base_clustered_names:
            continue

        if clustered[i]:
            continue

        new_name = all_names[i]
        embedding = embeddings[i]

        D, I = index.search(np.expand_dims(embedding, axis=0), 1000)

        cluster = [new_name]
        best_match = None
        best_score = 0

        for idx, score in zip(I[0], D[0]):
            candidate_name = all_names[idx]
            if idx == i or clustered[idx]:
                continue
            
            adjusted_threshold = 0.95 if is_person_name(new_name) or is_person_name(candidate_name) else similarity_threshold

            if score >= adjusted_threshold: #and shared_prefix(new_name, candidate_name):
                if candidate_name in base_clustered_names:
                    if score > best_score:
                        best_score = score
                        best_match = base_cluster_map[candidate_name] # Best match is an index to the clusters list!
                else:
                    cluster.append(candidate_name)
                    clustered[idx] = True

        if best_match is not None:
            clusters[best_match].extend(cluster)
        else:
            clusters.append(cluster)
        clustered[i] = True

    #print("Refining noisy clusters with Levenshtein (if necessary)...")
    final_clusters = []
    for cluster in clusters:
        if len(cluster) > 5:  # Filter only on bigger clusters
            refined = refine_noisy_cluster(cluster)
            final_clusters.extend(refined)
        else:
            final_clusters.append(cluster)

    return final_clusters


# NOTE: This function returns 1 thing:
# 1. The synonym clusters, a list of lists, where each list is a cluster 
def create_ground_truth_synonym_clusters(synonyms_map):
    # 1. Map every ID to the set of names that mention it
    id_to_names = defaultdict(set)
    name_to_ids = defaultdict(set)

    for variant, canon_map in synonyms_map.items():
        for canon, ids in canon_map.items():
            for _id in ids:
                id_to_names[_id].update([variant, canon])
                name_to_ids[variant].add(_id)
                name_to_ids[canon].add(_id)

    # 2. Union-find to merge IDs that share a name
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # initialize each ID as its own parent
    for _id in id_to_names:
        parent[_id] = _id

    # union all IDs that share any name
    for name, ids in name_to_ids.items():
        ids = list(ids)
        for i in range(1, len(ids)):
            union(ids[0], ids[i])

    # 3. Gather clusters by representative ID
    rep_to_names = defaultdict(set)
    for _id, names in id_to_names.items():
        rep = find(_id)
        rep_to_names[rep].update(names)

    # 4. Format result
    clusters = [sorted(names) for names in rep_to_names.values()]

    return clusters
