from itertools import combinations
import numpy as np
import Levenshtein
from Clustering.clustering import combined_similarity


def cluster_cohesion(cluster):
    """Compute the average pairwise similarity within a cluster"""
    if len(cluster) < 2:
        return 1.0
    sims = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            sims.append(combined_similarity(cluster[i], cluster[j]))
    return sum(sims) / len(sims)


def cluster_similarity(cluster_a, cluster_b):
    """Compute average similarity between two clusters"""
    sims = []
    for name1 in cluster_a:
        for name2 in cluster_b:
            sims.append(combined_similarity(name1, name2))
    return sum(sims) / len(sims)


def clean_ground_truth_clusters(clusters, cohesion_thresh=0.7, inter_thresh=0.65):
    """
    Removes:
    - Clusters with low internal cohesion.
    - Clusters that are highly similar to other clusters (assumed redundant).
    """
    # Step 1: Remove low-cohesion clusters
    filtered = [c for c in clusters if cluster_cohesion(c) >= cohesion_thresh]

    # Step 2: Remove redundant clusters
    to_keep = [True] * len(filtered)

    for i in range(len(filtered)):
        if not to_keep[i]:
            continue
        for j in range(i + 1, len(filtered)):
            if not to_keep[j]:
                continue
            if cluster_similarity(filtered[i], filtered[j]) >= inter_thresh:
                # Remove the smaller one
                if len(filtered[i]) >= len(filtered[j]):
                    to_keep[j] = False
                else:
                    to_keep[i] = False

    cleaned_clusters = [filtered[i] for i in range(len(filtered)) if to_keep[i]]
    return cleaned_clusters


# Purpose: 
#       Measures how similar the names within the same cluster are.
# How it works:
#       1. Iterates over each cluster.
#       2. For clusters with more than one name, generates all pairs (a, b) inside the cluster.
#       3. For each pair, calculates normalized Levenshtein similarity: sim = 1 - Levenshtein.distance(a, b) / max(len(a), len(b))
#       4. This gives a score between 0 (completely different) and 1 (identical).
#       5. Averages all these similarity scores across all clusters.
# Interpretation:
#       Higher score = better (clustering puts very similar names together).
#       If the score is ~0.9+, most clusters contain only slight spelling or format variations.
def average_intra_cluster_similarity(clusters):
    sims = []
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        for a, b in combinations(cluster, 2):
            sim = 1 - Levenshtein.distance(a, b) / max(len(a), len(b))
            sims.append(sim)
    return np.mean(sims) if sims else 0


# Purpose: 
#       Measures how similar the names are across different clusters.
# How it works:
#       1. Iterates over all pairs of distinct clusters (i, j) (where i < j).
#       2. Takes at most 2 names from each cluster to reduce computation.
#       3. Collects up to sample_size inter-cluster name pairs.
#       4. Calculates the normalized Levenshtein similarity for each pair, just like above.
#       5. Averages all similarity scores.
# Interpretation:
#     Lower score = better (clusters are well-separated — names from different clusters are dissimilar).
#     A high inter-cluster score might mean your clustering is merging different companies together.
def average_inter_cluster_similarity(clusters, sample_size=1000):
    sims = []
    pairs = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for a in clusters[i][:2]:
                for b in clusters[j][:2]:
                    pairs.append((a, b))
                    if len(pairs) >= sample_size:
                        break
                if len(pairs) >= sample_size:
                    break
            if len(pairs) >= sample_size:
                break
        if len(pairs) >= sample_size:
            break

    for a, b in pairs:
        sim = 1 - Levenshtein.distance(a, b) / max(len(a), len(b))
        sims.append(sim)
    return np.mean(sims) if sims else 0


# NOTE: To calculate the clustering accuracy, we need to compare the true clusters with the predicted clusters.
# The function get_pairs takes a cluster (a list of lists) and returns a set of pairs of names.
# Each pair is a tuple of two names, and the pairs are sorted to ensure consistency.
# This function was done to avoid the problem of having to find the correct cluster to compare with the predicted clusters.
# However, according to Professor LSL, this is an creative solution, however confusing and its not clear if it is correct.
# Nevertheless, this is a known algorithm called "Pairwise F1-Score" and it is used in many clustering algorithms.
# But for professors to understand it and belive in it, we need to find papers and articles that use it and cite them.
def get_pairs(cluster):
    pairs = set()
    for group in cluster:
        if len(group) == 1:
            a = group[0]
            pairs.add((a, a))
        else:
            for a, b in combinations(sorted(group), 2):
                pairs.add((a, b))

    return pairs

# NOTE: This function calculates the clustering accuracy based on the true clusters and predicted clusters.
# It uses precision, recall, and F1 score as metrics.
# The function also takes an optional parameter training_names to filter the clusters based on the training set.
# For more details on the metrics, search about "precision", "recall", and "F1 score" in the internet, as it is a well documented subject.
def clustering_accuracy(true_clusters, predicted_clusters):  
    true_pairs = get_pairs(true_clusters)
    pred_pairs = get_pairs(predicted_clusters)

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# cluster_A = [["Test", "Test1"], ["Hello"], ["Sun", "Sun1"]]
# print(get_pairs(cluster_A))
# # {('Sun', 'Sun1'), ('Test', 'Test1'), ('Hello', 'Hello')}
# cluster_B = [["Test", "Test1", "Test2"], ["Hello"]]
# print(get_pairs(cluster_B))
# # {('Test', 'Test1'), ('Test', 'Test2'), ('Test1', 'Test2'), ('Hello', 'Hello')}

# metrics = clustering_accuracy(cluster_A, cluster_B)
# print(metrics)
# # {'precision': 0.5, 'recall': 0.6666666666666666, 'f1_score': 0.5714285714285715}
