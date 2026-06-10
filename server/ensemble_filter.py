# coding: utf-8
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

def run_ensemble_filter(features_list, historical_cos_sims, filtered_indices_list):
    """
    Stage 1 & 2 of the Two-Stage Ensemble Clustering:
    - Stage 1 (Probabilistic Grouping): MinMaxScaler + GaussianMixture(n_components=2)
    - Stage 2 (Cosine Sanity Check): Forces clients with cos_sim < -0.5 into the malicious cluster.

    Returns:
        cluster_1_indices, cluster_2_indices
    """
    if len(features_list) <= 1:
        return filtered_indices_list, filtered_indices_list

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    reduced_features = scaler.fit_transform(features_list)
    reduced_features = np.nan_to_num(
        reduced_features, nan=0.0, posinf=1.0, neginf=-1.0
    ).tolist()

    # Stage 1: Probabilistic Grouping
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    cluster_labels = gmm.fit_predict(reduced_features)
    
    value_indices = {
        value: [index for index, val in enumerate(cluster_labels) if val == value]
        for value in set(cluster_labels)
    }
    if 0 not in value_indices: value_indices[0] = []
    if 1 not in value_indices: value_indices[1] = []

    # Stage 2: Cosine Sanity Check
    c1_cosines = [historical_cos_sims[i] for i in value_indices[0]]
    c2_cosines = [historical_cos_sims[i] for i in value_indices[1]]
    
    c1_has_negative = any(c < -0.5 for c in c1_cosines)
    c1_has_positive = any(c > 0.0 for c in c1_cosines)
    c2_has_negative = any(c < -0.5 for c in c2_cosines)
    c2_has_positive = any(c > 0.0 for c in c2_cosines)
    
    if c1_has_negative and c1_has_positive:
        to_move = [i for i, c in zip(value_indices[0], c1_cosines) if c < -0.5]
        value_indices[0] = [i for i in value_indices[0] if i not in to_move]
        value_indices[1].extend(to_move)
    
    if c2_has_negative and c2_has_positive:
        to_move = [i for i, c in zip(value_indices[1], c2_cosines) if c < -0.5]
        value_indices[1] = [i for i in value_indices[1] if i not in to_move]
        value_indices[0].extend(to_move)

    cluster_1_indices = [filtered_indices_list[i] for i in value_indices[0]]
    cluster_2_indices = [filtered_indices_list[i] for i in value_indices[1]]

    return cluster_1_indices, cluster_2_indices
