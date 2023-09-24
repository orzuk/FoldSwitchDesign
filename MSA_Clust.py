# Pipeline for clustering alignemnts, predicting contacts for each cluster and comparing them
# Goal is to identify evidence fro difference in co-evolution in the different clusters,
# possible indicative of differences in the structure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Run clustering of the Multiple Sequence alignemnts,
# Default is 2 clusters (?), but can be more
# USe the method from AF-clust
def cluster_MSAs(MSA, clust_params):
    # Run ClusterMSA script (??)

    return MSA_cluster  # an object containing the partition of the MSA into multiple clusters


# Compute pairwise distances between the different contact maps
def compute_cmap_distances(cmap):
    D = 0

    # Code here
    return D


# Compute pairwise distances between the different contact maps
def compute_seq_distances(MSA_clust):
    D = 0

    # Code here

    return D


# Predict if a family of proteins is fold-switching or is having a single structure,
# from the co-evolutionary patterns of the MSA
def predict_fold_switch_from_MSA_cluster(MSA, clust_params):

    MSA_clust = cluster_MSAs(MSA, clust_params)  # first cluster the MSAs
    n_clust = len(MSA_clust)  # number of clusters (can be set as a parameter)

    for i in range(n_clust):  # loop on clusters
        cmap[i] = MSA_transformer(MSA_clust[i])  # compute pairwise attention map for cluster

    cmap_dist = compute_cmap_distances(cmap)  # similarity of cluster contact maps
    seq_dist = compute_seq_distances(MSA_clust)  # sequence similarity of clusters

    fold_switch_pred_y = (cmap_dist - clust_params@beta * seq_dist > 0)  # replace by a learned function

    return fold_switch_pred_y, cmap_dist, seq_dist


# Run pipeline on a bunch of families (MSAs can be given, or read from file
# or generated on the fly)
# MSAs can be represented as a3m format
def predict_fold_switch_pipeline(MSAs_dir, clust_params):
    # loop on MSAs
    n_fam = len(MSA)

    pred_vec = [0] * n_fam
    for i in range(n_fam):
        cur_MSA = MSAs_dir[i] # This should be replaced by code generating/reading the MSA
        pred_vec[i] = predict_fold_switch_from_MSA_cluster(cur_MSA, clust_params)



