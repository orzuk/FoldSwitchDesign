# Pipeline for clustering alignemnts, predicting contacts for each cluster and comparing them
# Goal is to identify evidence for difference in co-evolution in the different clusters,
# possible indicative of differences in the structure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse
import hashlib
import jax
import jax.numpy as jnp
import re
from polyleven import levenshtein
import time
import random

import Levenshtein as pylev

import subprocess
from glob import glob
from Bio import SeqIO
from Bio.Align import AlignInfo
from Bio import AlignIO

import itertools as itl

from collections import defaultdict

#import glob

sys.path.append('alphafold')


# Run clustering of the Multiple Sequence alignemnts,
# Default is 2 clusters (?), but can be more
# USe the method from AF-clust
def cluster_MSAs(MSA, clust_params):
    # Run ClusterMSA script (??)

    # USe AF-cluster script
    AF_cluster_str = 'python ../AF_cluster/scripts/ClusterMSA.py EX -i ' + \
    '../AF_cluster/data_sep2022/00_KaiB/2QKEE_colabfold.a3m -o subsampled_MSAs'  # change later to input MSA
    subprocess.call(AF_cluster_str)

    clusters_dir = "../AF_Cluster/subsampled_MSAs"
    clusters_file_names = glob(clusters_dir + "/EX_*a3m")
    n_clusters = len(clusters_file_names)
    print(n_clusters)
    MSA_clusters = [None]*n_clusters  # replace by reading from output file
    for i in range(n_clusters):
        MSA_clusters[i] = AlignIO.read(open(clusters_file_names[i]), "fasta")


#    subprocess.call('python hello.py') # small check
    return MSA_clusters  # an object containing the partition of the MSA into multiple clusters


# Compute pairwise distances between the different contact maps
def compute_cmap_distances(cmap):
    D = 0
    n_maps = len(cmap)  # number of clusters/contact maps

    for i in range(n_maps): # Code here
        for j in range(i, n_maps):
            D = D + sum(cmap[i] - cmap[j])**2
    return 2*D/(n_maps*(n_maps-1))  # normalize


# Compute pairwise distances between the different clusters
def compute_seq_distances(MSA_clust):
    n_clusters = len(MSA_clust)
    D = np.zeros((n_clusters, n_clusters))

#    for i in range(n_clusters):
#        summary_align = AlignInfo.SummaryInfo(MSA_clust[i])
#        PSSM = summary_align.MSA_clust.pos_specific_score_matrix()  # Code here
#
#    avg_dist_to_query = np.mean([1-levenshtein(x, query_['sequence'].iloc[0])/L for x in df.loc[df.dbscan_label==-1]['sequence'].tolist()])
#    lprint('avg identity to query of unclustered: %.2f' % avg_dist_to_query,f)


    max_seq_per_cluster = 10  # maximum number of sequences per cluster
    for i in range(n_clusters):  # loop on pairs of clusters
        for j in range(i, n_clusters):
            n_i = len(MSA_clust[i])
            n_j = len(MSA_clust[j])
            II = random.sample(range(n_i), min(n_i, max_seq_per_cluster))
            JJ = random.sample(range(n_j), min(n_j, max_seq_per_cluster))

            for seq_i in II:
                for seq_j in JJ:
                    D[i,j] += levenshtein(str(MSA_clust[i][seq_i].seq), str(MSA_clust[j][seq_j].seq))
            D[i,j] = D[i,j] / (len(II)*len(JJ))  # normalize

    return D  # average seqeunce distance between


# Predict if a family of proteins is fold-switching or is having a single structure,
# from the co-evolutionary patterns of the MSA
def predict_fold_switch_from_MSA_cluster(MSA, clust_params):

    MSA_clust = cluster_MSAs(MSA, clust_params)  # first cluster the MSAs
    n_clust = len(MSA_clust)  # number of clusters (can be set as a parameter)
    print("Compute sequence similarity:")
    seq_dist = compute_seq_distances(MSA_clust)  # sequence similarity of clusters

    cmap = [None]*n_clust
    for i in range(n_clust):  # loop on clusters
        cmap[i] = MSA_transformer(MSA_clust[i])  # compute pairwise attention map for cluster
    cmap_dist = compute_cmap_distances(cmap)  # similarity of cluster contact maps

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




# Taken from here:
# https://stackoverflow.com/questions/76682974/similarity-score-sequence-alignment
def lev_distance_matrix(seqs):
    """Calculate Levenshtein distance and ratio metrics
       on input pair of strings.
    """
    seqs = sorted(seqs)

    return {
        seqs[0]: {
            seqs[1]: {
                "distance": pylev.distance(*seqs),
                "ratio": pylev.ratio(*seqs),
            }
        }
    }


# Main: test
t_init = time.time()

#cluster_MSAs([], [])

predict_fold_switch_from_MSA_cluster([], [])