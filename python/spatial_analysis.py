import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
import anndata as ad
from tqdm import tqdm
import scipy.stats 
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import KDTree
import multiprocessing
from joblib import Parallel, delayed

# for each cell compute statistics of neighbors within radius
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
#curr_adata = adata_annot[adata_annot.obs.batch==9]
from tqdm import tqdm
def compute_neighborhood_stats(pos, labels, radius=100):
    # record labels as numbers
    labels_quant = LabelEncoder().fit_transform(labels)
    # for each cell, look up neighbors
    kdtree = KDTree(pos)
    nbors_idx, nbors_dist = kdtree.query_radius(pos, r=radius, return_distance=True)
    nbor_stats = np.zeros((pos.shape[0], len(np.unique(labels_quant))))

    for i in tqdm(range(pos.shape[0])):
        curr_nbors_idx = np.sort(nbors_idx[i][nbors_dist[i]>0])#[1:]
        #curr_nbors_dists = nbors_dist[i][np.argsort(nbors_idx[i])]
        curr_nbors_labels = labels_quant[curr_nbors_idx]
        for j in curr_nbors_labels:
            nbor_stats[i,j] += 1
    # zscore across each cluster
    for i in range(nbor_stats.shape[0]):
        nbor_stats[i,:] = zscore(nbor_stats[i,:])
    nbor_stats[np.isinf(nbor_stats)] = 0
    return nbor_stats

def calc_pval(obs, rand, empirical=False):
    if empirical:
        return np.sum(obs <= np.array(rand))/len(rand)
    else:
        z = (obs - np.mean(rand))/np.std(rand)
        return scipy.stats.norm.sf(abs(z))*2

def calc_pval_onesided(obs, rand):
    z = (obs-np.mean(rand))/np.std(rand)
    if z > 0:
        return scipy.stats.norm.sf(abs(z))
    else:
        return 1

def count_nearest_neighbors(X,Y,dist_thresh):
    if X.shape[0] > 0 and Y.shape[0] > 0:
        kdtree = KDTree(Y)
        idx, dists = kdtree.query_radius(X, r=dist_thresh, count_only=False, return_distance=True)
        dists = np.hstack(dists)
        return len(dists[dists>0])
    else:
        return 0


def count_interactions(X,Y, dist_thresh=15):
    X_pos = X.obsm['spatial']#obs[["center_x","center_y"]].values
    Y_pos = Y.obsm['spatial']#obs[["center_x", "center_y"]].values
    return count_nearest_neighbors(X_pos, Y_pos, dist_thresh)

def _jitter_interaction_parallel(X_pos, Y_pos, dist_thresh, perturb_max):
    curr_X = X_pos + np.random.uniform(-perturb_max, perturb_max, (X_pos.shape[0],2))
    curr_Y = Y_pos + np.random.uniform(-perturb_max, perturb_max, (Y_pos.shape[0],2))
    return count_nearest_neighbors(curr_X, curr_Y, dist_thresh) + count_nearest_neighbors(curr_Y, curr_X, dist_thresh)

def score_neighborhood(X,Y, dist_thresh=150, niter=500):
    X_pos = X.obsm['spatial']#.obs[["center_x","center_y"]].values
    Y_pos = Y.obsm['spatial']#.obs[["center_x", "center_y"]].values
    obs_freq = count_nearest_neighbors(X_pos, Y_pos, dist_thresh) + count_nearest_neighbors(Y_pos, X_pos, dist_thresh)
    pvals = np.zeros((niter,))
    num_cores = multiprocessing.cpu_count()
    iterations = tqdm(range(niter))
    random_freq = Parallel(n_jobs=num_cores)(delayed(_jitter_interaction_parallel)(X_pos, Y_pos, dist_thresh, perturb_max) for i in iterations)
    return obs_freq, random_freq, calc_pval(obs_freq, random_freq)

def compare_celltype_interactions(A, B, celltype_key, celltypes=None, niter=1000):
    """
    Compute distributions of celltype interactions between two conditions.
    """
    if celltypes is None:
        celltypes = sorted(A.obs[celltype_key].unique())
    celltype_interactions = np.zeros((len(celltypes), len(celltypes)))
    celltype_pvals = np.zeros((len(celltypes), len(celltypes)))
    for i, c1 in enumerate(celltypes):
        print(c1)
        for j,c2 in enumerate(celltypes):
            obs_freq_A = count_interactions(A[A.obs[celltype_key]==c1], A[A.obs[celltype_key]==c2], dist_thresh=15)
            obs_freq_B = count_interactions(B[B.obs[celltype_key]==c1], B[B.obs[celltype_key]==c2], dist_thresh=15)
            
            combined_obs = np.hstack((obs_freq_A, obs_freq_B))
            obs_labels = np.hstack((np.zeros(len(obs_freq_A)), np.ones(len(obs_freq_B))))
            shuffled_obs = []
            for n in tqdm(range(niter)):
                # shuffle labels
                curr_obs_labels = obs_labels[np.random.permutation(len(obs_labels))] 
                # compute score
                shuffled_obs.append(np.mean(combined_obs[curr_obs_labels==1])/np.mean(combined_obs[curr_obs_labels==0]))
            obs_freq = np.mean(obs_freq_B)/np.mean(obs_freq_A)
            celltype_interactions[i,j] = obs_freq #(obs_freq - np.mean(random_freq))/np.std(random_freq)
            celltype_pvals[i,j] = np.sum(np.abs(obs_freq) < np.abs(shuffled_obs))/niter
    celltype_pvals = celltype_pvals.reshape((len(celltypes)**2,))
    #if len(celltype_pvals)>0:
    #    celltype_pvals = multipletests(celltype_pvals, method='fdr')
    celltype_pvals = celltype_pvals.reshape((len(celltypes), len(celltypes)))
    return celltype_interactions, celltype_pvals

def score_interactions(X,Y, dist_thresh=15, niter=100, perturb_max=50, thresh=10, one_sided=False):
    # compute pairwise distances
    X_pos = X.obsm['spatial']#.obs[["center_x","center_y"]].values
    Y_pos = Y.obsm['spatial']#.obs[["center_x", "center_y"]].values
    obs_freq = count_nearest_neighbors(X_pos, Y_pos, dist_thresh) + count_nearest_neighbors(Y_pos, X_pos, dist_thresh)
    if obs_freq < thresh:
        return obs_freq, 0, 1.0
    pvals = np.zeros((niter,))
    num_cores = multiprocessing.cpu_count()
    iterations = range(niter)
    random_freq = Parallel(n_jobs=num_cores)(delayed(_jitter_interaction_parallel)(X_pos, Y_pos, dist_thresh, perturb_max) for i in iterations)
    if one_sided:
        return obs_freq, random_freq, calc_pval_onesided(obs_freq, random_freq)
    else:
        return obs_freq, random_freq, calc_pval(obs_freq, random_freq)

def compute_celltype_interactions(A, celltype_key, celltypes=None, niter=100, perturb_max=50, dist_thresh=30, min_cells=10, onesided=False):
    import warnings
    warnings.filterwarnings("ignore")
    print("updated!")
    if celltypes is None:
        celltypes = sorted(A.obs[celltype_key].unique())
    celltype_interactions = np.zeros((len(celltypes), len(celltypes)))
    celltype_pvals = np.zeros((len(celltypes), len(celltypes)))
    for i, c1 in enumerate(celltypes):
        print(c1)
        for j,c2 in enumerate(celltypes):
            if i <= j:
                # don't do this for pairs where either has < min_cells
                if np.sum(A.obs[celltype_key]==c1) > min_cells and np.sum(A.obs[celltype_key]==c2) > min_cells:
                    obs_freq, random_freq, pval = score_interactions(A[A.obs[celltype_key]==c1],
                            A[A.obs[celltype_key]==c2], perturb_max=perturb_max, dist_thresh=dist_thresh, niter=niter, one_sided=onesided)
                    print(c1, c2, obs_freq, np.mean(random_freq), obs_freq/np.mean(random_freq), pval)
                    celltype_interactions[i,j] = np.log2(obs_freq/(1e-10+np.mean(random_freq)))#np.log2(obs_freq/np.mean(random_freq))#(obs_freq - np.mean(random_freq))/np.std(random_freq)
                    celltype_interactions[j,i] = celltype_interactions[i,j]#(obs_freq - np.mean(random_freq))/np.std(random_freq)#np.log2(obs_freq/np.mean(random_freq))#

                    celltype_pvals[i,j] = pval
                    celltype_pvals[j,i] = pval
                else:
                    celltype_pvals[i,j] = 1.
                    celltype_pvals[j,i] = 1.
    celltype_pvals = celltype_pvals.reshape((len(celltypes)**2,))
    celltype_qvals = np.zeros_like(celltype_pvals)
    if len(celltype_pvals)>0:
        for i in range(celltype_pvals.shape[0]):
            pass
            #celltype_qvals[i,:] = multipletests(celltype_pvals[i,:], method='fdr_bh')[1]
    celltype_pvals = celltype_pvals.reshape((len(celltypes), len(celltypes)))
    return celltype_interactions, celltype_pvals, celltype_qvals

def _compute_neighborhood(pos, labels, celltypes, radius):
    neighbors = np.zeros((len(celltypes), len(celltypes)))

    for i, c1 in enumerate(celltypes):
        curr_X = pos[labels==c1]
        #print(c1, curr_X.shape[0])
        for j, c2 in enumerate(celltypes):
            curr_Y = pos[labels==c2]
            if i <= j:
                neighbors[i,j] = np.sum(count_nearest_neighbors(curr_X, curr_Y, dist_thresh=radius))#/curr_X.shape[0]
                neighbors[j,i] = neighbors[i,j]
    return neighbors

def _compute_neighbor_shuffled(pos, labels, celltypes, radius):
    labels = labels[np.random.permutation(len(labels))]#[labels[i] for i in np.random.choice(len(labels),len(labels))]
    return _compute_neighborhood(pos, labels, celltypes, radius)

def compute_celltype_neighborhood(A, celltype_key, celltypes=None, radius=150, niter=10):
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
    pos = A.obsm['spatial']
    labels = A.obs[celltype_key]
    neighbors = _compute_neighborhood(pos, labels, celltypes, radius)
    iterations = tqdm(range(niter))
    # for each iteration, shuffle celltype labels
    num_cores = multiprocessing.cpu_count()
    random_freq = Parallel(n_jobs=num_cores)(delayed(_compute_neighbor_shuffled)(pos, labels, celltypes, radius) for i in iterations)    
    #print(len(random_freq))
    # z score
    zs = np.zeros_like(neighbors)
    pval = np.zeros_like(neighbors)

    shuffled_mean = np.dstack(random_freq).mean(2)
    shuffled_std = np.std(np.dstack(random_freq),2)
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            zs[i,j] = (neighbors[i,j] - shuffled_mean[i,j])/shuffled_std[i,j]
            pval[i,j] = calc_pval(neighbors[i,j],  np.dstack(random_freq)[i,j,:])#np.sum(neighbors[i,j] <= np.dstack(random_freq)[i,j,:])/niter#np.sum(neighbors[i,j] <= np.dstack(random_freq)[i,j,:])/niter #calc_pval(neighbors[i,j],  np.dstack(random_freq)[i,j,:])#np.sum(neighbors[i,j] <= np.dstack(random_freq)[i,j,:])/niter
    return neighbors, zs, pval

def _compare_neighborhoods(pos_A, pos_B, labels_A, labels_B, celltypes, radius):
    neighbors_A = _compute_neighborhood(pos_A, labels_A, celltypes, radius)
    neighbors_B = _compute_neighborhood(pos_B, labels_B, celltypes, radius)
    return neighbors_B - neighbors_A

def _compare_neighbor_shuffled(pos_A, pos_B, labels_A, labels_B, celltypes, radius):
    neighbors_A = np.zeros((len(celltypes), len(celltypes)))
    neighbors_B = np.zeros((len(celltypes), len(celltypes)))

    for i, c1 in enumerate(celltypes):
        curr_X_A = pos_A[labels_A==c1]
        curr_X_B = pos_B[labels_B==c1]
        for j, c2 in enumerate(celltypes):
            curr_Y_A = pos_A[labels_A==c2]
            curr_Y_B = pos_B[labels_B==c2]
            if i <= j:
                # make vector of label identities
                label_idents = np.hstack((np.zeros(curr_X_A.shape[0]), np.ones(curr_X_B.shape[0])))
                label_idents = label_idents[np.random.permutation(len(label_idents))]
                nn_A = count_nearest_neighbors(curr_X_A, curr_Y_A, dist_thresh=radius)
                nn_B = count_nearest_neighbors(curr_X_B, curr_Y_B, dist_thresh=radius)
                # shuffle which cells came from which identity
                combined_neighbors = np.hstack((nn_A, nn_B))
                neighbors_A[i,j] = np.sum(combined_neighbors[label_idents==0])
                neighbors_A[j,i] = neighbors_A[i,j]
                neighbors_B[i,j] = np.sum(combined_neighbors[label_idents==1])
                neighbors_B[j,i] = neighbors_B[i,j]
    return neighbors_B - neighbors_A
 
def compare_celltype_neighborbood(A, B, celltype_key, celltypes=None, radius=150, niter=10):
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
        
    pos_A = A.obsm['spatial']
    labels_A = A.obs[celltype_key]
    pos_B = B.obsm['spatial']
    labels_B = B.obs[celltype_key]
    #_compare_neighbor_shuffled(pos_A, pos_B, labels_A, labels_B, celltypes, radius)
    neighbors_A = _compute_neighborhood(pos_A, labels_A, celltypes, radius)
    neighbors_B = _compute_neighborhood(pos_B, labels_B, celltypes, radius)
    neighbor_diff = neighbors_B - neighbors_A
    iterations = tqdm(range(niter))
    # for each iteration, shuffle celltype labels
    num_cores = multiprocessing.cpu_count()
    random_freq = Parallel(n_jobs=num_cores)(delayed(_compare_neighbor_shuffled)(pos_A, pos_B, labels_A, labels_B, celltypes, radius) for i in iterations)    
    print(len(random_freq))
    # z score
    zs = np.zeros_like(neighbor_diff)
    pval = np.zeros_like(neighbor_diff)

    shuffled_mean = np.dstack(random_freq).mean(2)
    shuffled_std = np.std(np.dstack(random_freq),2)
    for i in range(neighbor_diff.shape[0]):
        for j in range(neighbor_diff.shape[1]):
            zs[i,j] = (neighbor_diff[i,j] - shuffled_mean[i,j])/shuffled_std[i,j]
            pval[i,j] = np.sum(np.abs(neighbor_diff[i,j]) <= np.abs(np.dstack(random_freq)[i,j,:]))/niter
    return neighbor_diff, zs, pval, random_freq

# algorithm:
# - for each cell - cell pair
#   - select all neighbors of a cell 
#   - compute average expression of all genes for neighbors
#   - compute average expression for all cells that aren't neighbors
#   - find difference
# - shuffle neighbor/not neighbor identities
def identify_nearest_neighbors(X,Y,dist_thresh, min_dist_thresh=0):
    """
    Find all the elements in Y that are neighbors of X.
    min_dist_thresh is to avoid contamination of stray counts from exactly neighboring cells
    """
    if X.shape[0] > 0 and Y.shape[0] > 0:
        kdtree = KDTree(Y)
        ind, dists = kdtree.query_radius(X, r=dist_thresh, count_only=False,return_distance=True)
        ind = np.hstack(ind)
        dists = np.hstack(dists)
        if len(ind) > 0:
            ind = ind[dists>min_dist_thresh]            
        return np.unique(ind)
    else:
        return np.array([])

def get_nearest_neighbor_dists(X,Y):
    kdtree = KDTree(Y)
    dist, idx = kdtree.query(X, k=1)
    return dist, idx


def _compute_neighborhood_expr(pos, expr, labels, celltypes, radius):
    expr_diff = np.zeros((len(celltypes), len(celltypes), expr.shape[1]))
    for i, c1 in enumerate(celltypes):
        curr_X = pos[labels==c1]
        print(c1, curr_X.shape[0])
        for j, c2 in enumerate(celltypes):
            if i != j:
                curr_Y = pos[labels==c2]
                neighbors_X = identify_nearest_neighbors(curr_X, curr_Y, dist_thresh=radius).astype(np.int)
                #print
                #print(curr_X.shape)
                #print(neighbors_X.shape)
                not_neighbors_X = np.array([i for i in np.arange(curr_X.shape[0]).astype(np.int) if i not in neighbors_X])
                #print(neighbors_X)
                #print(not_neighbors_X)
                #print(c1, c2, )
                expr_diff[i,j,:] = expr[neighbors_X,:].mean(0)/expr[not_neighbors_X,:].mean(0) #/curr_X.shape[0]
    return expr_diff

def _compute_neighbor_shuffled_expr(pos, expr, labels, celltypes, radius):
    # shuffle label
    expr_diff = np.zeros((len(celltypes), len(celltypes), expr.shape[1]))
    for i, c1 in enumerate(celltypes):
        curr_X = pos[labels==c1]
        for j, c2 in enumerate(celltypes):
            if i != j:
                curr_Y = pos[labels==c2]
                neighbors_X = identify_nearest_neighbors(curr_X, curr_Y, dist_thresh=radius)
                n_neighbors = len(neighbors_X)
                #print
                #print(curr_X.shape)
                #print(neighbors_X.shape)
                not_neighbors_X = np.array([i for i in np.arange(curr_X.shape[0]) if i not in neighbors_X])
                n_not_neighbors = len(not_neighbors_X)

                #print(neighbors_X)
                #print(not_neighbors_X)
                combined_neighbors = np.hstack((neighbors_X, not_neighbors_X))
                combined_neighbors = combined_neighbors[np.random.permutation(len(combined_neighbors))]
                neighbors_X = combined_neighbors[:n_neighbors]
                not_neighbors_X = combined_neighbors[n_neighbors:]
                expr_diff[i,j,:] = expr[neighbors_X,:].mean(0) - expr[not_neighbors_X,:].mean(0) #/curr_X.shape[0]
    return expr_diff

def bootstrap_expr_diff(X,Y,n=1000):
    combined_data = np.concatenate((X,Y))
    idx = np.concatenate((np.zeros(len(X)), np.ones(len(Y))))
    obs_diff = np.mean(X) - np.mean(Y)
    shuffle_diffs = []
    for i in range(n):
        shuffled_idx = idx[np.random.permutation(len(idx))]
        curr_X = combined_data[shuffled_idx==0]
        curr_Y = combined_data[shuffled_idx==1]
        shuffle_diffs.append(np.mean(curr_X)-np.mean(curr_Y))
    return obs_diff, np.sum(obs_diff <= np.array(shuffle_diffs))/n #calc_pval(obs_diff, np.array(shuffle_diffs))

def compute_celltype_neighborhood_regression(A, celltype_key, source, celltypes=None,min_radiu=0, obs_keys=None):
    if obs_keys is None:
        expr = A.X
    else:
        expr = np.array(A.obs.loc[:,obs_keys].values)
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
    pos = A.obsm['spatial']
    labels = A.obs[celltype_key]
    tstats = np.zeros((len(celltypes), expr.shape[1]))
    pvals = np.zeros((len(celltypes), expr.shape[1]))
    # get all the cells of a certain type
    curr_X = pos[labels==source]
    curr_expr = expr[labels==source]
    interactions = {}
    for i, c1 in enumerate(celltypes):
        # find all the cells of the neighboring type
        curr_Y = pos[labels==c1]
        # identify neighbors of target cell type X to cells in cell type Y
        dists, idx = get_nearest_neighbor_dists(curr_Y, curr_X)
        interactions[c1] = (dists, curr_expr[idx])
    return interactions

import scipy
def compute_celltype_neighborhood_ttest_single(A, celltype_key, source, celltypes=None, min_radius=15, radius=150, far_radius=250, niter=500, obs_keys=None,use_ttest=False,spatial_jitter=False):
    if obs_keys is None:
        expr = A.X
    else:
        expr = np.array(A.obs.loc[:,obs_keys].values)
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
    pos = A.obsm['spatial']
    labels = A.obs[celltype_key]
    tstats = np.zeros((len(celltypes), expr.shape[1]))
    pvals = np.zeros((len(celltypes), expr.shape[1]))
    # get all the cells of a certain type
    curr_X = pos[labels==source]
    curr_expr = expr[labels==source]
    for i, c1 in enumerate(celltypes):
        # find all the cells of the neighboring type
        curr_Y = pos[labels==c1]
        # identify neighbors of target cell type X to cells in cell type Y
        neighbors_X = identify_nearest_neighbors(curr_Y, curr_X, dist_thresh=radius, min_dist_thresh=min_radius).astype(np.int)
        far_neighbors_X = identify_nearest_neighbors(curr_Y, curr_X, dist_thresh=far_radius, min_dist_thresh=radius).astype(np.int)
        not_neighbors_X = np.array([i for i in far_neighbors_X if i not in neighbors_X])

        #neighbors_X = identify_nearest_neighbors(curr_Y, curr_X, dist_thresh=radius, min_dist_thresh=min_radius).astype(np.int)
        #not_neighbors_X = np.array([i for i in np.arange(curr_X.shape[0]).astype(np.int) if i not in neighbors_X])
        # shuffle what is a neighbor vs what isn't a neighbor
        if not spatial_jitter:
            if len(neighbors_X) > 0 and len(not_neighbors_X) > 0:
                mean_nbor = np.mean(curr_expr[neighbors_X])
                mean_not_nbor = np.mean(curr_expr[not_neighbors_X])
                print("X=%s, Y=%s, curr_X=%d, curr_Y=%d, nbor_X=%d, not_nbor_X=%d, mean_nbor_X=%0.04f, mean_not_nbor_X=%0.04f" % (c1, source, curr_X.shape[0], curr_Y.shape[0], len(neighbors_X), len(not_neighbors_X), mean_nbor, mean_not_nbor)) 
                if use_ttest:
                    ttest = scipy.stats.ttest_ind(curr_expr[neighbors_X], curr_expr[not_neighbors_X])
                else:
                    ttest = bootstrap_expr_diff(curr_expr[neighbors_X], curr_expr[not_neighbors_X])#
                tstats[i] =  ttest[0]#np.log2(np.mean(curr_expr[neighbors_X])/np.mean(curr_expr[not_neighbors_X]))#ttest[0]
                pvals[i] = ttest[1]#/curr_X.shape[0]
            else:
                pvals[i] = 1
                tstats[i] = 0
        else:
            pass
            # jitter cells in space and, then compute gene expression distribution
    return tstats, pvals


def compute_celltype_neighborhood_ttest(A, celltype_key, celltypes=None, min_radius=15, radius=150, far_radius=150, niter=500, obs_keys=None):
    if obs_keys is None:
        expr = A.X
    else:
        expr = np.array(A.obs.loc[:,obs_keys].values)
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
    pos = A.obsm['spatial']
    labels = A.obs[celltype_key]
    tstats = np.zeros((len(celltypes), len(celltypes), expr.shape[1]))
    pvals = np.zeros((len(celltypes), len(celltypes), expr.shape[1]))
    for i, c1 in enumerate(celltypes):
        curr_X = pos[labels==c1]
        print(c1, curr_X.shape[0])
        for j, c2 in enumerate(celltypes):
           # if i != j:
            curr_Y = pos[labels==c2]
            curr_expr = expr[labels==c2,:]
            # neighbors_X indexes into Y
            neighbors_X = identify_nearest_neighbors(curr_X, curr_Y, dist_thresh=radius, min_dist_thresh=min_radius).astype(np.int)
            far_neighbors_X = identify_nearest_neighbors(curr_X, curr_Y, dist_thresh=far_radius, min_dist_thresh=radius).astype(np.int)
            #print(curr_X.shape[0], curr_Y.shape[0], neighbors_X.max())
            #print
            #print(curr_X.shape)
            #print(neighbors_X.shape)
            not_neighbors_X = np.array([i for i in far_neighbors_X if i not in neighbors_X])
            #print(curr_expr.shape[0])
            if len(neighbors_X) > 0 and len(not_neighbors_X) > 0:
                print("X=%s, Y=%s, curr_X=%d, curr_Y=%d, nbor_X=%d, not_nbor_X=%d, max_nbor_X=%d, max_not_nbor_X=%d" % (c1, c2, curr_X.shape[0], curr_Y.shape[0], len(neighbors_X), len(not_neighbors_X), neighbors_X.max(), not_neighbors_X.max()))
                #print(neighbors_X)
                #print(not_neighbors_X)
                #print(c1, c2, )
                for k in range(expr.shape[1]):
                    ttest = scipy.stats.ttest_ind(curr_expr[neighbors_X,:][:,k], curr_expr[not_neighbors_X,:][:,k])
                    tstats[i,j,k] =  ttest[0]
                    pvals[i,j,k] = ttest[1]#/curr_X.shape[0]
            else:
                pvals[i,j,:] = 1
                tstats[i,j,:] = 0
            if i == j:
                pvals[i,j,:] = 1
                tstats[i,j,:] = 0
    return tstats, pvals

def compute_celltype_neighborhood_expr(A, celltype_key, celltypes=None, radius=150, niter=500):
    expr = A.X
    if celltypes is None:
        celltypes = list(sorted(A.obs[celltype_key].unique()))
    pos = A.obsm['spatial']
    labels = A.obs[celltype_key]
    expr_diff = _compute_neighborhood_expr(pos, expr, labels, celltypes, radius)
    iterations = tqdm(range(niter))
    # for each iteration, shuffle celltype labels
    num_cores = multiprocessing.cpu_count()
    # random_freq is niter x n_celltype x n_celltype x n_gene matrix
    #random_freq = np.stack(Parallel(n_jobs=num_cores)(delayed(_compute_neighbor_shuffled_expr)(pos, expr, labels, celltypes, radius) for i in iterations))
    # z score
    zs = np.zeros_like(expr_diff)
    pval = np.zeros_like(expr_diff)
   
    #shuffled_mean = random_freq.mean(0)
    #shuffled_std = np.std(random_freq,0)
    #for i in range(expr_diff.shape[0]):
    #    for j in range(expr_diff.shape[1]):
    #        for k in range(expr_diff.shape[2]):
    #            zs[i,j,k] = (expr_diff[i,j,k] - shuffled_mean[i,j,k])/shuffled_std[i,j,k]
    #            pval[i,j,k] = scipy.stats.norm.sf(abs(zs[i,j,k]))*2 #calc_pval(expr_diff[i,j,k],  random_freq[:,i,j,k])#np.sum(neighbors[i,j] <= np.dstack(random_freq)[i,j,:])/niter
    return expr_diff, zs, pval

def quantify_clust_spatial_enrichment(A,uniq_clusts=None,clust_key='clust_annot', normalize=True):
    if uniq_clusts is None:
        uniq_clusts = sorted(A.obs[clust_key].unique())
    n_spatial_domains = A.obs.spatial_clust_annots_value.max() + 1
    clust_counts = np.zeros((n_spatial_domains, len(uniq_clusts)))
    print(clust_counts.shape)
    for i in range(n_spatial_domains):
        curr_clusts = A[A.obs.spatial_clust_annots_value==i,:].obs[clust_key]
        for j,c in enumerate(uniq_clusts):
            clust_counts[i,j] = np.sum(curr_clusts==c)
    clust_avgs = clust_counts.copy()
    for i in range(clust_avgs.shape[0]):
        clust_avgs[i,:] /= np.sum(A.obs.spatial_clust_annots_value==i)#clust_avgs[i,:].sum()
    return clust_counts, clust_avgs
