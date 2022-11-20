import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import diffxpy.api as de

from sklearn.inspection import permutation_importance
def select_age_features(A: sc.AnnData, grouping, Nfeats=500, test_size=0.2):
    """
    Use RFClassifier to find important aging features
    """
    scores = {}
    feats = {}
    for i in np.unique(A.obs[grouping]):
        print(i)
        curr_adata = A[A.obs[grouping]==i]
        X = curr_adata.X.copy()
        y = curr_adata.obs.age
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = RandomForestClassifier(verbose=False, n_jobs=-1).fit(X_train, y_train)
        y_preds = clf.predict_proba(X_test)
        scores[i] = roc_auc_score(y_test, y_preds[:,1])
        feats[i] = pd.DataFrame({'clust': [i]*Nfeats,
                                          'importance': np.sort(clf.feature_importances_)[::-1][:Nfeats],
                                          'feats': curr_adata.var_names[np.argsort(clf.feature_importances_)[::-1][:Nfeats_age]]})
    return scores, feats

def select_celltype_features(A: sc.AnnData, grouping,Nfeats=1000, test_size=0.2):
    """
    Use RFClassifier to find important cell type distinguishing features.
    """
    scores = {}
    feats = {}
    X = A.X.copy()
    for i in np.unique(A.obs[grouping]):
        print(i)
        # train on one vs rest
        y = A.obs[grouping]==i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = RandomForestClassifier(verbose=False, n_jobs=-1).fit(X_train, y_train)
        y_preds = clf.predict_proba(X_test)
        scores[i] = roc_auc_score(y_test, y_preds[:,1])
        feats[i] = pd.DataFrame({'clust': [i]*Nfeats,
                                          'importance': np.sort(clf.feature_importances_)[::-1][:Nfeats],
                                          'feats': A.var_names[np.argsort(clf.feature_importances_)[::-1][:Nfeats]]})
    return scores, feats

def select_celltype_features_perm(A: sc.AnnData, grouping,Nfeats=1000, test_size=0.2, n_repeats=10):
    scores = {}
    feats = {}
    X = A.X.toarray().copy()
    for i in np.unique(A.obs[grouping]):
        print(i)
        # train on one vs rest
        y = A.obs[grouping]==i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = RandomForestClassifier(verbose=False, n_jobs=-1).fit(X_train, y_train)
        y_preds = clf.predict_proba(X_test)
        result = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        scores[i] = roc_auc_score(y_test, y_preds[:,1])
        feats[i] = pd.DataFrame({'clust': [i]*Nfeats,
                                          'importance': np.sort(result.importances_mean)[::-1][:Nfeats],
                                          'feats': A.var_names[np.argsort(result.importances_mean)[::-1][:Nfeats]]})
    return scores, feats

from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hc
import matplotlib

def plot_clustered_celltypes_by_genes(A: sc.AnnData, genes, normalize=True, figsize=(20,30)):

    marker_clust_avgs = []
    clust_avgs = []
    for i in A.obs.clust_label.unique():
        clust_avgs.append(compute_mean_expression(A[A.obs.clust_label==i,:]))
        marker_clust_avgs.append(compute_mean_expression(A[A.obs.clust_label==i,:][:,genes]))

    D = pdist(np.vstack(marker_clust_avgs).T, 'euclidean')
    Z = hc.linkage(D, 'ward', optimal_ordering=True)
    gene_ordering = hc.leaves_list(Z)

    D = pdist(clust_avgs, 'euclidean')
    Z = hc.linkage(D, 'ward', optimal_ordering=True)
    clust_ordering = hc.leaves_list(Z)
#plt.imshow(np.corrcoef(clust_avgs)[clust_ordering],vmin=0,vmax=1,cmap=plt.cm.viridis)
    matplotlib.rcParams.update({'font.size': 8})
    if normalize:
        sc.pl.heatmap(A, np.array(list(genes))[gene_ordering], 'clust_label', show_gene_labels=True, dendrogram=True,standard_scale='obs',figsize=figsize)
    else:
        sc.pl.heatmap(A, np.array(list(genes))[gene_ordering], 'clust_label', show_gene_labels=True, dendrogram=True,figsize=figsize)

def plot_clustered_ages_by_genes(A: sc.AnnData, genes, normalize=True, figsize=(20,30)):

    marker_clust_avgs = []
    clust_avgs = []
    clust_names = A.obs.clust_label.unique()
    for i in A.obs.clust_label.unique():
        clust_avgs.append(compute_mean_expression(A[A.obs.clust_label==i,:]))
        marker_clust_avgs.append(compute_mean_expression(A[np.logical_and(A.obs.clust_label==i, A.obs.age=='4wk'),:][:,genes])-
                         compute_mean_expression(A[np.logical_and(A.obs.clust_label==i, A.obs.age=='90wk'),:][:,genes]))

    marker_clust_avgs = np.vstack(marker_clust_avgs)
    D = pdist(marker_clust_avgs.T, 'euclidean')
    Z = hc.linkage(D, 'ward', optimal_ordering=True)
    gene_ordering = hc.leaves_list(Z)

    D = pdist(clust_avgs, 'euclidean')
    Z = hc.linkage(D, 'ward', optimal_ordering=True)
    clust_ordering = hc.leaves_list(Z)
    
    plt.figure(figsize=(20,10))
    plt.imshow(marker_clust_avgs[:, gene_ordering][clust_ordering,:],vmin=-2,vmax=2,cmap=plt.cm.bwr,aspect='auto',interpolation='none')
    plt.yticks(np.arange(marker_clust_avgs.shape[0]))
    plt.xticks(np.arange(marker_clust_avgs.shape[1]))
    plt.axes().set_xticklabels(np.array(genes)[gene_ordering],rotation=90);
    plt.axes().set_yticklabels(np.array(clust_names)[clust_ordering])
    plt.grid(False)

def compute_cluster_proportions(A: sc.AnnData, obs_type='clust_label'):
    """ Compute the fraction of cells in each cluster """
    clusts = A.obs[obs_type].unique()
    clust_proportions = np.zeros((len(clusts),1))
    for k,i in enumerate(clusts):
        clust_proportions[k] = np.sum(A.obs[obs_type]==i)/A.shape[0]
    return clust_proportions, clusts

def compute_average_celltype_expr(A: sc.AnnData, genes, obs_type="clust_label"):
    marker_clust_avgs = []
    for i in A.obs[obs_type].unique():
        marker_clust_avgs.append(compute_mean_expression(A[A.obs[obs_type]==i,:][:,genes]))
    return np.vstack(marker_clust_avgs), A.obs[obs_type].unique()

def compute_average_age_expr_change(A: sc.AnnData, genes):
    marker_clust_avgs = []
    for i in A.obs.clust_label.unique():
        marker_clust_avgs.append(compute_mean_expression(A[np.logical_and(A.obs.clust_label==i, A.obs.age=='4wk'),:][:,genes])-
                         compute_mean_expression(A[np.logical_and(A.obs.clust_label==i, A.obs.age=='90wk'),:][:,genes]))
    return np.vstack(marker_clust_avgs)

# compute per cluster average sparsity
def plot_per_celltype_sparsity(A: sc.AnnData, genes):
    sparsity = []
    celltypes = []
    for i in A.obs.cell_type.unique():
        curr_adata = A[A.obs.cell_type==i][:, genes]
        frac_expr = compute_frac_expressed(curr_adata)
        sparsity.extend(frac_expr)
        celltypes.extend([i]*len(frac_expr))
    sns.swarmplot(data=pd.DataFrame({'clust': celltypes, 'sparsity':sparsity}), 
                  x='clust',
                  y='sparsity')
    
def plot_per_celltype_totalexpr(A: sc.AnnData, genes, exp=False):
    expr = []
    celltypes = []
    for i in A.obs.cell_type.unique():
        curr_adata = A[A.obs.cell_type==i][:, genes]
        total_expr = np.array(curr_adata.X.sum(1)).flatten()
        expr.extend(total_expr)
        celltypes.extend([i]*len(total_expr))
    sns.violinplot(data=pd.DataFrame({'clust': celltypes, 'expr':expr}), 
                  x='clust',
                  y='expr')
    #return pd.DataFrame({'clust': celltypes, 'expr':expr})

    
def plot_per_gene_sparsity(A: sc.AnnData, genes):
    """
    Score each gene by the max fraction expression divided by the average across all clusters.
    """
    sparsity = []
    celltypes = []
    for i in A.obs.cell_type.unique():
        curr_adata = A[A.obs.cell_type==i][:, genes]
        frac_expr = compute_frac_expressed(curr_adata)
        sparsity.append(frac_expr)
        #celltypes.extend([i]*len(frac_expr))
    temp = np.vstack(sparsity)
    sparsity_score = temp.mean(0)/temp.max(0)
    sort_idx = np.argsort(sparsity_score)
    plt.figure(figsize=(15,5))
    plt.scatter(np.arange(temp.shape[1]), sparsity_score[sort_idx])
    plt.xticks(np.arange(temp.shape[1]));
    plt.axes().grid(False)
    plt.axes().set_xticklabels(np.array(genes)[sort_idx],rotation=90,fontsize=6);

def compute_mean_expression(A: sc.AnnData):
    """
    Average expression for each gene
    """
    return np.array(A.X.mean(0)).flatten()

def compute_frac_expressed(A: sc.AnnData):
    """
    Fraction of cells expressing each gene
    """
    return np.array((A.X>0).sum(0)/A.shape[0]).flatten()

def filter_2group_1way(A: sc.AnnData, obs_name: str, ident: str, min_pct=None, logfc_thresh=None, min_diff_pct=None, max_cells_per_ident=None, log=True):
    """
    Filter genes before differential expression testing. UNIDIRECTIONAL
    obs is grouping
    ident is what to be compared (ident vs ~ident)
    min_diff_pcr: minimum difference in percentage between genes
    log: is the data log transformed (usually this is the case)
    """
    n_cells, n_genes = A.shape
    X = A[A.obs[obs_name] == ident]
    Y = A[A.obs[obs_name] != ident]
    
    min_pct_mask = np.ones((n_genes,),dtype=np.bool)
    log_fc_mask = np.ones((n_genes,),dtype=np.bool)
    min_diff_pct_mask = np.ones((n_genes,),dtype=np.bool)
    
    pct_X = compute_frac_expressed(X)
    pct_Y = compute_frac_expressed(Y)

    if min_pct:
        min_pct_mask = (pct_X>min_pct).flatten()
        
    mean_X = compute_mean_expression(X)
    mean_Y = compute_mean_expression(Y)
    if log:
        logfc_XY = np.log(np.exp(mean_X)/np.exp(mean_Y))
    else:
        logfc_XY = np.log(mean_X/mean_Y)
      
    if logfc_thresh:
        log_fc_mask = (logfc_XY > logfc_thresh).flatten()
    
    if min_diff_pct:
        diff_pct_XY = pct_X-pct_Y
        min_diff_pct_mask = (diff_pct_XY > min_diff_pct).flatten()
    final_mask = np.logical_and(np.logical_and(min_pct_mask, log_fc_mask), min_diff_pct_mask).flatten()
    A = A[:, final_mask]
    
    if max_cells_per_ident:
        idx_X = np.nonzero((A.obs[obs_name]==ident).values)[0]
        idx_Y = np.nonzero((A.obs[obs_name]!=ident).values)[0]
        ids_X = idx_X[np.random.permutation(len(idx_X))[:max_cells_per_ident]]
        ids_Y = idx_Y[np.random.permutation(len(idx_Y))[:max_cells_per_ident]]
        combined_ids = np.hstack((ids_X, ids_Y)).flatten()
        return A[combined_ids,:], logfc_XY[np.array(final_mask).flatten()]
    else:
        
        return A, logfc_XY[np.array(final_mask).flatten()]
    

def compute_onevsall_de_for_clusts(A: sc.AnnData, clust_obs, n_de=5):
    clust_labels_uniq = list(np.unique(A.obs[clust_obs]))

    de_by_type = {}
    for n,i in enumerate(clust_labels_uniq):
        print(n+1,'/',len(clust_labels_uniq),':',i)
        curr_A = A[np.logical_or(A.obs[clust_obs]==i, A.obs[clust_obs]!=i)].copy()
        curr_A.obs['contrast'] = curr_A.obs[clust_obs]==i
        curr_A, _ = filter_2group_1way(curr_A, 'contrast', True, min_pct=0.2, logfc_thresh=np.log(1.5))
        res = de.test.t_test(data=curr_A, grouping='contrast')

        frac_foreground = compute_frac_expressed(curr_A[curr_A.obs[clust_obs]==i])
        frac_background = compute_frac_expressed(curr_A[curr_A.obs[clust_obs]!=i])
        # filter genes
        good_expr = np.logical_and(res.log10_fold_change()>=np.log10(2), res.qval<0.05)
        good_frac = np.logical_and(frac_foreground>0.4, frac_foreground>3*frac_background)
        # require 
        good_genes = np.logical_and(good_expr, good_frac)
        sort_idx = np.argsort(res.qval[good_genes])
        log10fc = res.log10_fold_change()[good_genes][sort_idx]
        de_by_type[i] = pd.DataFrame({'gene':res.gene_ids[good_genes][sort_idx][:n_de],
                                          'log10fc':log10fc[:n_de],
                                          'frac_fg':frac_foreground[good_genes][sort_idx][:n_de],
                                          'frac_bg':frac_background[good_genes][sort_idx][:n_de],
                                          'qval':res.qval[good_genes][sort_idx][:n_de]})
    return de_by_type

def select_age_markers(de_map, n_marker_genes=10):
    """
    Select top n_marker_genes for each cluster. log10fc is absolute value of coefficient.
    """
    clust_labels_uniq = list(de_map.keys())
    de_marker_genes = set()
    for n,i in enumerate(clust_labels_uniq):
        curr_contrast = de_map[i].sort_values('log10fc', ascending=False)
        for g in list(curr_contrast.head(n_marker_genes).gene):
            de_marker_genes.add(g)
    return de_marker_genes
        
def greedily_select_markers(de_map, min_marker_genes=4, pairwise=True, n_pass=10, de_marker_genes=None):
    if pairwise:
        clust_labels_uniq = list(set([i[0] for i in de_map.keys()]))
    else:
        clust_labels_uniq = list(de_map.keys())
    if de_marker_genes is None:
        de_marker_genes = set()
    else:
        de_marker_genes = set(de_marker_genes)
    # do n passes through the list, to ensure that all clusters have at least min_marker_genes included
    # in the set

    all_clusts_good = True
    for n in range(n_pass):
        if n > 0 and all_clusts_good:
            break
        else:
            all_clusts_good = True
            for n,i in enumerate(clust_labels_uniq):
                #print(n+1,'/',len(clust_labels_uniq),':',i)
                if pairwise:
                    for j in clust_labels_uniq:
                        if i != j:
                            curr_contrast = de_map[(i,j)].sort_values('log10fc', ascending=False)
                            curr_genes = list(curr_contrast.gene)
                            # check if has enough genes in this pair
                            if len(curr_genes) > 0:
                                # check how many of these genes are included in the working set of marker genes
                                n_curr_marker = np.sum([k in de_marker_genes for k in curr_genes])
                                # if this cluster has no markers in the marker gene set, add the remaining number
                                if n_curr_marker < min_marker_genes:
                                    all_clusts_good = False
                                    n_to_add = min(len(curr_genes), int(min_marker_genes-n_curr_marker))
                                    print("Adding", n_to_add)
                                    curr_to_add = [i for i in curr_genes if i not in de_marker_genes]
                                    for k in range(min(n_to_add, len(curr_to_add))):
                                        de_marker_genes.add(curr_to_add[k])
                else:
                    curr_contrast = de_map[i].sort_values('log10fc', ascending=False)
                    curr_genes = list(curr_contrast.gene)
                    if len(curr_genes) > 0:
                        # check how many of these genes are included in the working set of marker genes
                        n_curr_marker = np.sum([k in de_marker_genes for k in curr_genes])
                        # if this cluster has no markers in the marker gene set, add the remaining number
                        if n_curr_marker < min_marker_genes:
                            n_to_add = min(len(curr_genes), int(min_marker_genes-n_curr_marker))
                            for k in range(n_to_add):
                                de_marker_genes.add(curr_genes[k])

    return de_marker_genes

def compute_pairwise_de_for_clusts(A: sc.AnnData, clust_obs, n_de=5, min_pct=0.4):
    clust_labels_uniq = list(np.unique(A.obs[clust_obs]))

    pairwise_de = {}
    for n,i in enumerate(clust_labels_uniq):
        print(n+1,'/',len(clust_labels_uniq),':',i)
        for j in tqdm(clust_labels_uniq):
            if i != j:
                curr_A = A[np.logical_or(A.obs[clust_obs]==i, A.obs[clust_obs]==j)].copy()
                curr_A.obs['contrast'] = curr_A.obs[clust_obs]==i
                curr_A, _ = filter_2group_1way(curr_A, 'contrast', True, min_pct=min_pct, logfc_thresh=np.log(1.5))
                res = de.test.t_test(data=curr_A, grouping='contrast')

                frac_foreground = compute_frac_expressed(curr_A[curr_A.obs[clust_obs]==i])
                frac_background = compute_frac_expressed(curr_A[curr_A.obs[clust_obs]==j])
                # filter genes
                good_expr = np.logical_and(res.log10_fold_change()>=np.log10(2), res.qval<0.05)
                good_frac = np.logical_and(frac_foreground>0.4, frac_foreground>3*frac_background)
                good_genes = np.logical_and(good_expr, good_frac)
                sort_idx = np.argsort(res.qval[good_genes])
                log10fc = res.log10_fold_change()[good_genes][sort_idx]
                pairwise_de[(i,j)] = pd.DataFrame({'gene':res.gene_ids[good_genes][sort_idx][:n_de],
                                                  'log10fc':log10fc[:n_de],
                                                  'frac_fg':frac_foreground[good_genes][sort_idx][:n_de],
                                                  'frac_bg':frac_background[good_genes][sort_idx][:n_de],
                                                  'qval':res.qval[good_genes][sort_idx][:n_de]})
    
    # greedily select marker genes based on differential expression ranking
    return pairwise_de