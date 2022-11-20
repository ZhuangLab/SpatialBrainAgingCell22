from statsmodels.stats.multitest import multipletests
import numpy as np

def fdr_correct(X):
    new_X = np.zeros_like(X)
    for i in range(X.shape[-1]):
        pvals = multipletests(X[i,:],method='fdr_bh')[0]
        new_X[i,:] = multipletests(X[i,:],method='fdr_bh')[0]
        new_X[:,i] = new_X[i,:]
    #X = multipletests(X.flatten(), method='fdr_bh')[0]
    return new_X#X.reshape(X_shape)

from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hc


def order_values(X, metric='correlation', return_linkage=False):
    D = pdist(X,metric)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    Z = hc.linkage(D,'complete',optimal_ordering=True)
    dn = hc.dendrogram(Z,no_plot=True)
    if not return_linkage:
        return np.array(dn['leaves'])
    else:
        return np.array(dn['leaves']), Z

def relabel_clust(A, orig_clust, new_clust,key="clust_annot"):
    clusts = np.array(list(A.obs[key]))
    clusts[clusts==orig_clust] = new_clust
    A.obs[key] = list(clusts)
    return A

def relabel_anatomy(A, annot_old, annot_new):
    A = relabel_clust(A, annot_old, annot_new, key='spatial_clust_annots')
    spatial_clust_annots_values = {
        'Pia' : 0,
        'Cortex':1,
        'LatSept':2,
        'CC':3,
        'Striatum':4,
        'Ventricle':5
        }
    A.obs['spatial_clust_annots_value'] = [spatial_clust_annots_values[i] if i in spatial_clust_annots_values else None for i in A.obs.spatial_clust_annots]
    return A

def relabel_all_clusts(A, clust_mapping,key='clust_annot'):
    old_clust_annots = np.array(A.obs[key].copy())
    new_clust_annots = np.array(old_clust_annots.copy())
    for k,v in clust_mapping.items():
        new_clust_annots[old_clust_annots==k] = v
    A.obs[key] = list(new_clust_annots.copy())
    return A

def cleanup_section(A_section,n_neighbors=25):
    np.random.seed(31415)
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=-1,n_neighbors=n_neighbors,weights='uniform').fit(A_section.obsm['spatial'],A_section.obs.spatial_clust_annots_value)
    A_section.obs['smoothed_spatial_clust_annot_values'] = list(clf.predict(A_section.obsm['spatial']))
    return A_section