import seaborn as sns
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
def gen_light_palette(prefix, color_name, uniq_clusts):
    n = np.sum([1 if prefix in i else 0 for i in uniq_clusts])
    return sns.light_palette(color_name, n_colors=n+2)[2:]

def gen_dark_palette(prefix, color_name, uniq_clusts):
    n = np.sum([1 if prefix in i else 0 for i in uniq_clusts])
    return sns.dark_palette(color_name, n_colors=n+2)[2:]

major_cell_types = {
    # Astro -- green
    "Astro" : "seagreen",
    # Excitatory -- red/orange
    "ExN" : "lightcoral",
    # inhibitory -- blue/purple
    "InN" : "cornflowerblue",
    "MSN" : "mediumpurple",
    
    # immune cells + microglia -- pink
    "Micro" : "pink",
    "T cell" : "deeppink",
    "Macro" : "violet",
    
    # Endothelial/vasculure -- gold/tan
    "Vlmc" : "gold",
    "Endo" : "khaki",
    "Peri" : "goldenrod",
    "Epen" : "darkkhaki",
    
    # Oligodendrocytes
    "Olig" : "slategrey",
    "OPC" : "black"
}

clust_cell_types = {
    # Astro -- green
    "Astro" : "seagreen",
    # Excitatory -- red/orange
    "ExN-L2/3" : "darkorange",
    "ExN-L5" : "lightsalmon",
    "ExN-L6" : "maroon",
    "ExN-Olf" : "firebrick",
    # inhibitory -- blue/purple
    'InN-Olf' : "cornflowerblue",
    #'InN-Adarb2' : "lightsteelblue",
    'InN-Chat' : "lavender",
    #'InN-Egfr' : "turquoise",
    #'InN-Calb' : "teal",
     'InN-Lhx6':'lightsteelblue',

    'InN-Calb2' : "navy",
    'InN-Lamp5' : "royalblue",
    'InN-Pvalb' : "steelblue",
    'InN-Sst' : "dodgerblue",
    'InN-Vip' : "deepskyblue",
    "MSN-D1" : "mediumslateblue",
    "MSN-D2" : "rebeccapurple",
    # immune cells + microglia -- pink
    "Micro" : "deeppink",
    "T cell" : "crimson",
    "Macro" : "hotpink",
    
    # Endothelial/vasculure -- gold/tan
    "Vlmc" : "olive",
    "Endo" : "khaki",
    "Peri" : "goldenrod",
    "Epen" : "burlywood",
    
    # Oligodendrocytes
    "Olig" : "slategrey",
    "OPC" : "black"
}



def generate_palettes(A,clust_key="clust_annot", cell_type_key="cell_type"):
    print("Updated")
    uniq_celltypes = np.sort(np.unique(A.obs[cell_type_key]))
    uniq_clusts = np.sort(A.obs[clust_key].unique())

    celltype_pals = []
    for i in uniq_celltypes:
        pal = gen_dark_palette(i, major_cell_types[i], uniq_celltypes)
        celltype_pals.append(pal)
    celltype_pals = cycler(color=np.vstack(celltype_pals))

    celltype_colors = {}
    for i,c in enumerate(iter(celltype_pals)):
        celltype_colors[uniq_celltypes[i]] = c['color']

    clust_pals = []
    label_colors = {}
    for i in sorted(clust_cell_types.keys()):
        n = np.sum([1 if i in j else 0 for j in uniq_clusts])
        if n > 0:
            pal = gen_dark_palette(i, clust_cell_types[i], uniq_clusts)
            print(i,pal)
            clust_pals.append(pal)
            # find palettes for cell types
            curr_clusts = sorted([k for k in uniq_clusts if i in k])
            for n,p in enumerate(pal):
                label_colors[curr_clusts[n]] = p
        else:
            print("Couldn't find clust", i)
    clust_pals = cycler(color=np.vstack(clust_pals))
    #label_colors = {}
    #for i, c in enumerate(iter(clust_pals)):
    #    label_colors[valid_clusts[i]] = c['color']

    return celltype_colors, celltype_pals, label_colors, clust_pals

def calculate_aspect_ratio(A, rot=0,fov_size=221):
    all_pts = A.obsm['spatial']
    if rot>0:
        rotate(all_pts, degrees=rot)
    max_x = all_pts[:,0].max()
    min_x = all_pts[:,0].min()
    max_y = all_pts[:,1].max()
    min_y = all_pts[:,1].min()
    n_tiles_x = np.round((max_x-min_x)/fov_size)
    n_tiles_y = np.round((max_y-min_y)/fov_size)
    aspect_ratio = n_tiles_x/n_tiles_y
    return aspect_ratio, n_tiles_x, n_tiles_y

def plot_clust_subset(A, cell_types, cmap, ax=None, rot=0, s=0.1, xlim=None, ylim=None,alpha=0.1,clust_key="clust_annot"):
    if ax is None:
        f,ax = plt.subplots()
    all_pts = A.obsm['spatial'].copy()#np.array([A.obs.center_x, A.obs.center_y]).T
    # zero center all_pts
    all_pts = rotate(all_pts, degrees=rot)
    all_pts[:,0] -= all_pts[:,0].min()
    all_pts[:,1] -= all_pts[:,1].min()

    curr_idx = np.argwhere([i in cell_types for i in A.obs[clust_key]]).flatten()
    curr_pts = all_pts[curr_idx,:]
    other_idx = np.array([i for i in np.arange(all_pts.shape[0]) if i not in curr_idx])
    if len(other_idx) > 0:
        ax.scatter(all_pts[other_idx,:][:,0],all_pts[other_idx,:][:,1],s=s,vmin=0,vmax=1, c='lightgray', alpha=alpha,rasterized=True)
    print(all_pts[:,0].min(), all_pts[:,0].max(),all_pts[:,1].min(), all_pts[:,1].max())
    ax.scatter(curr_pts[:,0],curr_pts[:,1],s=s,vmin=0,vmax=A.obs.clust_encoding.max(),c=A.obs.clust_encoding[curr_idx],rasterized=True,cmap=cmap)
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_seg(A, cmap, ax=None, rot=0, s=0.1, xlim=None, ylim=None,key='spatial_clust_annots_value',vmax=7):
    if ax is None:
        f,ax = plt.subplots()
    all_pts = A.obsm['spatial'].copy()#np.array([A.obs.center_x, A.obs.center_y]).T
    # zero center all_pts
    all_pts = rotate(all_pts, degrees=rot)
    all_pts[:,0] -= all_pts[:,0].min()
    all_pts[:,1] -= all_pts[:,1].min()
    ax.scatter(all_pts[:,0], all_pts[:,1],s=s, c=A.obs[key],cmap=cmap,vmin=0,vmax=vmax)
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
from scipy.stats import ttest_ind, ranksums
def calc_pvals_for_grouping(x,y,data,hue,order=None):
    if order is None:
        order = sorted(list(data[x].unique()))
    hue_conds = list(data[hue].unique()) # assumes there are only two for this
    pvals = []
    for i in order:
        A = data[np.logical_and(data[x]==i, data[hue]==hue_conds[0])][y]
        B = data[np.logical_and(data[x]==i, data[hue]==hue_conds[1])][y]
        pval = ranksums(A,B)
        pvals.append(pval[1])
    return pvals

def plot_pvals(ax, pvals):
    ymin, ymax = ax.get_ylim()
    xticks = ax.get_xticks()
    for i,p in enumerate(pvals):
        if p < 0.01:
            ax.text(xticks[i], ymax, '*')
 
def plot_cond_obs_comparison(data, x, y, cell_type, figsize=(5,3), order=None, clust_key='cell_type', cond_pal=sns.color_palette(['g','m']), ylim=None):
    f, ax = plt.subplots(figsize=figsize)
    curr_df = data[data.obs[clust_key]==cell_type].obs
    if order is None:
        order = sorted(curr_df[x].unique())
    #sns.violinplot(x=x, y=y, data=curr_df,hue='age',fliersize=1,linewidth=1,palette=age_pal, ax=ax,inner=None,order=order,rasterized=True)
    sns.boxplot(x=x, y=y, data=curr_df,hue='cond',fliersize=0,linewidth=1,ax=ax, palette=cond_pal,order=order)
    sns.stripplot(data=curr_df, x=x, y=y, hue="cond",jitter=0.15,size=0.5,dodge=True,color='k', rasterized=True,ax=ax, order=order)
    sns.despine()
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.legend([],[], frameon=False)

#    sns.despine()
#    plt.legend([],[], frameon=False)
    #if show_pvals:
    #    pvals = calc_pvals_for_grouping(x,y,curr_df, "cond",order=order)
    #    plot_pvals(ax, pvals)
    return f
           
def plot_age_obs_comparison(data, x, y, cell_type, figsize=(5,3), show_pvals=False, order=None, clust_key='cell_type', age_pal=sns.color_palette(['cornflowerblue','thistle','lightcoral'])):
    f, ax = plt.subplots(figsize=(5,3))
    curr_df = data[data.obs[clust_key]==cell_type].obs
    if order is None:
        order = sorted(curr_df[x].unique())
    #sns.violinplot(x=x, y=y, data=curr_df,hue='age',fliersize=1,linewidth=1,palette=age_pal, ax=ax,inner=None,order=order,rasterized=True)
    sns.boxplot(x=x, y=y, data=curr_df,hue='age',fliersize=0,linewidth=1,palette=age_pal, ax=ax,order=order)

    sns.stripplot(data=curr_df, x=x, y=y, hue="age", ax=ax,jitter=0.15,size=0.5,dodge=True,color='k',order=order, rasterized=True)

    sns.despine()
    plt.legend([],[], frameon=False)
    if show_pvals:
        pvals = calc_pvals_for_grouping(x,y,curr_df, "age",order=order)
        plot_pvals(ax, pvals)
    return f

def plot_obs(A, cell_types, obs_name, cmap, ax=None, rot=0, s=0.1, xlim=None, ylim=None,vmin=0,vmax=10,alpha=0.1,key="clust_annot"):
    if ax is None:
        f,ax = plt.subplots()
    all_pts = A.obsm['spatial'].copy()#np.array([A.obs.center_x, A.obs.center_y]).T
    print("Shape", all_pts.shape)
    # zero center all_pts
    all_pts = rotate(all_pts, degrees=rot)
    all_pts[:,0] -= all_pts[:,0].min()
    all_pts[:,1] -= all_pts[:,1].min()

    curr_idx = np.argwhere([i in cell_types for i in A.obs[key]]).flatten()
    curr_pts = all_pts[curr_idx,:]
    other_idx = np.array([i for i in np.arange(all_pts.shape[0]) if i not in curr_idx])
    if len(other_idx) > 0:
        ax.scatter(all_pts[other_idx,:][:,0],all_pts[other_idx,:][:,1],s=s,vmin=0,vmax=1, c='lightgray', alpha=alpha,rasterized=True, edgecolors='face')
    #print(all_pts[:,0].min(), all_pts[:,0].max(),all_pts[:,1].min(), all_pts[:,1].max())
    ax.scatter(curr_pts[:,0],curr_pts[:,1],s=s,vmin=vmin,vmax=vmax,c=np.array(A[curr_idx,:].obs[obs_name]),rasterized=True,cmap=cmap, edgecolors='face')
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_gene_expr(A, cell_types, gene_name, cmap, ax=None, rot=0, s=0.1, xlim=None, ylim=None,vmin=0,vmax=10,use_raw=True,key='clust_annot',alpha=1):
    if ax is None:
        f,ax = plt.subplots()
    curr_idx = np.argwhere([i in cell_types for i in A.obs[key]]).flatten()
    other_idx = np.array([i for i in np.arange(A.shape[0]) if i not in curr_idx]).astype(np.int)
    curr_adata = A[curr_idx, :]
    other_adata = A[other_idx, :]
    if use_raw:
        curr_adata = curr_adata.raw.to_adata()
    curr_pts = curr_adata.obsm['spatial']#[curr_idx]
    other_pts = other_adata.obsm['spatial']#[other_idx]
    # zero center all_pts
    curr_pts = rotate(curr_pts, degrees=rot)
    curr_pts[:,0] -= curr_pts[:,0].min()
    curr_pts[:,1] -= curr_pts[:,1].min()

    gene_idx = np.argwhere([i==gene_name for i in A.var_names]).flatten()[0]
    if len(other_idx) > 0 and other_pts.shape[0] != curr_pts.shape[0]:
        print("plotting background")
        other_pts = rotate(other_pts, degrees=rot)
        other_pts[:,0] -= other_pts[:,0].min()
        other_pts[:,1] -= other_pts[:,1].min()

        ax.scatter(other_pts[:,0],other_pts[:,1],s=s,vmin=0,vmax=1, c='lightgray', rasterized=True, zorder=0,alpha=alpha)
    expr = np.array(curr_adata[:,gene_name].X.toarray()).flatten()
    ax.scatter(curr_pts[:,0],curr_pts[:,1],s=s,vmin=vmin,vmax=vmax,c=expr,rasterized=True,cmap=cmap, zorder=1,alpha=alpha)
    #print(curr_pts.shape, len(np.array(curr_adata[:,gene_name].X.flatten())))
    #ax.scatter(curr_pts[:,0],curr_pts[:,1],s=s,c=np.array(curr_adata[:,gene_name].X.toarray()))
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def plot_obs_by_cells(A, obs_name, s=0.1, cmap=plt.cm.gist_rainbow, show_legend=False, vmax=None, rot=0):
    pts = A.obsm['spatial']#np.array([A.obs.center_x, A.obs.center_y]).T
    if rot != 0:
        pts = rotate(pts, degrees=rot)
    pts = pd.DataFrame({'x': pts[:,0], 'y': pts[:,1], 'obs':A.obs[obs_name]})
    if vmax is None:
        vmax = len(pts.obs.unique())
    cols = cmap(np.linspace(0,1,vmax+1))
    #for n,i in enumerate(pts.obs.unique()):
        #curr_pts = pts[pts.obs==i]
    plt.scatter(pts.x,pts.y,s=s,vmin=0,vmax=vmax,c=pts.obs,cmap=cmap)
    if show_legend:
        plt.legend(pts.obs.unique())
        
#def plot_gene_by_cells(A, gene_name, s=0.1, cmap=plt.cm.Reds):
#    gene_idx = np.argwhere(A.var_names==gene_name)[0][0]
#    pts = pd.DataFrame({'x': A.obs.center_x, 'y':  A.obs.center_y, 'obs':A.X[:,gene_idx]})
#    plt.scatter(pts.x,pts.y,c=pts.obs, cmap=cmap, s=s)


def plot_expr_matrix_single(tstats, pvals, celltypes, vmin=-25, vmax=25,cmap=plt.cm.seismic, ax=None):
    pvals[np.isnan(pvals)] = 1
    pvals_correct = multipletests(pvals.flatten(), method='fdr_bh')[1]
    pvals_correct = pvals_correct.reshape(tstats.shape)
    pvals_correct[pvals_correct<1e-10] = 1e-10
    #for idx in range(200):
    if ax is None:
        f, ax = plt.subplots(figsize=(5,1))
    for i in range(tstats.shape[0]):
        if pvals_correct[i] < 0.05:
            ax.scatter(i, 1, s=-np.log10(pvals_correct[i])*10, c=tstats[i],vmin=vmin,vmax=vmax,cmap=cmap, lw=1, edgecolor='k')
        else:
            ax.scatter(i, 1, s=-np.log10(pvals_correct[i])*10, c=tstats[i],vmin=vmin,vmax=vmax,cmap=cmap, lw=1, edgecolor='w')
    ax.set_xticks(np.arange(len(celltypes)));
    ax.set_yticks([])
    ax.set_xticklabels(celltypes,rotation=90);

def plot_expr_matrix_by_name(tstats, pvals, gene_name, var_names,celltypes, vmin=-25, vmax=25,cmap=plt.cm.seismic):
    idx = np.argwhere(var_names==gene_name)[0]
    pvals[np.isnan(pvals)] = 1
    pvals_correct = multipletests(pvals.flatten(), method='fdr_bh')[1]
    pvals_correct = pvals_correct.reshape(tstats.shape)
    pvals_correct[pvals_correct<1e-10] = 1e-10
    #for idx in range(200):
    f, ax = plt.subplots(figsize=(5,5))
    ax.set_title(var_names[idx])
    for i in range(tstats.shape[0]):
        for j in range(tstats.shape[1]):
            if pvals_correct[i,j,idx] < 0.05:
                ax.scatter(i, j, s=-np.log10(pvals_correct[i,j,idx])*10, c=tstats[i,j,idx],vmin=vmin,vmax=vmax,cmap=plt.cm.bwr, lw=1, edgecolor='k')
            else:
                pass
#                ax.scatter(i, j, s=-np.log10(pvals_correct[i,j,idx])*10, c=tstats[i,j,idx],vmin=vmin,vmax=vmax,cmap=plt.cm.bwr, lw=1, edgecolor='w')

        ax.set_xticks(np.arange(len(celltypes)));
        ax.set_yticks(np.arange(len(celltypes)));
        ax.set_xticklabels(celltypes,rotation=90);
        ax.set_yticklabels(celltypes);
        ax.set_xlabel('Source')
        ax.set_ylabel('Neighbor')

def plot_interactions(pvals, interactions, celltypes, celltype_colors,figsize=(5,5),seg_points=None,vmin=0,vmax=5,cmap=plt.cm.Reds, qval_thresh=0.1):
    pvals[pvals<1e-10] = 1e-10
    f, ax = plt.subplots(figsize=figsize)
    gs = plt.GridSpec(nrows=2,ncols=2, width_ratios=[0.5,20], height_ratios=[20,0.5], wspace=0.1, hspace=0.1)
    ax = plt.subplot(gs[0,0])
    curr_cmap = mpl.colors.ListedColormap([celltype_colors[i] for i in celltypes])
    ax.imshow(np.expand_dims(np.arange(interactions.shape[0])[::-1],1),aspect='auto',interpolation='none',cmap=curr_cmap)
    sns.despine(ax=ax,bottom=True,left=True)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(celltypes)));
    ax.set_yticklabels(celltypes[::-1]);

    ax = plt.subplot(gs[0,1])
    ax.imshow(np.zeros_like(interactions), cmap=plt.cm.seismic, rasterized=True, aspect='auto',interpolation='none', vmin=-1,vmax=1)
    for i in range(interactions.shape[0]):
        for j in range(interactions.shape[0]):
            if pvals[i,j] < qval_thresh:
                #ax.scatter(i,j, s=-np.log10(pvals[i,j])*10, c=interactions[i,j],cmap=cmap,vmin=vmin,vmax=vmax, lw=1, edgecolor='k')
                ax.scatter(i, interactions.shape[0]-j-1, s=100, c=interactions[i,j],cmap=cmap,vmin=vmin,vmax=vmax, lw=1, edgecolor='k')

            else:
                pass
                #ax.scatter(i, interactions.shape[0]-j-1, s=100, c=interactions[i,j],cmap=cmap,vmin=vmin,vmax=vmax, lw=1, edgecolor='w')
    #ax.set_xlim([-1, 1+len(celltypes)])
    #ax.set_ylim([-1, 1+len(celltypes)])
    #ax.set_xticks(np.arange(len(celltypes)))
    #ax.set_yticks(np.arange(len(celltypes)))
    if seg_points is not None:
        for i in seg_points:
            ax.axvline(i-0.5,color='k',linestyle='--')
            ax.axhline(len(clust_annots)-i-0.5,color='k',linestyle='--')

    ax.axis('off')
    ax = plt.subplot(gs[1,1])
    curr_cmap = mpl.colors.ListedColormap([celltype_colors[i] for i in celltypes])
    ax.imshow(np.expand_dims(np.arange(interactions.shape[0]),1).T,aspect='auto',interpolation='none',cmap=curr_cmap)
    sns.despine(ax=ax,bottom=True,left=True)
    ax.set_xticks(np.arange(len(celltypes)));
    ax.set_xticklabels(celltypes,rotation=90);
    ax.set_yticks([])
    return f
def plot_clust_spatial_enrichment(A,vmin=0,vmax=1,uniq_clusts=None,clust_key='clust_annot',label_colors=None, spatial_domains=['Pia','L2/3', 'L5','L6', 'LatSept', 'CC', 'Striatum','Ventricle'],
    seg_cmap=plt.cm.viridis):
    if uniq_clusts is None:
        uniq_clusts = sorted(A.obs[clust_key].unique())
    n_spatial_domains = int(A.obs.spatial_clust_annots_value.max() + 1)
    clust_counts = np.zeros((n_spatial_domains, len(uniq_clusts)))
    print(clust_counts.shape)
    for i in range(n_spatial_domains):
        curr_clusts = A[A.obs.spatial_clust_annots_value==i,:].obs[clust_key]
        for j,c in enumerate(uniq_clusts):
            clust_counts[i,j] = np.sum(curr_clusts==c)
    clust_avgs = clust_counts.copy()
    for i in range(clust_avgs.shape[1]):
        clust_avgs[:,i] /= clust_avgs[:,i].sum()

    f, ax = plt.subplots(figsize=(5.5,1))
    gs = plt.GridSpec(nrows=2,ncols=2,width_ratios=[0.36, 20], height_ratios=[20,2], wspace=0.01, hspace=0.05)

    ax = plt.subplot(gs[0,0])
    ax.imshow(np.expand_dims(np.arange(n_spatial_domains),1),aspect='auto',interpolation='none', cmap=seg_cmap,rasterized=True)
    sns.despine(ax=ax,bottom=True,left=True)
    ax.set_yticks(np.arange(clust_avgs.shape[0]));
    ax.set_yticklabels(spatial_domains,fontsize=6)
    ax.set_xticks([])
    ax = plt.subplot(gs[0,1])
    ax.imshow(clust_avgs,aspect='auto',vmin=vmin,vmax=vmax, cmap=plt.cm.viridis)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    #for i in range(clust_counts.shape[0]):
        #ax.scatter(np.arange(clust_counts.shape[1]), i*np.ones(clust_counts.shape[1]), s=0.005*clust_counts[i,:],c='k')
    ax = plt.subplot(gs[1,1])
    if label_colors is None:
        curr_cmap = plt.cm.viridis
    else:
        curr_cmap = mpl.colors.ListedColormap([label_colors[i] for i in uniq_clusts])
    ax.imshow(np.expand_dims(np.arange(len(uniq_clusts)),1).T,aspect='auto',interpolation='none', cmap=curr_cmap,rasterized=True)

    ax.set_xticks(np.arange(clust_avgs.shape[1]));
    ax.set_yticks([])
    ax.set_xticklabels(uniq_clusts,rotation=90,fontsize=6);
    sns.despine(ax=ax, left=True, bottom=True)
    return clust_avgs, clust_counts
