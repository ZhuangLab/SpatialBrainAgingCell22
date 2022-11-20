import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from tqdm import tqdm
import diffxpy as de

def lrtest(llmin,llmax):
    lr = likelihood_ratio(llmin, llmax)
    p = chi2.sf(lr,1)
    return p
from scipy.stats.distributions import chi2
def likelihood_ratio(llmin, llmax):
    llmin = -llmin
    llmax = -llmax
    return(2*(llmax-llmin))

def run_glm_de_age_lps_merfish(adata, family='poisson', grouping='cell_type_annot', logfc_thresh=np.log(1)):
    # do LR test
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PrecisionWarning, IterationLimitWarning, EstimationWarning, SingularMatrixWarning
    #from statsmodels.regression.linear_model.OLSResults import compare_lr_test
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', PrecisionWarning)
    warnings.simplefilter('ignore', IterationLimitWarning)
    warnings.simplefilter('ignore', EstimationWarning)
    warnings.simplefilter('ignore', SingularMatrixWarning)
    warnings.simplefilter('ignore', FutureWarning)

    if family == 'nb':
        family = sm.families.NegativeBinomial()
    elif family == 'poisson':
        family = sm.families.Poisson()
    
    all_model_fits = {}
    all_results = {}
    
    for clust in adata.obs[grouping].unique()[::-1]:
        print(clust)
        curr_adata = adata[adata.obs[grouping]==clust].copy()
        print(curr_adata.shape)
        curr_coefs_age = []
        curr_coefs_lps = []
        curr_pvals = []
        curr_genes = list(curr_adata.var_names)
        for i in tqdm(range(len(curr_genes))):
            try:
                
                curr_adata.obs["Y"] = curr_adata[:,curr_genes[i]].X.toarray()
                formula = "Y ~ C(age) + C(cond) + log_umi" 
                #mdf = smf.glm(formula, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
                mdf = smf.ols(formula, data=curr_adata.obs).fit(maxiter=50,disp=0)
                #mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=50,disp=0)
                formula_reduced = "Y ~ log_umi"
                #mdf_reduced = smf.glm(formula_reduced, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
                mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=50,disp=0)
                curr_coefs_age.append(mdf.params['C(age)[T.90wk]'])
                curr_coefs_lps.append(mdf.params["C(cond)[T.lps]"])
                curr_pvals.append(lrtest(mdf.llf, mdf_reduced.llf))
                #curr_pvals.append(mdf.compare_lr_test(mdf_reduced)[])
            except Exception as e:
                print(e)
                curr_coefs_age.append(None)
                curr_coefs_lps.append(None)
                curr_pvals.append(None)
        #curr_genes = [curr_genes[i] for i in range(len(curr_genes)) if curr_coefs_age[i] is not None or curr_coefs_lps is not None]
        #coef_age = [c for c in curr_coefs_age if c is not None]
        #coef_lps = [c for c in curr_coefs_lps if c is not None]
        pvals = [p for p in curr_pvals if p is not None]
        results = pd.DataFrame({'cell_type':clust, 'coef_age':curr_coefs_age, 'coef_lps':curr_coefs_lps, 'pval':pvals, 'gene':curr_genes})
        results['qval'] = multipletests(results.pval, method='fdr_bh')[1]
        all_results[clust] = results
    return all_results

def run_glm_de_age_merfish(adata, family='poisson', grouping='cell_type_annot', obs_name="age", comp_name="T.90wk", logfc_thresh=np.log(1)):
    # do LR test
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PrecisionWarning, IterationLimitWarning, EstimationWarning, SingularMatrixWarning 
    #from statsmodels.regression.linear_model.OLSResults import compare_lr_test
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', PrecisionWarning)
    warnings.simplefilter('ignore', IterationLimitWarning)
    warnings.simplefilter('ignore', EstimationWarning)
    warnings.simplefilter('ignore', SingularMatrixWarning)
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('ignore',RuntimeWarning)
    if family == 'nb':
        family = sm.families.NegativeBinomial()
    elif family == 'poisson':
        family = sm.families.Poisson()
    
    all_model_fits = {}
    all_results = {}
    
    for clust in adata.obs[grouping].unique()[::-1]:
        print(clust)
        curr_adata = adata[adata.obs[grouping]==clust].copy()
        print(curr_adata.shape)
        curr_coefs = []
        curr_pvals = []
        curr_stderr = []
        curr_genes = list(curr_adata.var_names)
        for i in tqdm(range(len(curr_genes))):
            try:
                
                curr_adata.obs["Y"] = curr_adata[:,curr_genes[i]].X.toarray()
                formula = f"Y ~ C({obs_name}) + 1"# + log_umi"
                if family != "ols":
                    mdf = smf.glm(formula, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
                else:
                    mdf = smf.ols(formula, data=curr_adata.obs).fit(maxiter=50,disp=0)
                #mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=50,disp=0)
                formula_reduced = "Y ~ 1" #log_umi"
                if family != "ols":
                    mdf_reduced = smf.glm(formula_reduced, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
                else:
                    mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=50,disp=0)
                curr_coefs.append(mdf.params[f'C({obs_name})[{comp_name}]'])
                curr_pvals.append(lrtest(mdf.llf, mdf_reduced.llf))
                curr_stderr.append(mdf.bse[f'C({obs_name})[{comp_name}]'])
                #curr_pvals.append(mdf.compare_lr_test(mdf_reduced)[])
            except Exception as e:
                print(e)
                curr_coefs.append(None)
                curr_pvals.append(None)
                curr_stderr.append(None)
        #curr_genes = [curr_genes[i] for i in range(len(curr_genes)) if curr_coefs[i] is not None]
        #coef = [c for c in curr_coefs if c is not None]
        #pvals = [p for p in curr_pvals if p is not None]
        stderrs = [s for s in curr_stderr if s is not None]
        results = pd.DataFrame({'cell_type':clust, 'coef':curr_coefs, 'pval':curr_pvals, 'gene':curr_genes, 'stderr': curr_stderr})
        results['qval'] = multipletests(results.pval, method='fdr_bh')[1]
        all_results[clust] = results
    return all_results


def geomean(X,axis=1,epsilon=1):
    return np.exp(np.mean(np.log(X+epsilon), axis))-epsilon

def avg_umi_per_gene(X):
    return np.sum(X,1)

def compute_frac_expressed(A):
    return np.array((A.X>0).sum(0)/A.shape[0]).flatten()

def compute_mean_expression(A):
    return np.array(A.X.mean(0)).flatten()

def filter_2group_1way(A, obs_name, ident, min_pct=None, logfc_thresh=None, min_diff_pct=None, max_cells_per_ident=None, log=True):
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
        idx_X = np.nonzero((adata.obs[obs_name]==ident).values)[0]
        idx_Y = np.nonzero((adata.obs[obs_name]!=ident).values)[0]
        ids_X = idx_A[np.random.permutation(len(idx_X))[:max_cells_per_ident]]
        ids_Y = idx_B[np.random.permutation(len(idx_Y))[:max_cells_per_ident]]
        combined_ids = np.hstack((ids_X, ids_Y)).flatten()
        return A[combined_ids,:], logfc_XY[np.array(final_mask).flatten()]
    else:
        
        return A, logfc_XY[np.array(final_mask).flatten()]

def filter_2group(A, obs_name, ident, min_pct=None, logfc_thresh=None, min_diff_pct=None, max_cells_per_ident=None, log=True):
    """
    Filter genes before differential expression testing. NOTE this is bidirectional
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
        min_pct_mask = np.logical_or(pct_X>min_pct, pct_Y>min_pct).flatten()
        
    mean_X = compute_mean_expression(X)
    mean_Y = compute_mean_expression(Y)
    if log:
        logfc_XY = np.log(np.exp(mean_X)/np.exp(mean_Y))

        logfc_YX = np.log(np.exp(mean_Y)/np.exp(mean_X))
    else:
        logfc_XY = np.log(mean_X/mean_Y)
        logfc_YX = np.log(mean_Y/mean_X)
      
    if logfc_thresh:
        log_fc_mask = np.logical_or(logfc_XY > logfc_thresh, logfc_YX > logfc_thresh).flatten()
    
    if min_diff_pct:
        diff_pct_XY = pct_X-pct_Y
        diff_pct_YX = pct_Y-pct_X
        min_diff_pct_mask = np.logical_or(diff_pct_XY > min_diff_pct, diff_pct_YX > min_diff_pct).flatten()
    final_mask = np.logical_and(np.logical_and(min_pct_mask, log_fc_mask), min_diff_pct_mask).flatten()
    A = A[:, final_mask]
    
    if max_cells_per_ident:
        idx_X = np.nonzero((adata.obs[obs_name]==ident).values)[0]
        idx_Y = np.nonzero((adata.obs[obs_name]!=ident).values)[0]
        ids_X = idx_A[np.random.permutation(len(idx_X))[:max_cells_per_ident]]
        ids_Y = idx_B[np.random.permutation(len(idx_Y))[:max_cells_per_ident]]
        combined_ids = np.hstack((ids_X, ids_Y)).flatten()
        return A[combined_ids,:], logfc_XY[np.array(final_mask).flatten()]
    else:
        
        return A, logfc_XY[np.array(final_mask).flatten()]
    
from scipy.stats.distributions import chi2
def likelihood_ratio(llmin, llmax):
    llmin = -llmin
    llmax = -llmax
    return(2*(llmax-llmin))

def lrtest(llmin,llmax):
    lr = likelihood_ratio(llmin, llmax)
    p = chi2.sf(lr,1)
    return p

def run_glm_de_pairwise(curr_adata, contrast, lognorm=False):
    # run glm on pair of clusters, using contrast as True/False
    curr_coefs = []
    curr_pvals = []
    curr_stderr = []
    curr_genes = list(curr_adata.var_names)
    family = sm.families.NegativeBinomial()
    for i in range(len(curr_genes)):
        try: 
            if lognorm:
                curr_adata.obs["Y"] = np.log1p(curr_adata[:,curr_genes[i]].layers['counts'].toarray())
            else:
                curr_adata.obs["Y"] = curr_adata[:,curr_genes[i]].layers['counts'].toarray()
            formula = f"Y ~  C({contrast}) + log_umi + avg_UMI"
            mdf = smf.glm(formula, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)

            #mdf = smf.ols(formula, data=curr_adata.obs).fit(maxiter=30,disp=0)
            formula_reduced = "Y ~ log_umi + avg_UMI"
            #mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=30,disp=0)

            mdf_reduced = smf.glm(formula_reduced, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
            curr_coefs.append(mdf.params[f'C({contrast})[T.True]'])
            curr_pvals.append(lrtest(mdf.llf, mdf_reduced.llf))
            curr_stderr.append(mdf.bse[f'C({contrast})[T.True]'])
        except Exception as e:
            print(e)
            curr_coefs.append(None)
            curr_pvals.append(None)
            curr_stderr.append(None)
    curr_genes = [curr_genes[i] for i in range(len(curr_genes)) if curr_coefs[i] is not None]
    coef = [c for c in curr_coefs if c is not None]
    pvals = [p for p in curr_pvals if p is not None]
    stderrs = [s for s in curr_stderr if s is not None]
    results = pd.DataFrame({'coef':coef, 'pval':pvals, 'gene':curr_genes, 'stderr': curr_stderr})
    results['qval'] = multipletests(results.pval, method='fdr_bh')[1]
    return results


def run_ttest_de_age(adata, family='nb', grouping='cell_type',  lognorm=False, min_pct=0.1, logfc_thresh=np.log(1)):
    # do LR test
    import warnings
    from scipy.stats import mannwhitneyu
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PrecisionWarning, IterationLimitWarning, EstimationWarning, SingularMatrixWarning
    #from statsmodels.regression.linear_model.OLSResults import compare_lr_test
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', PrecisionWarning)
    warnings.simplefilter('ignore', IterationLimitWarning)
    warnings.simplefilter('ignore', EstimationWarning)
    warnings.simplefilter('ignore', SingularMatrixWarning)
    warnings.simplefilter('ignore', FutureWarning)
    from scipy.stats import ttest_ind 
    all_model_fits = {}
    all_results = {}
    
    for clust in adata.obs[grouping].unique()[::-1]:
        print(clust)
        curr_adata = adata[adata.obs[grouping]==clust].copy()
        curr_adata, _ = filter_2group(curr_adata, "age", "4wk", min_pct=min_pct, logfc_thresh=logfc_thresh)
        print(curr_adata.shape)
        curr_coefs = []
        curr_pvals = []
        curr_stderr = []
        curr_genes = list(curr_adata.var_names)
        for i in tqdm(range(len(curr_genes))):
            try:
                #if lognorm:
                #    curr_adata.obs["Y"] = np.log1p(curr_adata[:,curr_genes[i]].X.toarray())
                #else:
                X = curr_adata[:,curr_genes[i]].X.toarray()
                young_X = X[curr_adata.obs['age'] == '4wk']
                old_X = X[curr_adata.obs['age'] == '90wk']
                curr_coefs.append(np.log(old_X.mean()/young_X.mean()))
                curr_pvals.append(ttest_ind(old_X, young_X)[1])
                #curr_pvals.append(mannwhitneyu(old_X, young_X)[1])
                #curr_pvals.append(mdf.compare_lr_test(mdf_reduced)[])
            except Exception as e:
                #print(e)
                curr_coefs.append(None)
                curr_pvals.append(None)
        curr_genes = [curr_genes[i] for i in range(len(curr_genes)) if curr_coefs[i] is not None]
        coef = [c for c in curr_coefs if c is not None]
        pvals = [p for p in curr_pvals if p is not None]
        results = pd.DataFrame({'coef':coef, 'pval':pvals, 'gene':curr_genes})
        results['qval'] = multipletests(results.pval, method='fdr_bh')[1]
        all_results[clust] = results
    return all_results

def run_glm_de_age(adata, family='nb', grouping='cell_type',  lognorm=False, min_pct=0.1, logfc_thresh=np.log(1)):
    # do LR test
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PrecisionWarning, IterationLimitWarning, EstimationWarning, SingularMatrixWarning
    #from statsmodels.regression.linear_model.OLSResults import compare_lr_test
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', PrecisionWarning)
    warnings.simplefilter('ignore', IterationLimitWarning)
    warnings.simplefilter('ignore', EstimationWarning)
    warnings.simplefilter('ignore', SingularMatrixWarning)
    warnings.simplefilter('ignore', FutureWarning)
    if family == 'nb':
        family = sm.families.NegativeBinomial()
    elif family == 'poisson':
        family = sm.families.Poisson()
    
    all_model_fits = {}
    all_results = {}
    
    for clust in adata.obs[grouping].unique()[::-1]:
        print(clust)
        curr_adata = adata[adata.obs[grouping]==clust].copy()
        print(clust, curr_adata.shape)
        curr_adata, _ = filter_2group(curr_adata, "age", "4wk", min_pct=min_pct, logfc_thresh=logfc_thresh)
        print(curr_adata.shape)
        curr_coefs = []
        curr_pvals = []
        curr_stderr = []
        curr_genes = list(curr_adata.var_names)
        print("Using new formula")
        umi_coef = []
        for i in tqdm(range(len(curr_genes))):
            try:
                #if lognorm:
                #    curr_adata.obs["Y"] = np.log1p(curr_adata[:,curr_genes[i]].X.toarray())
                #else:
                curr_adata.obs["Y"] = curr_adata[:,curr_genes[i]].X.toarray()
                formula = "Y ~  C(age) + + log_umi"

                if family == "ols":
                    mdf = smf.ols(formula, data=curr_adata.obs).fit(maxiter=50,disp=0)
                else:
                    mdf = smf.glm(formula, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)

                formula_reduced = "Y ~ log_umi" # low_umi
                if family == "ols":
                    mdf_reduced = smf.ols(formula_reduced, data=curr_adata.obs).fit(maxiter=50,disp=0)
                else:
                    mdf_reduced = smf.glm(formula_reduced, data=curr_adata.obs, family=family).fit(maxiter=50,disp=0)
                umi_coef.append(mdf.params['log_umi'])
                curr_coefs.append(mdf.params['C(age)[T.90wk]'])
                curr_pvals.append(lrtest(mdf.llf, mdf_reduced.llf))
                curr_stderr.append(mdf.bse['C(age)[T.90wk]'])
                #curr_pvals.append(mdf.compare_lr_test(mdf_reduced)[])
            except Exception as e:
                #print(e)
                curr_coefs.append(None)
                curr_pvals.append(None)
                curr_stderr.append(None)
        print('Mean UMI coef', np.mean(umi_coef))
        curr_genes = [curr_genes[i] for i in range(len(curr_genes)) if curr_coefs[i] is not None]
        coef = [c for c in curr_coefs if c is not None]
        pvals = [p for p in curr_pvals if p is not None]
        stderrs = [s for s in curr_stderr if s is not None]
        results = pd.DataFrame({'coef':coef, 'pval':pvals, 'gene':curr_genes, 'stderr': curr_stderr})
        results['qval'] = multipletests(results.pval, method='fdr_bh')[1]
        all_results[clust] = results
    return all_results


def save_de_results(df_map, out_fname):
    """
    df_map is map of cell_type -> dataframe of differential gene expression results
    """
    de = []
    for k,v in df_map.items():
        v['cell_type'] = k
        #v['qval'] = multipletests(np.array(v['pval']), method='fdr_bh')[1]
        de.append(v)
    out_df = pd.concat(de)
    out_df.to_csv(out_fname)
    return out_df