from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#from xgboost import XGBClassifier
def train_umap_classifier(adata, label_column, umap_key='X_umap', n_dims=30, method='knn'):
    '''Train a KNN classifier using the UMAP coordinates.'''
    if method == 'knn':
        knnc = KNeighborsClassifier(n_jobs=-1, n_neighbors=25) ##KNeighborsClassifier(n_jobs=-1)#
    elif method == "mlp":
        knnc = MLPClassifier()
    knnc.fit(adata.obsm[umap_key][:, :n_dims], np.array(adata.obs[label_column]))
    return knnc


def impute_classification(adata, classifier, prediction_column, probability_column, umap_key='X_umap', n_dims=30):
    '''Impute using a trained classifier.'''
    classes = classifier.classes_
    probas = classifier.predict_proba(adata.obsm[umap_key][:, :n_dims])
    max_ids = np.argmax(probas, axis=1)
    max_probas = np.max(probas, axis=1)
    predicted_classes = [classes[i] for i in max_ids]
    
    adata.obs[prediction_column] = predicted_classes
    adata.obs[probability_column] = max_probas

    
def impute_gene_expression():
    pass