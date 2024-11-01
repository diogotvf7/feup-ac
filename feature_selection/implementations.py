from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression  
from scipy.stats import spearmanr


def select_features(features, target, score_func, k=10):
    if(score_func == RFE):
        linear_regression = LinearRegression()
        selector = RFE(estimator=linear_regression, n_features_to_select=k)  
        _ = selector.fit(features, target) 
        selected_features = selector.get_support(indices=True)  
    else:
        selector = SelectKBest(score_func=score_func, k=k)
        _ = selector.fit_transform(features, target)
        selected_features = selector.get_support(indices=True)  

    return features.columns[selected_features] 


def select_spearman(features, target, threshold=0.2, k = 10):
    spearman_corr = []
    for feature in features.columns:
        corr, _ = spearmanr(features[feature], target)
        spearman_corr.append((feature, corr))
    filtered_features = [feature for feature, corr in spearman_corr if abs(corr) >= threshold]
    return filtered_features