from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2

CORRECT_TEAMS = ['IND', 'ATL', 'DET', 'SAS', 'PHO', 'SEA', 'LAS', 'WAS']

def calculate_precision(predicted_teams):
    correct_predictions = len(set(predicted_teams) & set(CORRECT_TEAMS))
    precision = correct_predictions / len(predicted_teams)
    return precision

def feature_selection(features, target, k=10):
    select_k_best = SelectKBest(score_func=chi2, k=k)
    _ = select_k_best.fit_transform(features, target)
    selected_features = features.columns[select_k_best.get_support()].tolist()
    return selected_features

def evaluate(model, training_dataset, evaluate_dataset, select_features=False):
    training_data_copy = training_dataset.copy()
    evaluate_data_copy = evaluate_dataset.copy()

    features_train = training_data_copy.drop(columns=['playoff'])
    target_train = training_data_copy['playoff']

    if select_features:
        selected_features = feature_selection(features_train, target_train, k=10)
        features_train = features_train[selected_features]
        evaluate_features = evaluate_data_copy[selected_features]
    else:
        evaluate_features = evaluate_data_copy.drop(columns=['tmID'])

    # Train the model
    model.fit(features_train, target_train)

    # Predict probabilities
    playoff_probabilities = model.predict_proba(evaluate_features)[:, 1]
    evaluate_data_copy['playoff_probability'] = playoff_probabilities
    
    top_8_teams = evaluate_data_copy[['tmID', 'playoff_probability']].sort_values(by='playoff_probability', ascending=False).head(8)
    #print("Predicted top 8 teams:\n", top_8_teams)

    predicted_teams = top_8_teams['tmID'].tolist()

    precision = calculate_precision(predicted_teams)
    # print(f"Prediction Precision: {precision:.2f}")

    return {
        'predicted_teams': predicted_teams, 
        'precision': precision
    }

def evaluate_model(model, training_dataset, evaluate_dataset):
    without_feature_selection = evaluate(model, training_dataset, evaluate_dataset, False)
    with_feature_selection = evaluate(model, training_dataset, evaluate_dataset, True)
    return {
        'without_feature_selection': without_feature_selection, 
        'with_feature_selection': with_feature_selection
    }