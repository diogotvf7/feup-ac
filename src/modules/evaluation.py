from sklearn.feature_selection import RFE

CORRECT_TEAMS = ['IND', 'ATL', 'DET', 'SAS', 'PHO', 'SEA', 'LAS', 'WAS']

def calculate_precision(predicted_teams):
    correct_predictions = len(set(predicted_teams) & set(CORRECT_TEAMS))
    precision = correct_predictions / len(predicted_teams)
    return precision


def feature_selection(model, training_dataset, evaluate_dataset):
    # Feature Selection
    rfe = RFE(model, n_features_to_select=10)

    X_train_rfe = rfe.fit_transform(training_dataset, evaluate_dataset.columns)

    print("\033[31mSelected features:\033[41m", training_dataset.columns[rfe.get_support()])

def evaluate(model, training_dataset, evaluate_dataset):
    training_data_copy = training_dataset.copy()
    evaluate_data_copy = evaluate_dataset.copy()

    features_train = training_data_copy.drop(columns=['playoff'])
    target_train = training_data_copy['playoff']

    # Feature Selection
    # feature_selection(model, training_data_copy, evaluate_data_copy)

    model.fit(features_train, target_train)

    #TODO:Shouldn't need to drop columns here, besides tmID
    s10_team_features = evaluate_data_copy.drop(columns=['tmID'])
    playoff_probabilities = model.predict_proba(s10_team_features)[:, 1]
    
    evaluate_data_copy['playoff_probability'] = playoff_probabilities

    top_8_teams = evaluate_data_copy[['tmID', 'playoff_probability']].sort_values(by='playoff_probability', ascending=False).head(8)
    #print("Predicted top 8 teams:\n", top_8_teams)

    predicted_teams = top_8_teams['tmID'].tolist()

    precision = calculate_precision(predicted_teams)
    print(f"Prediction Precision: {precision:.2f}")

