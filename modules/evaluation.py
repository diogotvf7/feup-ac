correct_teams = ['IND', 'ATL', 'DET', 'SAS', 'PHO', 'SEA', 'LAS', 'WAS']

def calculate_precision(predicted_teams):
    correct_predictions = len(set(predicted_teams) & set(correct_teams))
    precision = correct_predictions / len(predicted_teams)
    return precision


def evaluate(model, training_dataset, evaluate_dataset):

    training_data_copy = training_dataset.copy()
    evaluate_data_copy = evaluate_dataset.copy()

    features_train = training_data_copy.drop(columns=['playoff'])
    target_train = training_data_copy['playoff']

    model.fit(features_train, target_train)

    #TODO:Shouldn't need to drop columns here, besides tmID
    s10_team_features = evaluate_data_copy.drop(columns=['tmID', 'fgAttempted', 'ftAttempted', 'playoffs_percentage', 'points'])
    playoff_probabilities = model.predict_proba(s10_team_features)[:, 1]
    
    evaluate_data_copy['playoff_probability'] = playoff_probabilities

    top_8_teams = evaluate_data_copy[['tmID', 'playoff_probability']].sort_values(by='playoff_probability', ascending=False).head(8)
    #print("Predicted top 8 teams:\n", top_8_teams)

    predicted_teams = top_8_teams['tmID'].tolist()

    precision = calculate_precision(predicted_teams)
    print(f"Prediction Precision: {precision:.2f}")

