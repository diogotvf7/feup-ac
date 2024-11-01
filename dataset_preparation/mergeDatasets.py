import pandas as pd

def drop_unnecessary_columns(teams_post):
    """Drop unnecessary columns from teams_post dataset."""
    return teams_post.drop(columns=['W', 'L'])

def merge_teams_with_playoff_status(teams, teams_post):
    """Merge teams with teams_post to determine playoff status."""
    teams_post = drop_unnecessary_columns(teams_post)
    teams_playoffs = pd.merge(teams, teams_post, how='left', left_on=['year', 'tmID'], right_on=['year', 'tmID'], indicator=True)
    teams_playoffs['playoff'] = (teams_playoffs['_merge'] == 'both').astype(int)
    teams_playoffs.drop(columns=['_merge'], inplace=True)
    return teams_playoffs

def merge_players_with_teams(players_teams, players):
    """Merge player and playersTeams datasets."""
    players_playersTeams = pd.merge(players_teams, players, how='left', right_on='bioID', left_on='playerID')
    return players_playersTeams.drop(columns=['bioID'])

def assign_award_points(awards_players):
    """Assign points to awards based on predefined scores."""
    award_scores = {
        "WNBA All-Decade Team": 20,
        "Coach of the Year": 10,
        "Most Valuable Player": 10,
        "WNBA All Decade Team Honorable Mention": 10,
        "WNBA Finals Most Valuable Player": 8,
        "Defensive Player of the Year": 7,
        "Rookie of the Year": 5,
        "Most Improved Player": 4,
        "Sixth Woman of the Year": 3,
        "Kim Perrot Sportsmanship": 1,
        "Kim Perrot Sportsmanship Award": 1,
        "All-Star Game Most Valuable Player": 1
    }
    awards_players['awardPointsPlayers'] = awards_players['award'].map(award_scores)
    return awards_players

def merge_awards_with_players(players_playersTeams, awards_players):
    """Merge awards with players and fill missing award points."""
    players_awards_merged = pd.merge(players_playersTeams, awards_players, how='left', left_on=['playerID', 'year'], right_on=['playerID', 'year'])
    players_awards_merged = players_awards_merged.drop(columns=['award'])
    players_awards_merged['awardPointsPlayers'] = players_awards_merged['awardPointsPlayers'].fillna(0)
    return players_awards_merged

def merge_awards_with_coaches(teams_playoffs, coaches, awards_players):
    """Merge awards with coaches and fill missing award points."""
    teams_playoffs_coaches_merged = pd.merge(teams_playoffs, coaches, how='left', on=['tmID', 'year'])
    teams_playoffs_coaches_merged = teams_playoffs_coaches_merged.drop(columns=['won_y', 'lost_y', 'post_wins', 'post_losses'])

    coach_of_year = awards_players[awards_players['award'] == "Coach of the Year"][['playerID', 'year']]
    coach_of_year['awardPointsCoaches'] = 10 

    teams_playoffs_coaches_merged = pd.merge(teams_playoffs_coaches_merged, coach_of_year, how='left', left_on=['year', 'coachID'], right_on=['year', 'playerID'])
    teams_playoffs_coaches_merged['awardPointsCoaches'] = teams_playoffs_coaches_merged['awardPointsCoaches'].fillna(0)
    teams_playoffs_coaches_merged.drop(columns=['playerID'], inplace=True)
    return teams_playoffs_coaches_merged

def create_final_dataset(players_awards_merged, teams_playoffs_coaches_merged):
    """Create the final dataset by merging players and coaches data."""
    final_dataset = pd.merge(players_awards_merged, teams_playoffs_coaches_merged, how='inner', on=['tmID', 'year'])
    return final_dataset

def drop_text_features(final_dataset):
    """Drop unnecessary text features from the final dataset."""
    return final_dataset.drop(columns=['playerID', 'tmID', 'pos', 'college', 'collegeOther', 'birthDate', 'deathDate', 'franchID', 'coachID'])

def process_data(datasets):
    """Main function to process the datasets."""
    teams_playoffs = merge_teams_with_playoff_status(datasets['teams'], datasets['teams_post'])
    players_playersTeams = merge_players_with_teams(datasets['players_teams'], datasets['players'])
    awards_players = assign_award_points(datasets['awards_players'])
    players_awards_merged = merge_awards_with_players(players_playersTeams, awards_players)
    teams_playoffs_coaches_merged = merge_awards_with_coaches(teams_playoffs, datasets['coaches'], awards_players)
    final_dataset = create_final_dataset(players_awards_merged, teams_playoffs_coaches_merged)
    final_dataset = drop_text_features(final_dataset)
    
    return final_dataset