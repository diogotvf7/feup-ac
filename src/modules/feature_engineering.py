import pandas as pd
import numpy as np
from dataset_preparation.create_final_dataset import calculate_avg_stats_rookie

MIN_FG_ATTEMPTS = 5
POST_WIN_WEIGHT = 1.5

def safe_divide(a,b, threshold = 0, default = 0):
    return a / b if b > threshold else default

def player_feature_engineering(dataset):
    # Shooting Metrics
    dataset['fg_percentage'] = dataset.apply(lambda row: safe_divide(row['fgMade'], row['fgAttempted'], MIN_FG_ATTEMPTS), axis=1)
    dataset['ft_percentage'] = dataset.apply(lambda row: safe_divide(row['ftMade'], row['ftAttempted']), axis=1)
    dataset['three_percentage'] = dataset.apply(lambda row: safe_divide(row['threeMade'], row['threeAttempted']), axis=1)
    dataset['true_shooting_percentage'] = dataset.apply(lambda row: safe_divide(row['points'], 2 * (row['fgAttempted'] + 0.44 * row['ftAttempted']), MIN_FG_ATTEMPTS), axis=1)
    ## PerMinute Stats
    dataset['rebounds_per_minute'] = dataset.apply(lambda row: safe_divide(row['rebounds'], row['minutes']), axis=1)
    dataset['steals_per_minute'] = dataset.apply(lambda row: safe_divide(row['steals'], row['minutes']), axis=1)
    dataset['blocks_per_minute'] = dataset.apply(lambda row: safe_divide(row['blocks'], row['minutes']), axis=1)
    dataset['assists_per_minute'] = dataset.apply(lambda row: safe_divide(row['assists'], row['minutes']), axis=1)
    dataset['assist_turnover_ratio'] = dataset.apply(lambda row: safe_divide(row['assists'], row['turnovers'], default=row['assists']), axis=1)

    ## Dream Statistic -> PER 

    #https://www.teamrankings.com/nba/player/nikola-jokic
    dataset['effective_fg_percentage'] = dataset.apply(lambda row: safe_divide(row['fgMade'] + 0.5 * row['threeMade'], row['fgAttempted'], MIN_FG_ATTEMPTS), axis=1)
    #turnovers cant be done individually :     https://sportsjourneysinternational.com/sji-coaches-corner/turnover-percentage-the-second-most-important-factor-of-basketball-success/#:~:text=The%20easiest%20way%20to%20look%20at%20the%20individual,provides%20a%20good%20baseline%20for%20your%20individual%20statistics.
    
    columns_to_keep = [
        'playerID', 'year',  'tmID', 'GP', 'GS', 'minutes', 'points', 'fgAttempted', 
        'ftAttempted',
        'fg_percentage', 
        'ft_percentage', 'three_percentage', 'true_shooting_percentage',
        'rebounds_per_minute', 'steals_per_minute', 'blocks_per_minute', 'assists_per_minute',
        'assist_turnover_ratio', 'effective_fg_percentage'
    ]

    final_player = dataset[columns_to_keep]
    rookie_stats = calculate_avg_stats_rookie(final_player)

    rows_with_80_percent_zeros = (final_player == 0).sum(axis=1) >= 4
    for idx in final_player.index[rows_with_80_percent_zeros]:
        for column in rookie_stats:
            final_player.loc[idx, column] = rookie_stats[column]

    # print(f"Replacing {rows_with_80_percent_zeros.sum()} rows where zero values exceed {0.8*100}% of the columns.\n")
    
    return final_player


def prepTrainingDataset(teams, teams_post, coaches):    
    # Shooting Metrics
    teams['fg_percentage'] = teams.apply(lambda row: safe_divide(row['o_fgm'], row['o_fga']), axis=1)
    teams['ft_percentage'] = teams.apply(lambda row: safe_divide(row['o_ftm'], row['o_fta']), axis=1)
    teams['three_percentage'] = teams.apply(lambda row: safe_divide(row['o_3pm'], row['o_3pa']), axis=1)
    teams['true_shooting_percentage'] = teams.apply(lambda row: safe_divide(row['o_pts'], 2 * (row['o_fga'] + 0.44 * row['o_fta'])), axis=1)
    ## PerMinute Stats
    teams['rebounds_per_minute'] = teams.apply(lambda row: safe_divide(row['o_reb'], row['min']), axis=1)
    teams['steals_per_minute'] = teams.apply(lambda row: safe_divide(row['o_stl'], row['min']), axis=1)
    teams['blocks_per_minute'] = teams.apply(lambda row: safe_divide(row['o_blk'], row['min']), axis=1)
    teams['assists_per_minute'] = teams.apply(lambda row: safe_divide(row['o_asts'], row['min']), axis=1)
    teams['assist_turnover_ratio'] = teams.apply(lambda row: safe_divide(row['o_asts'], row['o_to'], default=row['o_asts']), axis=1)
    teams['playoffs_percentage'] = teams.apply(lambda row: get_playoff_percentage(row, teams_post), axis =1)
    teams['playoff'] = get_playoff_status(teams, teams_post)
    teams['points'] = teams['o_pts']
    teams['fgAttempted'] = teams['o_fga']
    teams['ftAttempted'] = teams ['o_fta']

    teams['team_weighted_playoff_attendance'] = teams.apply(
        lambda row: team_weighted_playoff_attendance(row, teams_post, teams),
        axis=1
    )
    
    coach_aggregates = coaches.groupby('coachID').agg(
        total_wins=('won', 'sum'),
        total_losses=('lost', 'sum'),
        average_wins=('won', 'mean'),
        average_losses=('lost', 'mean'),
        total_post_wins=('post_wins', 'sum'),
        total_post_losses=('post_losses', 'sum'),
    ).reset_index()
    
    coach_aggregates['coach_weighted_win_ratio'] = coach_aggregates.apply(
        lambda row: safe_divide(
            row['total_wins'] + POST_WIN_WEIGHT * row['total_post_wins'], 
            row['total_wins'] + row['total_losses'] + row['total_post_wins'] + row['total_post_losses']
        ), 
        axis=1
    )
    
    coach_aggregates['coach_playoff_attendance'] = coach_aggregates.apply(
        lambda row: coach_playoff_attendance(row, coaches),
        axis=1
    )
    
    coach_aggregates['coach_consistency'] = coach_aggregates.apply(
        lambda row: coach_consistency(row, coaches),
        axis=1
    )        
    
    # for each year of the team, append the coach data of the last coach on that year
    for i, row in teams.iterrows():
        year = row['year']
        tmID = row['tmID']
        last_coach = coaches[(coaches['year'] == year) & (coaches['tmID'] == row['tmID'])].iloc[-1]
        teams.loc[(teams['year'] == year) & (teams['tmID'] == tmID), 'coachID'] = last_coach['coachID']
        
    teams = pd.merge(teams, coach_aggregates, on='coachID', how='left')
        
    #https://www.teamrankings.com/nba/player/nikola-jokic
    teams['effective_fg_percentage'] = teams.apply(lambda row: safe_divide(row['o_fgm'] + 0.5 * row['o_3pm'], row['o_fga']), axis=1)
    #turnovers cant be done individually :     https://sportsjourneysinternational.com/sji-coaches-corner/turnover-percentage-the-second-most-important-factor-of-basketball-success/#:~:text=The%20easiest%20way%20to%20look%20at%20the%20individual,provides%20a%20good%20baseline%20for%20your%20individual%20statistics.
    # dataset['turnover_percentage'] = dataset.apply(lambda row: safe_divide(row['o_to'], (row['o_fga']- row['o_oreb'])) + row['o_to'] + (row['o_fta']*0.475))
    
    columns_to_keep = [
        'points', 
        'fgAttempted', 
        'ftAttempted',
        'fg_percentage', 
        'ft_percentage', 
        'three_percentage', 
        'true_shooting_percentage',
        'rebounds_per_minute', 
        'steals_per_minute', 
        'blocks_per_minute', 
        'assists_per_minute', 
        'assist_turnover_ratio',
        'effective_fg_percentage', 
        'playoffs_percentage', 
        'team_weighted_playoff_attendance',
        'coach_weighted_win_ratio', 
        'coach_playoff_attendance', 
        'coach_consistency',
        'playoff'
    ]

    return teams[columns_to_keep]

def coaches_feature_engineering(coaches):
    
    coach_aggregates = coaches.groupby('coachID').agg(
        total_wins=('won', 'sum'),
        total_losses=('lost', 'sum'),
        average_wins=('won', 'mean'),
        average_losses=('lost', 'mean'),
        total_post_wins=('post_wins', 'sum'),
        total_post_losses=('post_losses', 'sum'),
    ).reset_index()

    # Weighted Win Ratio - wins / (wins + losses) - post season wins value more
    coach_aggregates['coach_weighted_win_ratio'] = coach_aggregates.apply(
        lambda row: 
            safe_divide(
                row['total_wins'] + POST_WIN_WEIGHT * row['total_post_wins'], 
                row['total_wins'] + row['total_losses'] + row['total_post_wins'] + row['total_post_losses']
            ), 
            axis=1
    )
    # Playoff Attendance - percentage of seasons where coach made it to the playoffs
    coach_aggregates['coach_playoff_attendance'] = coach_aggregates.apply(
        lambda row: 
            coach_playoff_attendance(row, coaches),
            axis=1
    ) 
    # Coach Consistency - standard deviation of wins 
    coach_aggregates['coach_consistency'] = coach_aggregates.apply(
        lambda row: coach_consistency(row, coaches),
        axis=1
    )
    
    columns_to_keep = [
        "coachID", "coach_weighted_win_ratio", "coach_playoff_attendance", "coach_consistency"
    ]

    return coach_aggregates[columns_to_keep]

def coach_consistency(row, coaches):
    coach_seasons = coaches[coaches['coachID'] == row['coachID']]
    if len(coach_seasons) > 1:
        return coach_seasons['won'].std()
    else:
        return 0

def coach_playoff_attendance(row, coaches):
    total_seasons = len(coaches[coaches['coachID'] == row['coachID']])
    playoff_seasons = len(coaches[(coaches['coachID'] == row['coachID']) & (coaches['post_wins'] + coaches['post_losses'] > 0)])

    return safe_divide(playoff_seasons, total_seasons)


def get_playoff_percentage(row, teams_post):
    relevant_years = teams_post[teams_post['year'] <= (row['year'] - 1)]
    total_years = len(relevant_years['year'].unique())
    playoff_appearances = len(relevant_years[relevant_years['tmID'] == row['tmID']])

    return 0 if total_years == 0 else round(playoff_appearances / total_years, 3)


def get_playoff_status(teams, teams_post):
    teams_playoffs = pd.merge(teams, teams_post, how='left', left_on=['year', 'tmID'], right_on=['year', 'tmID'], indicator=True)
    teams_playoffs['playoff'] = (teams_playoffs['_merge'] == 'both').astype(int)
    teams_playoffs.drop(columns=['_merge'], inplace=True)

    return teams_playoffs['playoff']
    

def team_playoffs_consistency(row, teams_post, teams):
    team_seasons = teams[teams['tmID'] == row['tmID']]
    teams_playoffs_seasons = teams_post[teams_post['tmID'] == row['tmID']] 

    merged = pd.merge(team_seasons, teams_playoffs_seasons, how='left', left_on=['year', 'tmID'], right_on=['year', 'tmID'], indicator=True)
    merged['playoff'] = (merged['_merge'] == 'both').astype(int)
    merged.drop(columns=['_merge'], inplace=True)
    
    if len(merged) > 1:
        return merged['playoff'].std()
    else:
        return 0

def team_weighted_playoff_attendance(row, teams_post, teams, YEAR_WEIGHT=10):
    # value the attendence of playoffs of the higher years more
    team_seasons = teams[teams['tmID'] == row['tmID']]
    team_seasons['year_weight'] = team_seasons['year'].apply(lambda x: 1 + (x * YEAR_WEIGHT))
    
    team_seasons['weighted_playoffs'] = team_seasons['year_weight'] * team_seasons['playoff']
    
    total_playoffs = team_seasons['weighted_playoffs'].sum()
    total_years = team_seasons['year_weight'].sum()
    
    return safe_divide(total_playoffs, total_years)
    
def team_progression(row, teams_post, POST_WIN_WEIGHT=1.5, YEAR_WEIGHT=10):
    team_seasons = teams_post[teams_post['tmID'] == row['tmID']]
    team_seasons['year_weight'] = team_seasons['year'].apply(lambda x: 1 + (x * YEAR_WEIGHT))
    
    team_seasons['weighted_wins'] = team_seasons['W'] * team_seasons['year_weight']
    team_seasons['weighted_losses'] = team_seasons['L'] * team_seasons['year_weight']
    
    total_wins = team_seasons['weighted_wins'].sum()
    total_losses = team_seasons['weighted_losses'].sum()
    
    return safe_divide(total_wins, total_wins + total_losses)

def team_feature_engineering(dataset):
    teams = dataset['teams'].copy()
    teams_post = dataset['teams_post'].copy()

    team_aggregates = teams.groupby('tmID').agg(
        total_wins=('won', 'sum'),
        total_losses=('lost', 'sum'),
        average_wins=('won', 'mean'),
        average_losses=('lost', 'mean'),
    ).reset_index()
    
    team_aggregates['team_playoffs_consistency'] = team_aggregates.apply(
        lambda row: team_playoffs_consistency(row, teams_post, teams),
        axis=1
    )
    
    team_aggregates['team_weighted_playoff_attendance'] = team_aggregates.apply(
        lambda row: team_weighted_playoff_attendance(row, teams_post, teams),
        axis=1
    )
        
    team_aggregates['team_win_percentage'] = team_aggregates.apply(
        lambda row: safe_divide(row['total_wins'], row['total_wins'] + row['total_losses']),
        axis=1
    )
    
    team_aggregates['team_progression'] = team_aggregates.apply(
        lambda row: team_progression(row, teams_post),
        axis=1
    )

    columns_to_keep = [
        'tmID', 
        'team_playoffs_consistency', 
        'team_weighted_playoff_attendance', 
        'team_win_percentage',
        'team_progression'
    ]
    
    return team_aggregates[columns_to_keep]


def feature_engineering(dataset):
    dataset['coaches_data'] = coaches_feature_engineering(dataset['coaches'])
    dataset['players_teams'] = player_feature_engineering(dataset['players_teams'])
    dataset['training_dataset'] = prepTrainingDataset(dataset['teams'], dataset['teams_post'], dataset['coaches'])
    dataset['teams_data'] = team_feature_engineering(dataset)
    
    return dataset