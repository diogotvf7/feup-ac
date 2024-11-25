import pandas as pd

WEIGHTED_STATS = ['points', 'fgAttempted', 'ftAttempted', 'fg_percentage', 'ft_percentage',
                    'three_percentage', 'true_shooting_percentage', 'rebounds_per_minute',
                    'steals_per_minute', 'blocks_per_minute', 'assists_per_minute',
                    'assist_turnover_ratio', 'effective_fg_percentage']

AGGREGATION_FUNCTIONS = {        
    'points': 'sum',           
    'fgAttempted': 'sum',      
    'ftAttempted': 'sum',      
    'fg_percentage': 'mean',   
    'ft_percentage': 'mean',   
    'three_percentage': 'mean',
    'true_shooting_percentage': 'mean', 
    'rebounds_per_minute': 'mean',      
    'steals_per_minute': 'mean',        
    'blocks_per_minute': 'mean',        
    'assists_per_minute': 'mean',       
    'assist_turnover_ratio': 'mean',    
    'effective_fg_percentage': 'mean',  
}

STATS = ['points',                          
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
    'effective_fg_percentage']

def calculate_avg_stats_rookie(players, change_year=False):

    rookie_years = players.groupby('playerID')['year'].min().reset_index()

    rookies = players.merge(rookie_years, on=['playerID', 'year'])    
    rookie_avg_stats = rookies[STATS].mean()

    stats_to_round = ['points', 'fgAttempted', 'ftAttempted']
    rookie_avg_stats[stats_to_round] = rookie_avg_stats[stats_to_round].round()
    if change_year:
        rookie_avg_stats['year'] = 1
        rookie_avg_stats['stint'] = 0
    return rookie_avg_stats.to_dict()


def calculate_weighted_avg_player_stats(merged_data, weighted_stats):
    weighted_data = merged_data.groupby([ 'playerID', 'tmID' ]).apply(
        lambda x: pd.Series({
            stat: (x[stat].sum()) / x['year_weight'].sum()
            for stat in weighted_stats
        })
    ).reset_index()
    
    return weighted_data

def calculate_avg_player_stats(merged_data):
    avg_data = merged_data.groupby(['playerID', 'tmID'])[STATS].mean().reset_index()
    return avg_data


def calculate_mixed_avg_player_stats(merged_data, weighted_stats, unweighted_stats):

    mixed_data = merged_data.groupby(['playerID', 'tmID']).apply(
        lambda x: pd.Series({
            stat: (x[stat].sum()) / x['year_weight'].sum()
            if stat in weighted_stats else x[stat].mean()  # Simple average for unweighted stats
            for stat in weighted_stats + unweighted_stats
        })
    ).reset_index()
    
    return mixed_data



def teams_playoffs(teams_stats, teams_post_df, teams_df):
    teams = teams_stats['tmID']
    playoffs_percentage = {}

    for team in teams:
        times_competed = len(teams_df[teams_df['tmID'] == team])
        times_playoffs = len(teams_post_df[teams_post_df['tmID'] == team])

        if (times_competed > 0 and times_playoffs > 0):
            playoffs_percentage[team] = round(times_playoffs / times_competed, 2)
        else:
            playoffs_percentage[team] = 0    

    return playoffs_percentage

def calculate_new_coach_avg_stats(coaches):
    coach_avg_stats = coaches.groupby('coachID')[["average_wins", "average_losses", "weighted_win_ratio", "playoff_attendance", "coach_consistency"]].mean()
    return coach_avg_stats.to_dict()

def create_final_dataset(coaches_df, teams_post_df, teams_df, players, target_year=9, aggregation_method='none'):
    s10 = pd.read_csv('dataset/finals/s10.csv') 
    new_coach_avg_stats = calculate_new_coach_avg_stats(coaches_df)
    new_player_avg_stats = calculate_avg_stats_rookie(players, True)

    players = players.drop(columns=['tmID', 'GP', 'GS', 'minutes'])
    merged_data = pd.merge(s10, players, on='playerID', how='left')
    merged_data = merged_data.fillna(new_player_avg_stats)
    merged_data = merged_data.merge(coaches_df, on='coachID', how='left')
    merged_data = merged_data.fillna(new_coach_avg_stats)

    if aggregation_method == 'fully_weighted':
        merged_data['year_weight'] = merged_data['year'].apply(lambda y: 1 / (target_year - y + 1))
        for stat in WEIGHTED_STATS:
            merged_data[stat] = merged_data[stat] * merged_data['year_weight']
        player_avg = calculate_weighted_avg_player_stats(merged_data, WEIGHTED_STATS)

    elif aggregation_method == 'fully_average':
        player_avg = calculate_avg_player_stats(merged_data)

    elif aggregation_method == 'mixed':
        mix_weighted = ['points', 'fgAttempted', 'ftAttempted']
        mix_unweighted = ['fg_percentage', 'ft_percentage', 'three_percentage', 'true_shooting_percentage', 
                          'rebounds_per_minute', 'steals_per_minute', 'blocks_per_minute', 'assists_per_minute', 
                          'assist_turnover_ratio', 'effective_fg_percentage']
        merged_data['year_weight'] = merged_data['year'].apply(lambda y: 1 / (target_year - y + 1))
        for stat in mix_weighted:
            merged_data[stat] = merged_data[stat] * merged_data['year_weight']
        player_avg = calculate_mixed_avg_player_stats(merged_data, mix_weighted, mix_unweighted)

    else:  
        player_avg = merged_data

    team_stats = player_avg.groupby('tmID').agg(AGGREGATION_FUNCTIONS).reset_index()
    playoffs_percentage = teams_playoffs(team_stats, teams_post_df, teams_df)

    team_stats['playoffs_percentage'] = team_stats['tmID'].map(playoffs_percentage)
    team_stats.to_csv('dataset/finals/s10_team_stats.csv', index=False)
    return team_stats