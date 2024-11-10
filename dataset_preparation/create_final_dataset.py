import pandas as pd


aggregation_functions = {        
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

def calculate_avg_stats_rookie(players):
    stats = ['points',                          
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
    rookie_years = players.groupby('playerID')['year'].min().reset_index()

    rookies = players.merge(rookie_years, on=['playerID', 'year'])    
    
    rookie_avg_stats = rookies[stats].mean()

    return rookie_avg_stats.to_dict()


#TODO: Implement this function to predict 
def predict_players_performance_target_season(players):
    pass


def teams_playoffs(teams_stats, teams_post_df, teams_df):
    teams = teams_stats['tmID']
    playoffs_percentage = {}

    for team in teams:
        times_competed = len(teams_df[teams_df['tmID'] == team])
        times_playoffs = len(teams_post_df[teams_post_df['tmID'] == team])

        if (times_competed > 0 and times_playoffs > 0):
            playoffs_percentage[team] = round(times_playoffs / times_competed, 2)

    return playoffs_percentage


def create_final_dataset(teams_post_df, teams_df, players):

    s10 = pd.read_csv('dataset/finals/s10.csv') 
    rookie_avg_stats = calculate_avg_stats_rookie(players)

    #Fill information regarding rookies
    players = players.drop(columns=['tmID', 'GP', 'GS', 'minutes'])
    merged_data = pd.merge(s10, players, on='playerID', how='left')
    merged_data = merged_data.fillna(rookie_avg_stats)

    #Calculate Team Stats
    team_stats = merged_data.groupby(['tmID']).agg(aggregation_functions).reset_index()
    team_stats = team_stats.round(3)
    playoffs_percentage = teams_playoffs(team_stats, teams_post_df, teams_df)

    team_stats['playoffs_percentage'] = team_stats['tmID'].map(playoffs_percentage)
    team_stats.to_csv('dataset/finals/s10_team_stats.csv', index=False)    

    return team_stats
