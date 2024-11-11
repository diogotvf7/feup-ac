import pandas as pd
from dataset_preparation.create_final_dataset import teams_playoffs


def safe_divide(a,b, threshold = 0, default = 0):
    return a / b if b > threshold else default

def prepPlayerData(dataset):
    # Only total rebounds is important
    dataset.drop(columns=['oRebounds', 'dRebounds', 'stint'], inplace=True)

    ##DOING THIS TO MERGE PLAYERS WITH DIFFERENT STATS FOR THE SAME YEAR (BECAUSE DIFFERENT COACHES) + NO INTERMIDIATE AVERAGES PER PLAYER GAVE THE BEST RESULTS
    dataset = dataset.groupby(['playerID', 'year'], as_index=False).agg({
        'tmID': 'first',       
        'GP': 'sum',           
        'GS': 'sum',           
        'minutes': 'sum',       
        'points': 'sum',        
        'fgMade': 'sum',        
        'fgAttempted': 'sum',   
        'ftMade': 'sum',        
        'ftAttempted': 'sum',   
        'threeMade': 'sum',     
        'threeAttempted': 'sum',  
        'rebounds': 'sum',      
        'steals': 'sum',        
        'blocks': 'sum',        
        'assists': 'sum',       
        'turnovers': 'sum'      
    })
    min_fg_attempts = 5

    # Shooting Metrics
    dataset['fg_percentage'] = dataset.apply(lambda row: safe_divide(row['fgMade'], row['fgAttempted'], min_fg_attempts), axis=1)
    dataset['ft_percentage'] = dataset.apply(lambda row: safe_divide(row['ftMade'], row['ftAttempted']), axis=1)
    dataset['three_percentage'] = dataset.apply(lambda row: safe_divide(row['threeMade'], row['threeAttempted']), axis=1)
    dataset['true_shooting_percentage'] = dataset.apply(lambda row: safe_divide(row['points'], 2 * (row['fgAttempted'] + 0.44 * row['ftAttempted']), min_fg_attempts), axis=1)
    ## PerMinute Stats ##
    dataset['rebounds_per_minute'] = dataset.apply(lambda row: safe_divide(row['rebounds'], row['minutes']), axis=1)
    dataset['steals_per_minute'] = dataset.apply(lambda row: safe_divide(row['steals'], row['minutes']), axis=1)
    dataset['blocks_per_minute'] = dataset.apply(lambda row: safe_divide(row['blocks'], row['minutes']), axis=1)
    dataset['assists_per_minute'] = dataset.apply(lambda row: safe_divide(row['assists'], row['minutes']), axis=1)
    dataset['assist_turnover_ratio'] = dataset.apply(lambda row: safe_divide(row['assists'], row['turnovers'], default=row['assists']), axis=1)

    ## Dream Statistic -> PER 

    #https://www.teamrankings.com/nba/player/nikola-jokic
    dataset['effective_fg_percentage'] = dataset.apply(lambda row: safe_divide(row['fgMade'] + 0.5 * row['threeMade'], row['fgAttempted'], min_fg_attempts), axis=1)
    #turnovers cant be done individually :     https://sportsjourneysinternational.com/sji-coaches-corner/turnover-percentage-the-second-most-important-factor-of-basketball-success/#:~:text=The%20easiest%20way%20to%20look%20at%20the%20individual,provides%20a%20good%20baseline%20for%20your%20individual%20statistics.
    
   
   
    columns_to_keep = [
        'playerID', 'year',  'tmID', 'GP', 'GS', 'minutes', 'points', 'fgAttempted', 'ftAttempted',
        'fg_percentage', 'ft_percentage', 'three_percentage', 'true_shooting_percentage',
        'rebounds_per_minute', 'steals_per_minute', 'blocks_per_minute', 'assists_per_minute',
        'assist_turnover_ratio', 'effective_fg_percentage'
    ]

    return dataset[columns_to_keep]


def prepTrainingDataset(datasets):

    dataset = datasets['teams']
    
    # Shooting Metrics
    dataset['fg_percentage'] = dataset.apply(lambda row: safe_divide(row['o_fgm'], row['o_fga']), axis=1)
    dataset['ft_percentage'] = dataset.apply(lambda row: safe_divide(row['o_ftm'], row['o_fta']), axis=1)
    dataset['three_percentage'] = dataset.apply(lambda row: safe_divide(row['o_3pm'], row['o_3pa']), axis=1)
    dataset['true_shooting_percentage'] = dataset.apply(lambda row: safe_divide(row['o_pts'], 2 * (row['o_fga'] + 0.44 * row['o_fta'])), axis=1)
    ## PerMinute Stats ##
    dataset['rebounds_per_minute'] = dataset.apply(lambda row: safe_divide(row['o_reb'], row['min']), axis=1)
    dataset['steals_per_minute'] = dataset.apply(lambda row: safe_divide(row['o_stl'], row['min']), axis=1)
    dataset['blocks_per_minute'] = dataset.apply(lambda row: safe_divide(row['o_blk'], row['min']), axis=1)
    dataset['assists_per_minute'] = dataset.apply(lambda row: safe_divide(row['o_asts'], row['min']), axis=1)
    dataset['assist_turnover_ratio'] = dataset.apply(lambda row: safe_divide(row['o_asts'], row['o_to'], default=row['o_asts']), axis=1)
    dataset['playoffs_percentage'] = dataset.apply(lambda row: get_playoff_percentage(row, datasets['teams_post']), axis =1)
    dataset['playoff'] = get_playoff_status(datasets['teams'], datasets['teams_post'])
    dataset['points'] = dataset['o_pts']
    dataset['fgAttempted'] = dataset['o_fga']
    dataset['ftAttempted'] = dataset ['o_fta']


    #https://www.teamrankings.com/nba/player/nikola-jokic
    dataset['effective_fg_percentage'] = dataset.apply(lambda row: safe_divide(row['o_fgm'] + 0.5 * row['o_fgm'], row['o_fga']), axis=1)
    #turnovers cant be done individually :     https://sportsjourneysinternational.com/sji-coaches-corner/turnover-percentage-the-second-most-important-factor-of-basketball-success/#:~:text=The%20easiest%20way%20to%20look%20at%20the%20individual,provides%20a%20good%20baseline%20for%20your%20individual%20statistics.
    # dataset['turnover_percentage'] = dataset.apply(lambda row: safe_divide(row['o_to'], (row['o_fga']- row['o_oreb'])) + row['o_to'] + (row['o_fta']*0.475))
    
    dataset = dataset.drop(dataset[dataset.year == 10].index)

    columns_to_keep = [
        'points', 'fgAttempted', 'ftAttempted',
        'fg_percentage', 'ft_percentage', 'three_percentage', 'true_shooting_percentage',
        'rebounds_per_minute', 'steals_per_minute', 'blocks_per_minute', 'assists_per_minute', 'assist_turnover_ratio',
        'effective_fg_percentage', 'playoffs_percentage', 'playoff'
    ]

    return dataset[columns_to_keep]

def get_playoff_percentage(row, teams_post):
        
        relevant_years = teams_post[teams_post['year'] <= row['year']]
        total_years = len(relevant_years['year'].unique())
        playoff_appearances = len(relevant_years[relevant_years['tmID'] == row['tmID']])

        return (playoff_appearances / total_years).__round__(3)

def get_playoff_status(teams, teams_post):

    teams_playoffs = pd.merge(teams, teams_post, how='left', left_on=['year', 'tmID'], right_on=['year', 'tmID'], indicator=True)
    teams_playoffs['playoff'] = (teams_playoffs['_merge'] == 'both').astype(int)
    teams_playoffs.drop(columns=['_merge'], inplace=True)

    return teams_playoffs['playoff']

def calculate(dataset):
    player_aggregated_data = prepPlayerData(dataset['players_teams'])
    training_dataset = prepTrainingDataset(dataset)
    return player_aggregated_data, training_dataset
    

def modelling(dataset):
    players_teams, training_dataset = calculate(dataset)
    dataset['players_teams'] = players_teams
    dataset['training_dataset'] = training_dataset

    return dataset