def safe_divide(a,b, threshold = 0, default = 0):
    return a / b if b > threshold else default

def aggregatePlayerData(dataset):
    # Only total rebounds is important
    dataset.drop(columns=['oRebounds', 'dRebounds'], inplace=True)
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

    ## HOLY GRAIL: https://www.basketball-reference.com/about/factors.html
    ## Dream Statistic -> PER 

    #https://www.teamrankings.com/nba/player/nikola-jokic
    dataset['effective_fg_percentage'] = dataset.apply(lambda row: safe_divide(row['fgMade'] + 0.5 * row['threeMade'], row['fgAttempted'], min_fg_attempts), axis=1)
    #turnovers cant be done individually :     https://sportsjourneysinternational.com/sji-coaches-corner/turnover-percentage-the-second-most-important-factor-of-basketball-success/#:~:text=The%20easiest%20way%20to%20look%20at%20the%20individual,provides%20a%20good%20baseline%20for%20your%20individual%20statistics.
    #https://www.teamrankings.com/nba/player/stephen-curry
    dataset['free_throw_rate'] = dataset.apply(lambda row: safe_divide(row['ftMade'], row['ftAttempted'], min_fg_attempts), axis=1)
   
   
    columns_to_keep = [
        'playerID', 'year', 'stint', 'tmID', 'GP', 'GS', 'minutes', 'points', 'fgAttempted', 'ftAttempted',
        'fg_percentage', 'ft_percentage', 'three_percentage', 'true_shooting_percentage',
        'rebounds_per_minute', 'steals_per_minute', 'blocks_per_minute', 'assists_per_minute',
        'assist_turnover_ratio', 'effective_fg_percentage', 'free_throw_rate'
    ]

    return dataset[columns_to_keep]


def aggregateDataset(dataset):
    aggregated_data = aggregatePlayerData(dataset['players_teams'])
    
    csv_filename = 'aggregated_players_teams.csv'
    aggregated_data.to_csv(csv_filename, index=False)
    print(f"Aggregated data has been saved to {csv_filename}")

    return aggregated_data