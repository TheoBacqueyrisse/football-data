'''
This script will be used to collect data from the LaLiga 2015-2016 season.
380 open access match data
We collect every from sb.matches():
    - match_id : match_id
    - home and away team : home_team + away_team
    - match_day : match_week
    - score : home_score + away_score

For every match_id, we collect event data with sb.events()
    - Lineups / Tactics
    - Duals
    - Dribbles
    - Passes
    - Shots
'''

import pandas as pd
import numpy as np 

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from statsbombpy import sb

def main():
    '''
    This function loads from StatsBomb API the event data from the 15/16 LaLiga season and saves this data
    '''
    # competition_id = 11 : LaLiga, season_id = 27 : 15/16, most data available
    liga_1516 = sb.matches(competition_id=11, season_id=27) 
    liga_1516 = liga_1516[['match_id', 'home_team', 'away_team', 'match_week', 'home_score', 'away_score']]
    liga_1516.to_csv('data/liga_1516.csv', index=False)

    list_match_ids = liga_1516['match_id'].unique().tolist()
    del(liga_1516)

    match_events = []
    for m_id in tqdm(list_match_ids):
        # collect the events from a match_id
        events_id = sb.events(match_id=m_id)
        events_id['match_id'] = m_id

        match_events.append(events_id)

    # concatenate all events from all matches
    events_df = pd.concat(match_events, axis=0)

    # save the dataframe
    events_df.to_csv('data/all_events_liga_1516.csv', index=False)

if __name__ == '__main__':
    main()