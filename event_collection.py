'''
This script will be used to collect data from the LaLiga 2015-2016 season.
380 open access match data
We collect every from sb.matches():
    - match_id : match_id
    - home and away team : home_team + away_team
    - match_day : match_week
    - score : home_score + away_score

For every match_id, we collect event data with sb.events()
    - 
    - 
    - 
'''

import pandas as pd
import numpy as np 

from statsbombpy import sb

def main():
    # competition_id = 11 : LaLiga, season_id = 27 : 15/16, most data available
    liga_1516 = sb.matches(competition_id=11, season_id=27) 
    liga_1516 = liga_1516[['match_id', 'home_team', 'away_team', 'match_week', 'home_score', 'away_score']]

    list_match_ids = liga_1516['match_id'].unique().tolist()

    for m_id in list_match_ids:
        events_id = sb.events(match_id=m_id)
