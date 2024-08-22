import pandas as pd
import numpy as np 
import ast

from src.data.utils import clean_location, euclidean_distance, vector_from_points, dot_product, magnitude, goal_player_angle

import os
from tqdm import tqdm 

RAW_DATA_PATH = 'C:/Users/Aqsone/Desktop/football-data/data/raw/freekick_pass_shot.csv'
PROCESSED_DATA_PATH = 'C:/Users/Aqsone/Desktop/football-data/data/processed/clean_action_data.csv'

def read_raw_data():
    return pd.read_csv(RAW_DATA_PATH)

def get_action_data():
    df = read_raw_data()
    df.dropna(axis=1, how='all', inplace = True)

    # recover shots and free kick data
    shots_from_fk = []
    fk_assist_shot = []

    for i in tqdm(range(len(df))):
        if (df['pass_type'][i] == 'Free Kick'):
            rel_shot = df['pass_assisted_shot_id'][i] 
            if pd.notna(rel_shot):
                shots_from_fk.append(rel_shot)
                fk_assist_shot.append(df['id'][i])
                
    fk_data = df[df['id'].isin(fk_assist_shot)].drop('Unnamed: 0', axis = 1)
    shot_after_fk = df[df['id'].isin(shots_from_fk)].drop('Unnamed: 0', axis = 1)

    # select usefull shot columns
    shot_used_cols = ['id', 'shot_key_pass_id', 'duration', 'location', 'minute', 'period', 'position', 'shot_body_part', 'shot_freeze_frame', 'shot_technique', 
                'shot_open_goal', 'shot_statsbomb_xg']

    shot_after_fk = shot_after_fk[shot_used_cols]
    shot_after_fk.rename(columns={'id':'shot_id', 
                                'shot_key_pass_id':'fk_id', 
                                'duration':'shot_duration', 
                                'location':'shot_location', 
                                'position':'shooter_position'}, 
                        inplace=True)
    
    # select usefull free kick columns
    fk_used_cols = ['id', 'duration', 'location', 'pass_angle', 'pass_height', 'pass_length', 'pass_switch', 'position']

    fk_data = fk_data[fk_used_cols]
    fk_data.rename(columns={'id':'fk_id', 
                            'duration':'fk_duration', 
                            'location':'fk_location', 
                            'position':'fk_taker_position'}, 
                    inplace=True)
    
    # merge shot and fk_data
    action_data = shot_after_fk.merge(fk_data, how='left', on='fk_id')

    return action_data

def main():

    action_data = get_action_data()
    action_data.drop(['shot_id', 'fk_id'], axis=1, inplace=True) # drop id columns

    # convert locations to new columns
    action_data['shot_location'] = action_data['shot_location'].apply(clean_location)
    action_data['fk_location'] = action_data['fk_location'].apply(clean_location)

    shot_x = []
    shot_y = []
    for loc in action_data['shot_location']:
        shot_x.append(loc[0])
        shot_y.append(loc[1])
    action_data['shot_x'] = shot_x
    action_data['shot_y'] = shot_y

    fk_x = []
    fk_y = []
    for loc in action_data['fk_location']:
        fk_x.append(loc[0])
        fk_y.append(loc[1])
    action_data['fk_x'] = fk_x
    action_data['fk_y'] = fk_y

    action_data.drop(['shot_location', 'fk_location'], axis=1, inplace=True)

    # distance of shot to goal 
    action_data['distance_to_goal'] = [euclidean_distance([action_data.shot_x[i], action_data.shot_y[i]], [120, 40]) for i in range(len(action_data))]
    
    # encode boolean variables in 0/1
    action_data['shot_open_goal'] = np.where(action_data['shot_open_goal'], 1, 0)
    action_data['pass_switch'] = np.where(action_data['pass_switch'], 1, 0)

    # drop freeze frame, store it and get dummies 
    f_frame = action_data['shot_freeze_frame']
    action_data_encoded = pd.get_dummies(action_data.drop(['shot_freeze_frame'], axis=1), dtype=int)
    action_data_encoded['shot_freeze_frame'] = f_frame

    print(len(action_data_encoded.columns))

    players_close_dist = []
    players_goal_angle = []
    players_teammates = []

    # add distance, angle and teammates for k closest players
    k = 10
    for i in range(len(action_data_encoded)):
        dist = []
        angle = []
        teammate = []

        target_location = [action_data_encoded['shot_x'][i], action_data_encoded['shot_y'][i]]
        players = ast.literal_eval(action_data_encoded.shot_freeze_frame[i])
        for player in players:
            player['distance'] = euclidean_distance(player['location'], target_location)
            player['angle'] = goal_player_angle(player['location'], target_location)

        sorted_players = sorted(players, key=lambda x: x['distance'])

        closest_k_players = sorted_players[:k]

        for player in closest_k_players:
            dist.append(player['distance'])
            angle.append(player['angle'])
            teammate.append(1 if player['teammate'] else 0)

        players_close_dist.append(dist)
        players_goal_angle.append(angle)
        players_teammates.append(teammate)

    action_data_encoded['close_players_distance'] = players_close_dist
    action_data_encoded['close_players_angle'] = players_goal_angle
    action_data_encoded['teammates'] = players_teammates

    distance_expanded = pd.DataFrame(action_data_encoded['close_players_distance'].tolist(), columns=[f'distance_player_{i+1}' for i in range(action_data_encoded['close_players_distance'].apply(len).max())])
    angle_expanded = pd.DataFrame(action_data_encoded['close_players_angle'].tolist(), columns=[f'angle_player_{i+1}' for i in range(action_data_encoded['close_players_angle'].apply(len).max())])
    teammate_expanded = pd.DataFrame(action_data_encoded['teammates'].tolist(), columns=[f'teammates_player_{i+1}' for i in range(action_data_encoded['teammates'].apply(len).max())])

    df = pd.concat([action_data_encoded.drop(columns=['close_players_distance', 'close_players_angle', 'teammates']), distance_expanded, angle_expanded, teammate_expanded], axis=1)
    df.dropna(inplace = True) # drop rows for actions that do not have k players in freeze frame 

    # change col teammates players to integers
    for col in df.columns[-3:]:
        df[col] = df[col].astype(int)

    # drop useless columns
    df.drop(['shot_freeze_frame', 'shot_duration'], axis = 1, inplace = True)

    df.to_csv(PROCESSED_DATA_PATH, index = False)


if __name__ == '__main__':
    main()