import pandas as pd
from src.config import XGB_DATA_PATH
import wandb
import yaml
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

with open("C:/Users/Aqsone/Desktop/football-data/src/modelling/xgboost/sweep_xgb.yaml") as file:
    sweep_config = yaml.safe_load(file)

def read_data():
    return  pd.read_csv(XGB_DATA_PATH)

def train_xgboost(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        data = read_data() 

        # test to remove columns
        cols_to_keep = ['shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal', 'distance_player_1', 'distance_player_2', 'distance_player_3', 'distance_player_4', 
                        'angle_player_1', 'angle_player_2', 'angle_player_3', 'angle_player_4', 'teammates_player_1', 'teammates_player_2', 'teammates_player_3', 'teammates_player_4']
        X_train, X_test, y_train, y_test = train_test_split(data[cols_to_keep], data.shot_statsbomb_xg, test_size=0.25, random_state=42)

        # X_train, X_test, y_train, y_test = train_test_split(data.drop('shot_statsbomb_xg', axis = 1), data.shot_statsbomb_xg, test_size=0.25, random_state=42)
        
        model = xgb.XGBRegressor(
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            gamma=config.gamma,
            objective='reg:absoluteerror'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        wandb.log({"mae": mae})

sweep_id = wandb.sweep(sweep_config, entity='thomas-toulouse', project='xgb-football-xg')
wandb.agent(sweep_id, train_xgboost)
