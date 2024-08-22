import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from src.config import XGB_DATA_PATH, XGB_MODEL_PATH, XGB_RES_PATH, XGB_FEATURE_IMPORTANCE_PATH

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import wandb

SWEEP_ID = "thomas-toulouse/xgb-football-xg/r25vjm3e"

def read_data():
    return  pd.read_csv(XGB_DATA_PATH)


def xgb_model(X_train, y_train, params, save = False):

    model = xgb.XGBRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        gamma=params['gamma'],
        objective='reg:absoluteerror'
    )
        
    model.fit(X_train, y_train)

    if save :
        model.save_model(XGB_MODEL_PATH)
    
    return model

def plot_results(y_test, y_pred):

    plt.figure(figsize=(8, 8))

    plt.scatter(y_test, y_pred)

    max = np.max(y_test)
    plt.plot([0, max], [0, max], color='red', linestyle='-', linewidth=2)

    plt.savefig(XGB_RES_PATH)


def get_feature_imoprtance(model):
    plt.figure(figsize=(20, 20))
    xgb.plot_importance(model, max_num_features = 20)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(XGB_FEATURE_IMPORTANCE_PATH)


def main() : 
    # read data
    df = read_data()

    # train test split
    X = df.drop('shot_statsbomb_xg', axis = 1)
    y = df.shot_statsbomb_xg
    
    cols_to_keep = ['shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal', 'distance_player_1', 'distance_player_2', 'distance_player_3', 'distance_player_4', 
                    'angle_player_1', 'angle_player_2', 'angle_player_3', 'angle_player_4', 'teammates_player_1', 'teammates_player_2', 'teammates_player_3', 'teammates_player_4']
    X_train, X_test, y_train, y_test = train_test_split(X[cols_to_keep], y, test_size=0.25, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

    # train xgb

    api = wandb.Api()    
    sweep = api.sweep(SWEEP_ID)
    best_run = sweep.best_run()
    best_parameters = best_run.config

    model = xgb_model(X_train, y_train, params = best_parameters, save=True)

    # predict, plot mae, get results on test and feature importance
    test_pred = model.predict(X_test)

    # model = xgb_model(X_train.drop('minute', axis = 1), y_train, params = best_parameters, save=True)

    # # predict, plot mae, get results on test and feature importance
    # test_pred = model.predict(X_test.drop('minute', axis = 1))

    mae = mean_absolute_error(y_test, test_pred)
    print(mae)

    plot_results(y_test=y_test, y_pred=test_pred)
    get_feature_imoprtance(model=model)


if __name__ == '__main__':
    main()