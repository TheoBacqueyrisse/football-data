import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from src.config import XGB_DATA_PATH, XGB_MODEL_PATH, XGB_RES_PATH, XGB_FEATURE_IMPORTANCE_PATH

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def read_data():
    return  pd.read_csv(XGB_DATA_PATH).drop(['Unnamed: 0', 'shot_duration'], axis = 1)


def xgb_model(X_train, y_train, save = False):
    
    dtrain_reg = xgb.DMatrix(X_train, y_train)

    n = 100
    params = {'objective' : 'reg:absoluteerror'}

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        )

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # train xgb
    model = xgb_model(X_train, y_train, save=True)

    # predict, plot mae, get results on test and feature importance
    test_pred = model.predict(xgb.DMatrix(X_test))

    mae = mean_absolute_error(y_test, test_pred)
    print(mae)

    plot_results(y_test=y_test, y_pred=test_pred)
    get_feature_imoprtance(model=model)


if __name__ == '__main__':
    main()