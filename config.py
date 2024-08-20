import os

RAW_DATA_PATH = os.path.join('data', 'raw', 'encoded_action_data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_encoded_action_data.pkl')
PREPROCESSOR_PATH = os.path.join('src', 'modelling', 'boxcox.pkl')

XGB_DATA_PATH = os.path.join('data', 'raw', 'encoded_action_data_xgb_v2.csv')
XGB_MODEL_PATH = os.path.join('src', 'modelling', 'xgboost', 'xgb.json')
XGB_RES_PATH = os.path.join('src', 'modelling', 'xgboost', 'results.png')
XGB_FEATURE_IMPORTANCE_PATH = os.path.join('src', 'modelling', 'xgboost', 'feature_importance.png')

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
MAX_EPOCHS = 100