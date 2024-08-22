import os

RAW_DATA_PATH = os.path.join('data', 'raw', 'encoded_action_data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_encoded_action_data.pkl')
PREPROCESSOR_PATH = os.path.join('src', 'modelling', 'boxcox.pkl')

XGB_DATA_PATH = os.path.join('data', 'processed', 'clean_action_data.csv')
XGB_MODEL_PATH = os.path.join('src', 'modelling', 'xgboost', 'xgb_few_var.json')
XGB_RES_PATH = os.path.join('src', 'modelling', 'xgboost', 'results_few_var.png')
XGB_FEATURE_IMPORTANCE_PATH = os.path.join('src', 'modelling', 'xgboost', 'feature_importance_few_var.png')

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
MAX_EPOCHS = 100