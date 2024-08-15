import os


RAW_DATA_PATH = os.path.join('data', 'raw', 'encoded_action_data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_actions_data.pkl')
BATCH_SIZE = 64
MAX_EPOCHS = 50