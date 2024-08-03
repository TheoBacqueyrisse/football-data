import os


RAW_DATA_PATH = os.path.join('data', 'raw', 'freekick_pass_shot.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_freekick_pass_shot.pkl')
BATCH_SIZE = 64
MAX_EPOCHS = 50