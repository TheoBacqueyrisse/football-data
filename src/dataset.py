from pytorch_lightning import LightningDataModule
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, PREPROCESSOR_PATH
import pandas as pd
import ast
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import PowerTransformer
import numpy as np


# Create data loaders
class GraphDataModule(LightningDataModule):
    def __init__(self, train_graphs, test_graphs, batch_size=32):
        super().__init__()
        self.train_graphs = train_graphs
        self.test_graphs = test_graphs
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True, num_workers=11, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.test_graphs, batch_size=self.batch_size, num_workers=11, persistent_workers=True)


def read_data(data_path=PROCESSED_DATA_PATH):
    with open(data_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs


def make_data(data_path=RAW_DATA_PATH, save=False):

    data = pd.read_csv(data_path)

    # Filter for rows where there is a shot
    data = data[data['type'] == 'Shot']  # Assuming 'type' is the column indicating the event type

    data['freeze_frame_parsed'] = data['shot_freeze_frame'].map(parse_freeze_frame)

    graphs = data.apply(create_graph, axis=1).tolist()

   
    if save:
        boxcox_transformer = PowerTransformer(method='box-cox')
        y = boxcox_transformer.fit_transform(data['shot_statsbomb_xg'].values.reshape(-1,1)).ravel()

        with open(PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(boxcox_transformer, f)

        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(graphs, f)

    return graphs

def create_graph(row):
    players = row['freeze_frame_parsed']
    x = []
    edge_index = []
    y = [row['shot_statsbomb_xg']]  # assuming 'expected_goal' is the target column
    
    for player in players:
        x.append(player['location'])
    
    num_players = len(players)
    for i in range(num_players):
        for j in range(i+1, num_players):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    y = boxcox_transformer.transform(np.array(y).reshape(1,-1))

    y = torch.tensor(y, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def parse_freeze_frame(freeze_frame):
    players = ast.literal_eval(freeze_frame)
    return players


if __name__=='__main__':
    with open(PREPROCESSOR_PATH, 'rb') as f:
        boxcox_transformer = pickle.load( f)
    graphs = make_data(save=True)
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    data_module = GraphDataModule(train_graphs, test_graphs)
