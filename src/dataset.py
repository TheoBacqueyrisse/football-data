from pytorch_lightning import LightningDataModule
from src.config import RAW_DATA_PATH, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, PROCESSED_DATA_PATH
import pandas as pd
import numpy as np
import ast
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pickle

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

    data['freeze_frame_parsed'] = data['shot_freeze_frame'].map(parse_freeze_frame)

    # Create graph data

    graphs = data.apply(create_graph, axis=1).tolist()

    if save:
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(graphs, f)

    return graphs

def create_graph(row):
    
    node_feature_matrix = pd.DataFrame()

    player_loc = []
    players_pos = []

    edge_index = []
    edge_attr = []

    y = [row['shot_statsbomb_xg']]

    # add shooter and passer
    player_loc.append(ast.literal_eval(row['shot_location']))
    player_loc.append(ast.literal_eval(row['fk_location']))
    
    # add players from freeze frame
    players = row['freeze_frame_parsed']
    num_players = len(players)
    for player in players:
        player_loc.append(player['location'])

    for col in row.keys():
        if col not in ['shot_statsbomb_xg', 'shot_location', 'shooter_position', 'fk_location', 'fk_taker_position', 'freeze_frame_parsed', 'shot_freeze_frame']:
            if col[:4] == 'shot':
                node_feature_matrix[col] = [row[col]] + (num_players+1)*[0]
            else:
                node_feature_matrix[col] = [0] + [row[col]] + num_players*[0]
    
    for i in range(num_players):
        for j in range(i+1, num_players):
            edge_index.append([i, j])
            edge_index.append([j, i])
            if players[i]['teammate'] == players[j]['teammate']:
                edge_attr.append(1)
            else:
                edge_attr.append(0)
    
    node_feature_matrix = torch.tensor(node_feature_matrix.values)
    player_loc = torch.tensor(player_loc, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor(y, dtype=torch.float)
    
    return Data(x=node_feature_matrix, edge_index=edge_index, edge_attr = edge_attr, pos = player_loc, y=y)

def parse_freeze_frame(freeze_frame):
    players = ast.literal_eval(freeze_frame)
    return players


if __name__=='__main__':
    graphs = make_data(save=True)
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    data_module = GraphDataModule(train_graphs, test_graphs)
