import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from pytorch_lightning import LightningModule 
import torch.nn as nn
import wandb
from torch_geometric.nn import GATConv

class GNNModel(LightningModule):
    def __init__(self, conv_type, conv1_size, conv2_size, fc1_size, activation):
        super(GNNModel, self).__init__()
        self.save_hyperparameters()
        if conv_type == 'GCNConv':
            self.conv1 = GCNConv(71, conv1_size)
            self.conv2 = GCNConv(conv1_size, conv2_size)
        elif conv_type == 'GATConv':
            self.conv1 = GATConv(71, conv1_size)
            self.conv2 = GATConv(conv1_size, conv2_size)
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}")
        self.fc1 = nn.Linear(conv2_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 1)
        self.activation = activation
        self.criterion = torch.nn.L1Loss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = getattr(F, self.activation)(x)
        x = self.conv2(x, edge_index)
        x = getattr(F, self.activation)(x)
        
        x = global_mean_pool(x, batch)  # global pooling
        x = self.fc1(x)
        x = getattr(F, self.activation)(x)
        x = self.fc2(x)
        
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch.y)
        self.log('train_loss', loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the GNN model.

        Parameters:
        batch (object): The batch of data to be validated.
        batch_idx (int): The index of the batch.

        Returns:
        loss (float): The validation loss of the model.
        """
        output = self(batch)
        loss = self.criterion(output, batch.y)
        self.log('val_loss', loss, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self, lr=0.01):
        lr = wandb.config.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


if __name__=='__main__':
    model = GNNModel()

