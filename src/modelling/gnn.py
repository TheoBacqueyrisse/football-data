import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from pytorch_lightning import LightningModule 

class GNNModel(LightningModule):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(71, 16)  # 2 input features (x, y) and 16 output features
        self.conv2 = GCNConv(16, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        self.criterion = torch.nn.MSELoss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)  # global pooling
        x = self.fc1(x)
        x = F.relu(x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__=='__main__':
    model = GNNModel()

