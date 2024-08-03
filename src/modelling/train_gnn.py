from src.dataset import read_data
from src.config import PROCESSED_DATA_PATH, BATCH_SIZE
from src.modelling.gnn import GNNModel
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


graphs = read_data(data_path=PROCESSED_DATA_PATH)

# Split the data into training and testing sets
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

model = GNNModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Load the data

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
    return loss.item()

def test(loader):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss += criterion(output, data.y).item()
    return loss / len(loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on {device}')
model = model.to(device)

for epoch in range(1, 201):
    train_loss = train()
    test_loss = test(test_loader)
    print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
