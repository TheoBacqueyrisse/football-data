from src.dataset import read_data
from src.config import PROCESSED_DATA_PATH, BATCH_SIZE
from src.modelling.gnn import GNNModel
from sklearn.model_selection import train_test_split
from src.dataset import GraphDataModule
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger

# Initialize W&B
wandb.init(project='football-xg')

# Create a W&B logger
wandb_logger = WandbLogger()


graphs = read_data(data_path=PROCESSED_DATA_PATH)

# Split the data into training and testing sets
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
data_module = GraphDataModule(train_graphs, test_graphs, batch_size=BATCH_SIZE)

model = GNNModel()



trainer = Trainer(max_epochs=200, logger=wandb_logger)
trainer.fit(model, data_module)