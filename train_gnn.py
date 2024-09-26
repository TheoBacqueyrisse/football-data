from src.dataset import read_data
from src.config import PROCESSED_DATA_PATH, TRAIN_BATCH_SIZE, MAX_EPOCHS
from src.modelling.gnn import GNNModel
from sklearn.model_selection import train_test_split
from src.dataset import GraphDataModule
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__=='__main__':
    # Initialize W&B
    wandb.init(project='football-xg')
    
    graphs = read_data(data_path=PROCESSED_DATA_PATH)

    # Split the data into training and testing sets
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    data_module = GraphDataModule(train_graphs, test_graphs, batch_size=TRAIN_BATCH_SIZE)

    model = GNNModel(wandb.config.conv_type, wandb.config.conv1_size, wandb.config.conv2_size, wandb.config.fc1_size, wandb.config.activation)

    # log model only if `val_accuracy` increases
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = Trainer(max_epochs=MAX_EPOCHS, logger=wandb_logger, callbacks=[checkpoint_callback])

    trainer.fit(model, data_module)