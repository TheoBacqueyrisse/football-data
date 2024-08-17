import wandb
from src.modelling.gnn import GNNModel
import torch
from sklearn.metrics import mean_squared_error, r2_score
from src.config import PROCESSED_DATA_PATH, PREPROCESSOR_PATH, TEST_BATCH_SIZE
from sklearn.model_selection import train_test_split
from src.dataset import read_data
import matplotlib.pyplot as plt
import tqdm
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
import os
import wandb

api = wandb.Api()
sweep = api.sweep(f"football-data/sweeps/890adt0c")

# Get best run parameters
best_run = sweep.best_run()#order='validation/loss')
best_parameters = best_run.config
# Perform inference

# List all artifacts used or logged by this run
artifacts = best_run.logged_artifacts()

# Iterate over artifacts to find the one with the tag 'best'
for artifact in artifacts:
    if 'best' in artifact.aliases:  # Check if the artifact has the 'best' tag
        print(f"Found 'best' artifact: {artifact.name}")
        # Download the artifact
        artifact_dir = artifact.download()
        print(f"Best artifact downloaded to {artifact_dir}")
        break
else:
    print("No artifact with the 'best' tag found.")


model = GNNModel.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), **best_parameters)

def perform_inference(loader, model):
    model.eval()
    preds = []
    actuals = []
    for data in tqdm.tqdm(loader):
        data = data.to(model.device)
        with torch.no_grad():
            output = model(data)
            preds.append(output.detach().cpu().numpy())
            actuals.append(data.y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    return preds, actuals

# Get predictions and actual values
graphs = read_data(data_path=PROCESSED_DATA_PATH)

    # Split the data into training and testing sets
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

inference_loader = DataLoader(test_graphs, batch_size=TEST_BATCH_SIZE, shuffle=False)

y_pred, y_test_orig = perform_inference(inference_loader, model)

# Evaluate the model
with open(PREPROCESSOR_PATH, 'rb') as f:
    boxcox_transformer = pickle.load(f)

y_pred = boxcox_transformer.inverse_transform(y_pred.reshape(-1,1)).ravel()
y_test_orig = boxcox_transformer.inverse_transform(y_test_orig.reshape(-1,1)).ravel()

mse = mean_squared_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot y_test_orig vs y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Expected Goals (y_test)')
plt.ylabel('Predicted Expected Goals (y_pred)')
plt.title('Actual vs Predicted Expected Goals')
plt.legend()
plt.show()