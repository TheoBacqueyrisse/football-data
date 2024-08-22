import wandb
import pandas as pd
from src.config import XGB_DATA_PATH, XGB_MODEL_PATH
import pygad
import xgboost as xgb
import numpy as np
import os
from src.modelling.genetic.functions import euclidean_distance, goal_player_angle
import warnings
from mplsoccer import Pitch
import seaborn as sns
import matplotlib.pyplot as plt
import ast

warnings.filterwarnings('ignore')

# Initialize Weights & Biases
wandb.init(project="genetic_algorithm_optimization")

# Log configuration to Weights & Biases
config = wandb.config

# Load the pre-trained XGBoost model
xgboost_model = xgb.XGBRegressor()
xgboost_model.load_model(XGB_MODEL_PATH)

num_tuples = 1 #x and y for shot
x_range = (80, 112)
y_range = (10, 70)

def custom_gene_initialization():
    x = np.random.uniform(low=x_range[0], high=x_range[1])
    y = np.random.uniform(low=y_range[0], high=y_range[1])
    return [x, y]

def custom_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        if np.random.uniform(0, 100) < config.mutation_percent_genes:
            gene_idx = np.random.randint(low=0, high=offspring.shape[1]//2) * 2
            if np.random.rand() < 0.5:
                offspring[idx, gene_idx] = np.random.uniform(low=x_range[0], high=x_range[1])
            else:
                offspring[idx, gene_idx + 1] = np.random.uniform(low=y_range[0], high=y_range[1])
    return offspring

def reshape_solution(solution):
    new_data = initial_data.copy(deep=True).to_frame().T
    tuple_list = list(zip(solution[::2], solution[1::2]))
    for i, (x, y) in enumerate(tuple_list):
        new_data['shot_x'] = x
        new_data['shot_y'] = y
    return new_data[initial_data.copy(deep=True).to_frame().T.columns]

def fitness_function(ga_class, solution, solution_idx):
    model_input = reshape_solution(solution)
    predicted_xg = xgboost_model.predict(model_input)
    return predicted_xg

# cols_to_keep = [
#     'shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y',
#     'pass_angle', 'distance_to_goal', 'distance_player_1', 
#     'distance_player_2', 'distance_player_3', 'distance_player_4', 
#     'angle_player_1', 'angle_player_2', 'angle_player_3', 
#     'angle_player_4', 'teammates_player_1', 'teammates_player_2', 
#     'teammates_player_3', 'teammates_player_4'
# ]
cols_to_keep = ['shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal'] 
k = 10
for i in range(1,k+1):
        cols_to_keep = cols_to_keep + [f'distance_player_{i}', f'angle_player_{i}', f'teammates_player_{i}']

max_xg_improvement = 0
best_iloc = 0
df = pd.read_csv(XGB_DATA_PATH, index_col=0)[cols_to_keep]

initial_data = df.iloc[5]
initial_xg = initial_data.shot_statsbomb_xg
initial_data = initial_data.drop('shot_statsbomb_xg')

initial_population = [[gene for _ in range(num_tuples) for gene in custom_gene_initialization()] for _ in range(config.sol_per_pop)]

ga_instance = pygad.GA(
    num_generations=config.num_generations,
    num_parents_mating=config.num_parents_mating,
    fitness_func=fitness_function,
    gene_type=object,
    initial_population=initial_population,
    mutation_type=custom_mutation,
    mutation_percent_genes=config.mutation_percent_genes,
    crossover_type=config.crossover_type,
    parent_selection_type=config.parent_selection_type,
    keep_parents=config.keep_parents,
)

ga_instance.run()
best_solution, best_solution_fitness, _ = ga_instance.best_solution()

improvement = best_solution_fitness - initial_xg
if improvement > max_xg_improvement:
    max_xg_improvement = improvement

print("Best Solution:", best_solution)
print("Best Fitness:", best_solution_fitness)

# Logging the metrics to wandb
wandb.log({
    'best_fitness': best_solution_fitness,
    'xg_improvement': improvement,
    'initial_xg': initial_xg
})
