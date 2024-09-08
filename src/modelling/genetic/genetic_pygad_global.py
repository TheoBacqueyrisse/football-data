import pandas as pd
import sys
import os
sys.path.append('./')

from src.config import XGB_DATA_PATH, XGB_MODEL_PATH, GENETIC_RESULT_PATH
import pygad
import xgboost as xgb
import numpy as np
import os
from src.modelling.genetic.functions import euclidean_distance, goal_player_angle, calculate_position
import warnings
from mplsoccer import Pitch
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import wandb
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')

SWEEP_ID = "thomas-toulouse/football-data-src_modelling_genetic/jfr8m8x0"
SHOW_PLOT = True
XGB_DATA_PATH_2 = os.path.join('data', 'processed', 'clean_action_data_glob.csv')
MAX_RADIUS = 5

def custom_gene_initialization(x, y, max_radius=5):
    """Function to initialize each gene within a circle of a certain radius."""

    # for distance, angle in zip(*[iter(base_population[:2])]*2):
                          
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(-max_radius, max_radius)

    new_x = x + radius * np.cos(angle)
    new_y = y + radius * np.sin(angle)

    # Ensure the coordinates are within the specified x and y ranges
    # x = np.clip(x, x - MAX_RADIUS, x + MAX_RADIUS)
    # y = np.clip(y, y - MAX_RADIUS, y + MAX_RADIUS)

    new_x, new_y = clip_to_circle(new_x, new_y, x, y, max_radius)
    return [new_x, new_y]

def clip_to_circle(x, y, cx, cy, radius):
    # Calculate the distance from the center of the circle
    dx = x - cx
    dy = y - cy
    distance = math.sqrt(dx**2 + dy**2)
    
    # If the point is inside or on the circle, no need to clip
    if distance <= radius:
        return x, y
    
    # If the point is outside the circle, scale the coordinates back to the circle's boundary
    scale = radius / distance
    clipped_x = cx + dx * scale
    clipped_y = cy + dy * scale

    # print(f'clipped_x: {clipped_x}, clipped_y: {clipped_y}\nbase_x: {x}, base_y: {y}')
    return clipped_x, clipped_y

def custom_mutation(offspring, ga_instance, original_population):
    """Custom mutation function to mutate the genes relative to the original values."""
    for idx in range(offspring.shape[0]):
        if np.random.uniform(0, 100) < ga_instance.mutation_percent_genes:
            # Select a gene to mutate
            gene_idx = np.random.randint(low=0, high=offspring.shape[1] // 2) * 2

            base_x = original_population[0, gene_idx] #0 to have x and y from initial population (real position)
            base_y = original_population[0, gene_idx + 1]
  
            x_values = original_population[0][0::2]

            # Finding the maximum x-value
            x_max = np.max(x_values)
            # Use the original population value for mutation
            original_x = offspring[idx, gene_idx]
            original_y = offspring[idx, gene_idx + 1]
           
            # Apply the mutation based on the original values
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(-MAX_RADIUS, MAX_RADIUS)
            
         
            mutated_x = original_x + radius * np.cos(angle)
            mutated_y = original_y + radius * np.sin(angle)

            new_x, new_y = clip_to_circle(mutated_x, mutated_y, base_x, base_y, MAX_RADIUS)

            if gene_idx == 0: #To prevent from offside
                new_x = np.clip(new_x, 0, x_max)

            offspring[idx, gene_idx] = new_x
            offspring[idx, gene_idx + 1] = new_y
            # if gene_idx == 8 :
            #     print(f'For gene index : {gene_idx}, base_x: {base_x}, base_y: {base_y}, original_x: {original_x}, original_y: {original_y}, new_x: {new_x}, new_y: {new_y}')

    return offspring


# Function to reshape the solution into the desired format for the XGBoost model
def reshape_solution(solution):
    new_data = initial_data.copy(deep=True).to_frame().T

    tuple_list = list(zip(solution[::2], solution[1::2]))

    for i, (x, y) in enumerate(tuple_list):
        if i == 0:
            new_data['shot_x'] = x
            new_data['shot_y'] = y
            distance_to_goal = euclidean_distance((120, 40), (new_data['shot_x'].item(), new_data['shot_y'].item()))
            pass_angle = goal_player_angle((new_data['fk_x'].item(), new_data['fk_y'].item()), (new_data['shot_x'].item(), new_data['shot_y'].item()), use_goal=False)

            new_data['distance_to_goal'] = distance_to_goal
            new_data['pass_angle'] = pass_angle
        else:
            distance = euclidean_distance((x, y), (new_data['shot_x'].item(), new_data['shot_y'].item()))
            angle = goal_player_angle((x, y), (new_data['shot_x'].item(), new_data['shot_y'].item()))
            new_data[f'distance_{teammates[i-1]}'] = distance
            new_data[f'angle_{teammates[i-1]}'] = angle

    new_data = new_data[initial_data.copy(deep=True).to_frame().T.columns]
    return new_data

# Fitness function
def fitness_function(ga_class, solution, solution_idx):
    # Reshape the solution to fit the expected input of the XGBoost model
    model_input = reshape_solution(solution)
    
    # Get the feature names from the training data
    feature_names = xgboost_model.feature_names_in_

    # Get the feature names from the input data

    # Check if the feature names are consistent
    if len(feature_names) != len(model_input.columns):
        raise ValueError("Feature names mismatch")

    predicted_xg = xgboost_model.predict(model_input)
    
    return predicted_xg

def get_teammates(data):
    teammates_with_1 = data[data.index.str.startswith('teammates_player') & (data == 1)]
    columns_with_1 = teammates_with_1.index.tolist()
    return columns_with_1

def complete_list(lst):
    diff = len(df) - len(lst)

    # add None to the list
    lst.extend([None] * diff)

    return lst


def get_real_x_y():
    freekick_data = pd.read_csv(os.path.join('data', 'raw', 'freekick_pass_shot.csv'), index_col=0)
    freekick_data_2 = pd.read_csv(XGB_DATA_PATH)

    # Find the row where the shot_statsbomb_xg value matches your initial_xg value
    matching_row = freekick_data[freekick_data['shot_statsbomb_xg'] == true_xg]
    matching_row_2 = freekick_data_2[freekick_data_2['shot_statsbomb_xg'] == true_xg]

    freeze_frame = ast.literal_eval(matching_row['shot_freeze_frame'].values[0])
    shot_x = matching_row_2['shot_x'].values[0]
    shot_y = matching_row_2['shot_y'].values[0]
    # new_shot_x, new_shot_y = best_solution[:2]
    fk_x = matching_row_2['fk_x'].values[0]
    fk_y = matching_row_2['fk_y'].values[0]

    # Track the initial and optimized positions
    initial_positions_teammates = {'x': [], 'y': [], 'names': []}
    optimized_positions_teammates = {'x': [], 'y': []}
    initial_positions_non_teammates = {'x': [], 'y': [], 'names': []}

    # Extract initial positions from the freeze frame
    players = freeze_frame
    for player in players:
        player['distance'] = euclidean_distance(player['location'], (shot_x, shot_y))
        player['angle'] = goal_player_angle(player['location'], (shot_x, shot_y), use_goal=True)

    sorted_players = sorted(players, key=lambda x: x['distance'])
    
    for player in sorted_players:
        x = player['location'][0]
        y = player['location'][1]
        if player['teammate']:
            initial_positions_teammates['x'].append(x)
            initial_positions_teammates['y'].append(y)
            initial_positions_teammates['names'].append(player['position']['id'])
        else:
            initial_positions_non_teammates['x'].append(x)
            initial_positions_non_teammates['y'].append(y)
            initial_positions_non_teammates['names'].append(player['position']['id'])

    return shot_x, shot_y, fk_x, fk_y, initial_positions_teammates, initial_positions_non_teammates, optimized_positions_teammates


if __name__ == '__main__':
    
    cols_to_keep = ['shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal'] 
    k = 10
    for i in range(1,k+1):
        cols_to_keep = cols_to_keep + [f'distance_player_{i}', f'angle_player_{i}', f'teammates_player_{i}']
    # init better improvement 
    max = 0
    best_iloc = 0
    df_base = pd.read_csv(XGB_DATA_PATH_2)

    df = df_base[cols_to_keep]
    if 'pctge_improvement' in df_base.columns:
        overall_improv_list = list(filter(lambda x: not math.isnan(x), df_base['pctge_improvement'].to_list()))
        overall_fitness_list = list(filter(lambda x: not math.isnan(x), df_base['pred_improved_xg'].to_list()))
        overall_basexg = list(filter(lambda x: not math.isnan(x), df_base['pred_base_xg'].to_list()))
    else:
        overall_improv_list = []
        overall_fitness_list = []
        overall_basexg = []


    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(XGB_MODEL_PATH)  # Replace with the actual path to your model
    num_tuples = 1 #x and y for shot
    x_range = (80, 115)  # Example range for x
    y_range = (10, 70)   # Example range for y
    api = wandb.Api()    
    sweep = api.sweep(SWEEP_ID)
    best_run = sweep.best_run()
    best_parameters = best_run.config

    if not overall_improv_list:
        index_to_start = 0
    else:
        index_to_start = len(overall_improv_list)
        
    print("Starting from index :", index_to_start)

    for idx in tqdm(range(index_to_start, len(df))): 
        df_base = pd.read_csv(XGB_DATA_PATH_2)
        if 'pctge_improvement' in df_base.columns:
            overall_improv_list = list(filter(lambda x: not math.isnan(x), df_base['pctge_improvement'].to_list()))
            overall_fitness_list = list(filter(lambda x: not math.isnan(x), df_base['pred_improved_xg'].to_list()))
            overall_basexg = list(filter(lambda x: not math.isnan(x), df_base['pred_base_xg'].to_list()))
        else:
            overall_improv_list = []
            overall_fitness_list = []
            overall_basexg = []

        print("iloc :", idx)
        
        initial_data = df.iloc[idx]
        true_xg = df_base.iloc[idx]['shot_statsbomb_xg']
        initial_data = initial_data.drop('shot_statsbomb_xg')
        initial_xg = xgboost_model.predict(initial_data.to_frame().T)[0]
        

        teammates = get_teammates(initial_data)
        teammates = ['_'.join(teammate.split('_')[1:]) for teammate in teammates]
        shot_x, shot_y, fk_x, fk_y, initial_positions_teammates, initial_positions_non_teammates, optimized_positions_teammates = get_real_x_y()
    
        # Prepare the initial population
        base_population = [initial_data["shot_x"], initial_data["shot_y"]]
        # base_population = []
        for i,teammate in enumerate(teammates):
            x = initial_positions_teammates['x'][i]
            y = initial_positions_teammates['y'][i]
            # x, y = calculate_position((initial_data["shot_x"], initial_data["shot_y"]), initial_data[f'distance_{teammate}'], initial_data[f'angle_{teammate}'])
            base_population.append(x)
            base_population.append(y)

        initial_population = [base_population]
        
        for _ in range(best_parameters['sol_per_pop'] - 1):
            initial_gene = base_population[:2]
            for i in range(2,len(base_population),2):
                gene = custom_gene_initialization(base_population[i], base_population[i+1], max_radius=MAX_RADIUS)

                initial_gene.extend(gene)
            initial_population.append(initial_gene)
        print(f'Initial population : ({initial_population[0]})\nInitial xG : {initial_xg}')


        # Initialize the genetic algorithm
        # initial_population = [[initial_data["shot_x"], initial_data["shot_y"]]]
        # initial_population.extend([[gene for _ in range(num_tuples) for gene in custom_gene_initialization()] for _ in range(best_parameters['sol_per_pop']-1)])

        # Genetic Algorithm Parameters
        ga_instance = pygad.GA(
            num_generations=best_parameters['num_generations'],
            num_parents_mating=best_parameters['num_parents_mating'],
            fitness_func=fitness_function,
            gene_type=object,
            initial_population=initial_population,
            mutation_type=lambda offspring, ga_instance: custom_mutation(offspring, ga_instance, np.array(initial_population)),
            mutation_percent_genes=best_parameters['mutation_percent_genes'],
            crossover_type=best_parameters['crossover_type'],
            parent_selection_type=best_parameters['parent_selection_type'],
            keep_parents=best_parameters['keep_parents'],
        )
        

        # Run the Genetic Algorithm
        ga_instance.run()

        # After the algorithm completes, get the best solution
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        percent_improv = round(((best_solution_fitness - initial_xg)/initial_xg * 100)[0],2)
        print(f"Best Solution:{[round(x,2) for x in best_solution]}")
        print("Best Fitness:", best_solution_fitness[0])
        print(f"Improvement in %:  {percent_improv} %")

        overall_basexg.append(initial_xg)
        overall_improv_list.append(percent_improv)
        overall_fitness_list.append(best_solution_fitness[0])
        # Convert the best solution to a readable format (e.g., player positions)
        best_positions = reshape_solution(best_solution)
        # print("Optimal Player Positions:")
        best_positions['pred_base_xg'] = [initial_xg]
        best_positions['pred_improved_xg'] = [best_solution_fitness[0]]
        best_positions['pctge_improvement'] = [percent_improv]
        try:
            df_best_positions = pd.read_csv(GENETIC_RESULT_PATH)
            df_best_positions = pd.concat([df_best_positions, best_positions], ignore_index=True)
        except FileNotFoundError:
            df_best_positions = best_positions
        
        df_best_positions.to_csv(GENETIC_RESULT_PATH, index=False)

        if percent_improv > max:
            max = percent_improv
            best_iloc=i

        if SHOW_PLOT:
            new_shot_x, new_shot_y = best_solution[:2]

            from matplotlib.patches import Circle
                    # Load the freekick_pass_shot.csv file
            # Extract optimized positions from the best solution
            for i in range(0, len(best_solution[2:]), 2):
                optimized_positions_teammates['x'].append(best_solution[2 + i])
                optimized_positions_teammates['y'].append(best_solution[3 + i])

            # Draw the pitch
            pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
            fig, ax = pitch.draw()

            # Plot the initial positions of teammates
            sns.scatterplot(x=initial_positions_teammates['x'], y=initial_positions_teammates['y'], ax=ax, s=25, color='blue', label='Initial Position (Teammates)', marker='o')

            # Plot the optimized positions of teammates
            sns.scatterplot(x=optimized_positions_teammates['x'], y=optimized_positions_teammates['y'], ax=ax, s=25, color='red', label='Optimized Position (Teammates)', marker='X')

            # Plot the initial positions of non-teammates
            sns.scatterplot(x=initial_positions_non_teammates['x'], y=initial_positions_non_teammates['y'], ax=ax, s=25, color='orange', label='Initial Position (Non-Teammates)', marker='o')

            # Plot the initial shot position, freekick position, and optimized shot position
            sns.scatterplot(x=[shot_x], y=[shot_y], ax=ax, s=50, color='black', label='Initial Shot Position', marker='P', alpha=0.75)
            sns.scatterplot(x=[fk_x], y=[fk_y], ax=ax, s=50, color='purple', label='Freekick Position', marker='D', alpha=0.75)
            sns.scatterplot(x=[new_shot_x], y=[new_shot_y], ax=ax, s=50, color='white', label='Optimized Shot Position', marker='P', alpha=0.75)

            for x, y in zip(initial_positions_teammates['x'], initial_positions_teammates['y']):
                circle = Circle((x, y), MAX_RADIUS, color='blue', alpha=0.2, fill=True)
                ax.add_patch(circle)
            # Annotate the initial positions of teammates
            # for i in range(len(initial_positions_teammates['x'])):
            #     ax.annotate(initial_positions_teammates['names'][i], 
            #                 (initial_positions_teammates['x'][i], initial_positions_teammates['y'][i]), 
            #                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='blue')

            # # Annotate the initial positions of non-teammates
            # for i in range(len(initial_positions_non_teammates['x'])):
            #     ax.annotate(initial_positions_non_teammates['names'][i], 
            #                 (initial_positions_non_teammates['x'][i], initial_positions_non_teammates['y'][i]), 
            #                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='orange')

            # # Annotate the optimized positions of teammates
            # for i in range(len(optimized_positions_teammates['x'])):
            #     ax.annotate(f"Opt_{initial_positions_teammates['names'][i]}", 
            #                 (optimized_positions_teammates['x'][i], optimized_positions_teammates['y'][i]), 
            #                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='red')

            # Display the legend to differentiate between initial and optimized positions
            ax.legend(fontsize=8, loc='upper left')
            plt.savefig(f"{os.path.join('src','modelling','genetic','results', str(idx))}.png", format="png")

        print(f"Maximum Improvement: {max} %")
        print(f"Best Iloc: {best_iloc}")
        print('\n\n')
        df_base['pred_base_xg'] = complete_list(overall_basexg)
        df_base['pred_improved_xg'] = complete_list(overall_fitness_list)
        df_base['pctge_improvement'] = complete_list(overall_improv_list)

        df_base.to_csv(XGB_DATA_PATH_2, index = False)

        # with open(os.path.join('data','results','overall_improv.txt'), 'w') as f:
        #     for item in overall_improv_list:
        #         f.write("%s\n" % item)

        # with open(os.path.join('data','results','overall_fitness.txt'), 'w') as f: 
        #     for item in overall_fitness_list:
        #         f.write("%s\n" % item)
