import numpy as np
import random
import pandas as pd
from tqdm import tqdm

# Parameters
GAMMA = 0.9
REWARD = -0.04
THRESHOLD = 1e-4
ACTION_PROB = 0.8
PERPENDICULAR_ACTIONS_PROB = 0.1
# Grid dimensions
ROWS, COLS = 3, 4

ACTIONS = {
    "right": (0, 1),
    "down": (-1, 0),
    "left": (0, -1),
    "up": (1, 0)
}

PERPENDICULAR_ACTIONS = {
    "right": ["down", "up"],
    "down": ["right", "left"],
    "left": ["down", "up"],
    "up": ["right", "left"]
}

def NextState(current_state, action, walls):
    i,j = current_state
    rand = random.random()
    if rand < 0.8:
        di,dj = ACTIONS[action]
        valid_action = action
    elif rand < 0.9:
        di,dj = ACTIONS[PERPENDICULAR_ACTIONS[action][0]]
        valid_action = PERPENDICULAR_ACTIONS[action][0]
    else:
        di,dj = ACTIONS[PERPENDICULAR_ACTIONS[action][1]]
        valid_action = PERPENDICULAR_ACTIONS[action][1]
    
    new_i,new_j = i+di,j+dj
    # Check boundaries and wall
    if new_i < 0 or new_i >= ROWS or new_j < 0 or new_j >= COLS or ((new_i, new_j) in walls):
        new_i, new_j = i, j
        valid_action = "itself"
    return (new_i,new_j),valid_action




if __name__ == "__main__":
    num_simulations = 10000000
    pos_reward_cordinates = [(0, 3)]
    neg_reward_cordinates = [(1, 3)]
    termination_states = pos_reward_cordinates + neg_reward_cordinates
    walls = [(1,1)]
    policy = {
        (0,0):"right",
        (0,1):"right",
        (0,2):"right",
        (1,0):"up",
        (1,2):"up",
        (2,0):"up",
        (2,1):"left",
        (2,2):"left",
        (2,3):"left"
    }       
    # Initialize count matrices for transitions
    transition_counts = {(i, j): {action: {} for action in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS)} 
    # Initialize the counts to zero
    for state in transition_counts:
        for action in transition_counts[state]:
            for i in range(ROWS):
                for j in range(COLS):
                    transition_counts[state][action][(i, j)] = 0
            for pos_ter in pos_reward_cordinates:
                transition_counts[state][action][pos_ter] = 0
            for neg_ter in neg_reward_cordinates:
                transition_counts[state][action][neg_ter] = 0
    
    current_state = (2, 0)
    max_steps = 0
    stop_step = 1000
    progress_bar = tqdm(range(num_simulations), desc="Simulations")
    for _ in range(num_simulations):
        while (current_state not in termination_states) and (max_steps!=stop_step):
            if current_state in policy:
                action = policy[current_state]
                next_state, _ = NextState(current_state, action, walls)
                transition_counts[current_state][action][next_state] += 1
                current_state = next_state
            
            # Reset to start state if current state is terminal
            if current_state in termination_states:
                current_state = (2, 0)
            max_steps += 1
        progress_bar.update(1)
    # Calculate transition probabilities
    transition_probabilities = {(i, j): {action: {} for action in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS)}    


    for state in transition_probabilities:
        for action in transition_probabilities[state]:
            total_transitions = sum(transition_counts[state][action].values())
            if total_transitions > 0:
                for next_state in transition_counts[state][action]:
                    transition_probabilities[state][action][next_state] = transition_counts[state][action][next_state] / total_transitions
    
    # Print the transition probabilities
    for state in transition_probabilities:
        for action in transition_probabilities[state]:
            print(f"State: {state}, Action: {action}, Transition Probabilities: {transition_probabilities[state][action]}")