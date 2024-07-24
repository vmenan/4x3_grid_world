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
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0)
}

PERPENDICULAR_ACTIONS = {
    "right": ["down", "up"],
    "down": ["right", "left"],
    "left": ["down", "up"],
    "up": ["right", "left"]
}


def f(u, n):
    return 2 if n <= 5 else u

def create_grid(pos_reward_cordinates,neg_reward_cordinates):
    grid = np.zeros((ROWS,COLS))
    for x,y in pos_reward_cordinates:
        grid[x][y] = 1
    for x,y in neg_reward_cordinates:
        grid[x][y] = -1
    return grid


def NextState(current_state, action, walls):
    i, j = current_state
    rand = random.random()
    if rand < 0.8:
        di, dj = ACTIONS[action]
        valid_action = action
    elif rand < 0.9:
        di, dj = ACTIONS[PERPENDICULAR_ACTIONS[action][0]]
        valid_action = PERPENDICULAR_ACTIONS[action][0]
    else:
        di, dj = ACTIONS[PERPENDICULAR_ACTIONS[action][1]]
        valid_action = PERPENDICULAR_ACTIONS[action][1]

    new_i, new_j = i + di, j + dj
    # Check boundaries and wall
    if new_i < 0 or new_i >= ROWS or new_j < 0 or new_j >= COLS or ((new_i, new_j) in walls):
        new_i, new_j = i, j
        # valid_action = "itself"
    return (new_i, new_j), valid_action

def calculate_transition_probabilites(policy, num_simulations, pos_reward_cordinates, neg_reward_cordinates, termination_states, walls):
    # Initialize count matrices for transitions and action counts
    transition_counts = {(i, j): {action: {} for action in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS)}
    action_counts = {(i, j): {action: 0 for action in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS)}
    
    # Initialize the counts to zero
    for state in transition_counts:
        for action in transition_counts[state]:
            for i in range(ROWS):
                for j in range(COLS):
                    transition_counts[state][action][(i, j)] = 0
            transition_counts[state][action][pos_reward_cordinates[0]] = 0
            transition_counts[state][action][neg_reward_cordinates[0]] = 0

    epsilon = 1.0
    min_epsilon = 0.1
    decay = 0.999

    current_state = (2, 0)

    stop_step = 10
    progress_bar = tqdm(range(num_simulations), desc="Simulations")
    for _ in range(num_simulations):
        max_steps = 0
        while current_state not in termination_states and max_steps != stop_step:
            if random.random() < epsilon:
                action = random.choice(list(ACTIONS.keys()))  # Exploration
            else:
                action = policy[current_state]  # Exploitation

            next_state, valid_action = NextState(current_state, action, walls)
            transition_counts[current_state][action][next_state] += 1
            action_counts[current_state][action] += 1
            current_state = next_state

            if current_state in termination_states:
                current_state = (2, 0)
                break
            max_steps += 1
        epsilon = max(min_epsilon, epsilon * decay)  # Decay epsilon over time
        progress_bar.update(1)

    # Calculate transition probabilities
    transition_probabilities = {(i, j): {action: {} for action in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS)}

    for state in transition_probabilities:
        for action in transition_probabilities[state]:
            total_transitions = sum(transition_counts[state][action].values())
            if total_transitions > 0:
                for next_state in transition_counts[state][action]:
                    transition_probabilities[state][action][next_state] = transition_counts[state][action][next_state] / total_transitions
    return transition_probabilities,action_counts


def calculate_utility(U,pos_reward_cordinates,neg_reward_cordinates):
    # Initialize utilities
    utilities = {(i, j): 0 for i in range(ROWS) for j in range(COLS)}
    for state in pos_reward_cordinates:
        utilities[state] = 1
    for state in neg_reward_cordinates:
        utilities[state] = -1
    optimistic_utilities = utilities.copy()

    while True:
        delta = 0
        new_utilities = optimistic_utilities.copy()
        for state in utilities:
            if state in termination_states or state == (1, 1):  # Skip terminal and wall states
                continue
            max_utility = float('-inf')
            for action in ACTIONS.keys():
                expected_utility = 0
                for next_state, prob in transition_probabilities[state][action].items():
                    expected_utility += prob * optimistic_utilities[next_state]
                if action_counts[state][action] > 0:
                    utility_with_action_count = f(expected_utility, action_counts[state][action])
                    max_utility = max(max_utility, utility_with_action_count)
            new_utilities[state] = REWARD + GAMMA * max_utility
            delta = max(delta, abs(new_utilities[state] - optimistic_utilities[state]))
        optimistic_utilities = new_utilities
        if delta < THRESHOLD:
            break

    for state in optimistic_utilities.keys():
        U[state] = optimistic_utilities[state]
    return U


def find_optimal_policy(U,excempt,walls):
    # Calculate the optimal policy
    policy = np.full((ROWS, COLS), ' ')

    for i in range(ROWS):
        for j in range(COLS):
            # Skip terminal states and the wall
            if (i,j) in excempt:
                continue
            
            # Calculate the best action
            best_action = None
            best_value = float('-inf')
            
            for action, (di, dj) in ACTIONS.items():
                new_i, new_j = i + di, j + dj
                
                # Boundary conditions and wall check for intended move
                if new_i < 0 or new_i >= ROWS or new_j < 0 or new_j >= COLS or ((new_i, new_j) in walls):
                    new_i, new_j = i, j
                
                utility = ACTION_PROB * U[new_i, new_j]
                
                # Add utilities for perpendicular moves
                for perp_action in PERPENDICULAR_ACTIONS[action]:
                    perp_i, perp_j = i + ACTIONS[perp_action][0], j + ACTIONS[perp_action][1]
                    
                    # Boundary conditions and wall check for perpendicular moves
                    if perp_i < 0 or perp_i >= ROWS or perp_j < 0 or perp_j >= COLS or ((perp_i, perp_j) in walls):
                        perp_i, perp_j = i, j
                    
                    utility += PERPENDICULAR_ACTIONS_PROB * U[perp_i, perp_j]
                
                if utility > best_value:
                    best_value = utility
                    best_action = action
            
            policy[i, j] = best_action


    return policy


def print_grid_policy(policy):
    mapping = {
        "r": "→",
        "d": "↓",

        "l": "←",
        "u": "↑",
        " ": " "
    }
    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) in pos_reward_cordinates:
                print(f"| +1 |", end=" ")
            elif (i, j) in neg_reward_cordinates:
                print(f"| -1 |", end=" ")
            elif (i, j) in walls:
                print(f"|  W  |", end=" ")
            else:
                print(f"|  {mapping[policy[i, j]]}  |", end=" ")
        print()


if __name__ == "__main__":
    excempt = list()
    num_simulations = 100_000
    pos_reward_cordinates = [(0, 3)]
    neg_reward_cordinates = [(1, 3)]
    termination_states = pos_reward_cordinates + neg_reward_cordinates
    walls = [(1, 1)]
    U = create_grid(pos_reward_cordinates, neg_reward_cordinates)
    excempt.extend(pos_reward_cordinates)
    excempt.extend(neg_reward_cordinates)
    excempt.extend(walls)
    policy = {
        (0, 0): "right",
        (0, 1): "right",
        (0, 2): "right",
        (1, 0): "up",
        (1, 2): "up",
        (2, 0): "up",
        (2, 1): "left",
        (2, 2): "left",
        (2, 3): "left"
    }

    transition_probabilities,action_counts = calculate_transition_probabilites(policy, num_simulations, pos_reward_cordinates, neg_reward_cordinates, termination_states, walls)

    U = calculate_utility(U,pos_reward_cordinates,neg_reward_cordinates)
    policy = find_optimal_policy(U,excempt,walls)
    print("=======================================Answer for Q4=======================================")
    # Print the utilities
    print("Utilities:")
    print(U)
    print("\nOptimal policy for each cell:")
    print_grid_policy(policy)