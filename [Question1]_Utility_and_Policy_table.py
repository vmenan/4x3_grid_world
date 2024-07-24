import numpy as np

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



def create_grid(pos_reward_cordinates,neg_reward_cordinates):
    grid = np.zeros((ROWS,COLS))
    for x,y in pos_reward_cordinates:
        grid[x][y] = 1
    for x,y in neg_reward_cordinates:
        grid[x][y] = -1
    return grid


def calculate_utility(U,excempt,walls):
    while True:
        U_new = np.copy(U)
        delta = 0
        
        for i in range(ROWS):
            for j in range(COLS):
                # Skip terminal states and the wall
                if (i,j) in excempt:
                    continue
                
    # Calculate utility for each action
                action_values = []
                for action, (di, dj) in ACTIONS.items():
                    new_i, new_j = i + di, j + dj
                    
                    # Boundary conditions and wall check for intended move
                    if new_i < 0 or new_i >= ROWS or new_j < 0 or new_j >= COLS or ((new_i, new_j) in walls):
                        new_i, new_j = i, j
                    
                    utility =ACTION_PROB * U[new_i, new_j]
                    
                    # Add utilities for perpendicular moves
                    for perp_action in PERPENDICULAR_ACTIONS[action]:
                        perp_i, perp_j = i + ACTIONS[perp_action][0], j + ACTIONS[perp_action][1]
                        
                        # Boundary conditions and wall check for perpendicular moves
                        if perp_i < 0 or perp_i >= ROWS or perp_j < 0 or perp_j >= COLS or ((perp_i, perp_j) in walls):
                            perp_i, perp_j = i, j
                        
                        utility += PERPENDICULAR_ACTIONS_PROB * U[perp_i, perp_j]
                    
                    action_values.append(utility)
                
                # Bellman update
                U_new[i, j] = REWARD + GAMMA * max(action_values)
                delta = max(delta, abs(U_new[i, j] - U[i, j]))
        
        U = U_new
        # Check for convergence
        if delta < THRESHOLD:
            break
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
    pos_reward_cordinates = [(0, 3)]
    neg_reward_cordinates = [(1, 3)]
    walls = [(1,1)]
    U = create_grid(pos_reward_cordinates,neg_reward_cordinates)
    excempt.extend(pos_reward_cordinates)
    excempt.extend(neg_reward_cordinates)
    excempt.extend(walls)
    U = calculate_utility(U,excempt,walls)
    print("=======================================Answer for Q1 Using Value Iteration=======================================")
    print("[Answer for Table 1] Utility values for each cell:")
    print(U)
    policy = find_optimal_policy(U,excempt,walls)
    print("\nOptimal policy for each cell:")
    print_grid_policy(policy)