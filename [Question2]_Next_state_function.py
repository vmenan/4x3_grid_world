import numpy as np
import random
import pandas as pd

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


def test_NextState(test_dict,walls,num_tests=100):
    for state_action in test_dict["state_action"]:
        temp_counter = {
            "itself":0,
            "up":0,
            "down":0,
            "right":0,
            "left":0
        }
        for i in range(num_tests):
            next_state,valid_action = NextState(state_action[0],state_action[1],walls)
            temp_counter[valid_action] += 1
        for key in temp_counter.keys():
            test_dict[key].append(temp_counter[key]/num_tests)
    return test_dict




if __name__ == "__main__":
    excempt = list()
    pos_reward_cordinates = [(0, 3)]
    neg_reward_cordinates = [(1, 3)]
    walls = [(1,1)]
    test_dict = {
        "state_action":[((0,0),"right"),((0,0),"up"),((1,2),"down"),((1,2),"left"),((0,2),"left")],
        "itself":[],
        "up":[],
        "down":[],
        "right":[],
        "left":[]
    }
    test_dict = test_NextState(test_dict,walls,100)
    df = pd.DataFrame(test_dict)
    print(df)
