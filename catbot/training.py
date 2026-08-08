import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def get_pos(state):
    catBot_x = state // 1000
    catBot_y = (state // 100) % 10
    cat_x = (state // 10) % 10
    cat_y = state % 10
    return catBot_x, catBot_y, cat_x, cat_y


def get_manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)



#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    start = time.time() # FOR DEBUGGING PURPOSES, REMOVE IN FINAL SUBMISSION (Make sure to run with --render -1)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    # Dictionary lookups are slower, so we change it to numpy array indexing
    q_table = np.zeros((10000, env.action_space.n))

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    learning_rate = 0.1 # Alpha
    exploration_rate = 1.0 # Epsilon
    discount_factor = 0.99 # Gamma
    exploration_decay = 0.999 # reverted to 0.999
    exploration_min = 0.01

    steps_per_episode = []
    success_per_episode = []
    reward_per_episode = []

    MAX_STEPS = 60
    best_steps = float('inf')  # Set to inf muna since no best yet

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 

        state, info = env.reset()
        finished = False
        cat_caught = False
        episode_steps = 0
        episode_reward = 0

        while not finished:
            # Action Phase
            if np.random.rand() < exploration_rate:
                action = np.random.randint(0, 4)  # only 0-3, skip stay, staying never helps catch a cat (most of the time)
            else:
                action = np.argmax(q_table[state])  

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1

            # 1. EVALUATE ALL TERMINATION CONDITIONS FIRST
            if terminated or truncated or (episode_steps >= MAX_STEPS):
                finished = True

            # 2. REWARD SHAPING
            catBot_x, catBot_y, cat_x, cat_y = get_pos(state)
            next_catBot_x, next_catBot_y, next_cat_x, next_cat_y = get_pos(next_state)
            current_distance = get_manhattan(catBot_x, catBot_y, cat_x, cat_y)
            next_distance = get_manhattan(next_catBot_x, next_catBot_y, next_cat_x, next_cat_y)

            reward = -1  # Standard step penalty

            # Bring back the breadcrumbs, but ONLY the positive ones
            if next_distance < current_distance:
                reward += 3

            # Overwrite reward if it's the final step
            if finished:
                if terminated:
                    cat_caught = True
                    reward += 100
                    if episode_steps < best_steps:
                        reward += 50
                        best_steps = episode_steps
                else:
                    # If we hit 60 steps (or truncated), apply the massive penalty
                    reward -= 50

            # 3. Q-TABLE UPDATE PHASE
            current_q_table = q_table[state][action]
            
            if finished:
                max_next_q_table = 0  
            else:
                max_next_q_table = np.max(q_table[next_state])

            new_q = current_q_table + learning_rate * (reward + discount_factor * max_next_q_table - current_q_table)
            q_table[state][action] = new_q

            # Move to next state
            state = next_state
            episode_reward += reward
            

        steps_per_episode.append(episode_steps)
        reward_per_episode.append(episode_reward)
        success_per_episode.append(1 if cat_caught else 0)

        #Epsilon decay 
        exploration_rate = max(exploration_min, exploration_rate * exploration_decay)

        if ep % 100 == 0:
            avg_steps = np.mean(steps_per_episode[-100:]) if steps_per_episode else 0
            success_rate = np.mean(success_per_episode[-100:]) * 100

            print(f"Episode {ep}: "
                  f"Avg Steps={avg_steps:.1f}, "
                  f"Success={success_rate:.0f}%, "
                  f"Best={best_steps if best_steps != float('inf') else 'N/A'}, "
                  f"Epsilon={exploration_rate:.3f}")
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

        print(f"Training time: {time.time() - start:.2f} seconds") # FOR DEBUGGING PURPOSES, REMOVE IN FINAL SUBMISSION (Make sure to run with --render -1)
    return q_table