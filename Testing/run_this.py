"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaLambdaTable
from Puf_delay_model import*

def update():
    for episode in range(150):
        # initial observation
        observation = env.reset()
        #print(str(observation))
        # RL choose action based on observation
        action = RL.choose_action(str(observation), observation)
        #print(str(observation))
        print("Episode start")
        print("Current episode node {}".format(observation))
        print("Episode next action {}".format(action))
        
        #lambda
        RL.eligibility_trace *= 0
        
        while True:
            # fresh env
            print("Loop since not reach terminal")
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print("Update to episode node to {}".format(observation_))

            #print(env.step(action))
            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_), observation_)
            print("The new node next behaviour {}".format(action_))
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

def prediction_rate(q_table, nodes_location, actions):
    # method to calculate RL model accuracy
    # input path and calculate the sum of reward and compare that to label
    # run many times and see how many times we got it right
    puf = Puf()
    testing_crps = puf.testing_crps_for_RL()
    success = 0
    crps_size = len(testing_crps)
    #print(crps_size)
    
    for i in range(len(testing_crps)):
        current_challenge = array([testing_crps[i][0]])
        current_response_array = testing_crps[i][1]
        current_response = current_response_array[0][0]
        current_top_path, current_bottom_path = puf.puf_path(current_challenge)
        #print(current_challenge)
        #print(current_response)
        #print(current_top_path)
        #print(current_bottom_path)
        #print(nodes_location)
        top_reward = 0
        bottom_reward = 0
        for x in range(len(current_top_path)):
            if current_top_path[x] == 0:
                top_reward += q_table.loc[str(nodes_location[0])][2]
            elif current_top_path[x] == 1:
                top_reward += q_table.loc[str(nodes_location[0])][1]
            elif current_top_path[x-1]%2 == 0 and (current_top_path[x])%2 == 0:
                top_reward += q_table.loc[str(nodes_location[current_top_path[x]])][2]
            elif current_top_path[x-1]%2 != 0 and (current_top_path[x])%2 == 0:
                top_reward += q_table.loc[str(nodes_location[current_top_path[x]+1])][0]
            elif current_top_path[x-1]%2 == 0 and (current_top_path[x])%2 != 0:
                top_reward += q_table.loc[str(nodes_location[current_top_path[x]+1])][1]
            elif current_top_path[x-1]%2 != 0 and (current_top_path[x])%2 != 0:
                top_reward += q_table.loc[str(nodes_location[current_top_path[x]])][2]
            
        for y in range(len(current_bottom_path)):
            if current_bottom_path[y] == 0:
                bottom_reward += q_table.loc[str(nodes_location[1])][0]
            elif current_bottom_path[y] == 1:
                bottom_reward += q_table.loc[str(nodes_location[1])][2]
            elif current_bottom_path[y]%2 == 0 and (current_bottom_path[y-1])%2 == 0:
                bottom_reward += q_table.loc[str(nodes_location[current_bottom_path[y]])][2]
            elif current_bottom_path[y]%2 != 0 and (current_bottom_path[y-1])%2 == 0:
                bottom_reward += q_table.loc[str(nodes_location[current_bottom_path[y]+1])][1]
            elif current_bottom_path[y]%2 == 0 and (current_bottom_path[y-1])%2 != 0:
                bottom_reward += q_table.loc[str(nodes_location[current_bottom_path[y]+1])][0]
            elif current_bottom_path[y]%2 != 0 and (current_bottom_path[y-1])%2 != 0:
                bottom_reward += q_table.loc[str(nodes_location[current_bottom_path[y]])][2]
        #print(bottom_reward)

        #Compare top and bottom
        if bottom_reward > top_reward:
            label = -1
            #print("bottom fast")
        elif bottom_reward < top_reward:
            label = 1
            #print("top fast")
        else:
            print("speed equal")


        # check label is right and add a count to calculate the prediction rate 
        if label == current_response:
            success += 1

    accuracy = success/crps_size         
    print("prediction rate = {}".format(accuracy))

if __name__ == "__main__":
    #prediction_rate()
    env = Maze()
    nodes_location, actions = env.get_maze_information()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    env.after(150, update)
    env.mainloop()
    q_table = RL.get_q_table()
    print(q_table)
    prediction_rate(q_table, nodes_location, actions)