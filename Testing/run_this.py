"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable
from RL_brain import SarsaLambdaTable
from Puf_delay_model import*

def update():
    for episode in range(500):
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

if __name__ == "__main__":
    env = Maze()
    RL1 = SarsaTable(actions=list(range(env.n_actions)))
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(500, update)
    env.mainloop()