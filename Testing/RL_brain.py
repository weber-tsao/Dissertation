"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
from Puf_delay_model import Puf

UNIT = 40   # pixels
MAZE_H = 2  # grid height
#MAZE_W = 10  # grid width

class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.3):
        self.actions = action_space  # a list
        self.puf = Puf()
        self.reward_dict = self.puf.reward_related_to_delay()
        self.maze_w = int(len(self.reward_dict)/2)+2
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        #print(self.q_table)
        self.state_num = self.maze_w
        #row = [[], [], [], [], [], []]
        self.state_action_permit = pd.DataFrame()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        #print(state)
        #print(self.q_table)

    def choose_action(self, observation, observation_):
        self.check_state_exist(observation)
        # action selection
        #print(np.random.rand())
        #print("dddd {}".format(observation_))
        #print(type(observation_[0]))
        
        action_choose = []
        #print(observation_ == 'terminal')
        if observation_ != 'terminal':
            if (observation_[0] + UNIT) == (5+UNIT*(self.state_num-1)) and (observation_[1] + UNIT) == 45:
                action_choose.append(2)
                print("straight")
            elif (observation_[0] + UNIT) == (5+UNIT*(self.state_num-1)) and observation_[1] == 45:
                action_choose.append(0)
                print("cross up")
            else:
                # cross up
                if observation_[1] > UNIT and observation_[0] < (self.maze_w - 1) * UNIT:
                    action_choose.append(0)
                    print("cross up")
                # cross down
                if observation_[1] < (MAZE_H - 1) * UNIT and observation_[0] < (self.maze_w - 1) * UNIT:
                    action_choose.append(1)
                    print("cross down")
                # straight
                if observation_[0] < (self.maze_w - 1) * UNIT:
                    action_choose.append(2)
                    print("straight")
        else:
            action_choose = [0, 1, 2]
            print("straight")
            
        #action = np.random.choice(action_choose)
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, action_choose]
            #print(self.q_table)
            #print(state_action[2])
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            #print(action)
        else:
            # choose random action
            action = np.random.choice(action_choose)
            #print(action)
            
        return action

    def learn(self, *args):
        pass


# off-policy
'''class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.5):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print(self.q_table)


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.5):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        print(self.q_table)
        #print("---------------------------------")'''
        
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_
        #print(self.q_table)
    
    def get_q_table(self):
        q_table = self.q_table
        return q_table
