"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
from Puf_delay_model import Puf

UNIT = 40   # pixels
MAZE_H = 2  # grid height
#MAZE_W = 10  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.puf = Puf()
        self.reward_dict = self.puf.reward_related_to_delay()
        self.maze_w = int(len(self.reward_dict)/2)+2
        self.action_space = ['cross_up', 'cross_down', 'straight']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.n_features = 2
        #self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.all_nodes_coord = []
        self._build_maze()
        self.change = 0
        self.reward = 0
        self.done = False
        
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=self.maze_w * UNIT)
            
        # create origin
        origin = np.array([20, 20])
        self.all_nodes = []
        
        for i in range(self.maze_w):
            
            if i == 0:
                node_center = origin + np.array([i, i])
                self.node = self.canvas.create_oval(
                    node_center[0] - 15, node_center[1] - 15,
                    node_center[0] + 15, node_center[1] + 15,
                    fill='green')
                
                node_down_center = origin + np.array([i, UNIT])
                self.node_down = self.canvas.create_oval(
                    node_down_center[0] - 15, node_down_center[1] - 15,
                    node_down_center[0] + 15, node_down_center[1] + 15,
                    fill='green')
                
                self.all_nodes.append(self.node)
                self.all_nodes.append(self.node_down)
            elif i == (self.maze_w-1):
                oval_center = origin + np.array([UNIT*i, 0])
                self.oval = self.canvas.create_oval(
                    oval_center[0] - 15, oval_center[1] - 15,
                    oval_center[0] + 15, oval_center[1] + 15,
                    fill='yellow')
                self.all_nodes.append(self.oval)
            else:
                node_center = origin + np.array([UNIT*i, 0])
                self.node = self.canvas.create_rectangle(
                    node_center[0] - 15, node_center[1] - 15,
                    node_center[0] + 15, node_center[1] + 15,
                    fill='black')
                
                node_down_center = origin + np.array([UNIT*i, UNIT])
                self.node_down = self.canvas.create_rectangle(
                    node_down_center[0] - 15, node_down_center[1] - 15,
                    node_down_center[0] + 15, node_down_center[1] + 15,
                    fill='black')
                
                self.all_nodes.append(self.node)
                self.all_nodes.append(self.node_down)
        
        # create red oval
        self.rect = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()
        
        for i in range(len(self.all_nodes)):
            self.all_nodes_coord.append(self.canvas.coords(self.all_nodes[i]))
                

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        
        origin = np.array([20, 20])
        '''node1_center = origin + np.array([0, UNIT])
        self.rect = self.canvas.create_oval(
            node1_center[0] - 15, node1_center[1] - 15,
            node1_center[0] + 15, node1_center[1] + 15,
            fill='red')'''
        if self.change == 0:
            self.change = 1
            node1_center = origin + np.array([0, 0])
            self.rect = self.canvas.create_oval(
                node1_center[0] - 15, node1_center[1] - 15,
                node1_center[0] + 15, node1_center[1] + 15,
                fill='red')
        else:
            self.change = 0
            node2_center = origin + np.array([0, UNIT])
            self.rect = self.canvas.create_oval(
                node2_center[0] - 15, node2_center[1] - 15,
                node2_center[0] + 15, node2_center[1] + 15,
                fill='red')
        # return observation
        #print(np.array(self.canvas.coords(self.oval)[:2]))
        #return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.node)[:2]))/(MAZE_H*UNIT)
        #print(self.canvas.coords(self.node)[:2])
        print(np.array(self.canvas.coords(self.node)[:2]))
        return np.array(self.canvas.coords(self.node)[:2])

    def step(self, action):
        s = self.canvas.coords(self.rect)
        #print("ddd {}".format(s))
        base_action = np.array([0, 0])
        if action == 0:   # cross up
            if s[1] > UNIT and s[0] < (self.maze_w - 1) * UNIT:
                base_action[0] += UNIT
                base_action[1] -= UNIT
                #print("cross up")
        elif action == 1:   # cross down
            if s[1] < (MAZE_H - 1) * UNIT and s[0] < (self.maze_w - 1) * UNIT:
                base_action[0] += UNIT
                base_action[1] += UNIT
                #print("cross down")
        elif action == 2:   # straight
            if s[0] < (self.maze_w - 1) * UNIT:
                base_action[0] += UNIT
                #print("straight")

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state
        #print(self.canvas.coords(self.rect))
        #print("Next state {}".format(s_))
        #print(self.all_nodes[4])
        # reward function
        if s_ == self.canvas.coords(self.all_nodes[-1]):
            self.reward = 0.0
            self.done = True
            #s_ = ['terminal', 'terminal']

        
        for i in range(2, (len(self.all_nodes)-1), 1):
            if s_ == self.canvas.coords(self.all_nodes[i]):
                self.reward = self.reward_dict[str(i-2)]
                self.done = False
                #print(np.array(s_[:2]))
        #print((np.array(s_[:2]) - np.array(self.canvas.coords(self.all_nodes[-1])[:2])))
        #s_ = (np.array(s_[:2]) - np.array(self.canvas.coords(self.all_nodes[-1])[:2]))/(MAZE_H*UNIT)
        #print(np.array(s_[:2]))
        return np.array(s_[:2]), self.reward, self.done

    def render(self):
        time.sleep(0.1)
        self.update()
        
    def get_maze_information(self):
        nodes_location = self.all_nodes_coord
        actions  = self.action_space
        return nodes_location, actions


